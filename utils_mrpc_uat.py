from operator import itemgetter
from copy import deepcopy
import heapq
import numpy
import torch
import torch.optim as optim
import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score
import random
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

# +
extracted_grads = []

def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])

def get_embedding_weight(language_model):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == 30000: # only add a hook to wordpiece embeddings, not position embeddings
                return module.weight.detach()

# add hooks for embeddings
def add_hooks(language_model):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == 30000: # only add a hook to wordpiece embeddings, not position
                module.weight.requires_grad = True
                module.register_backward_hook(extract_grad_hook)

# +
def get_inputs(data, tokenizer, trigger):
    if trigger == None:
        sent1 = data['sentence1']
    else:
        sent1 = [trigger +" "+x for x in data['sentence1']]
    sent2 = data['sentence2']
#     inp1 = tokenizer.convert_tokens_to_ids(sent1.split())
#     inp2 = tokenizer.convert_tokens_to_ids(sent2.split())
# #     print(inp1, inp2)
#     return_d = {}
#     l = len(inp1)+len(inp2)+3
#     max_len = 128
#     return_d['input_ids'] = torch.tensor([[2]+inp1+[3]+inp2+[3] + [0]*(max_len-l)]).to("cuda:1")
#     return_d['token_type_ids'] = torch.tensor([[0]*(len(inp1)+2) + [1]*(len(inp2)+1) + [0]*(max_len-l)]).to("cuda:1")
#     return_d['attention_mask'] = torch.tensor([[1]*(len(inp1)+len(inp2)+3) + [0]*(max_len-l)]).to("cuda:1")
#     label = data['label']

    encoded_pair = tokenizer(sent1, sent2, return_tensors='pt', \
                             padding =True).to('cuda:1') 

    token_ids = encoded_pair['input_ids']  # tensor of token ids
    attn_masks = encoded_pair['attention_mask'] # binary tensor with "0" for padded values and "1" for the other values
    token_type_ids = encoded_pair['token_type_ids']  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

    return_d = {}
    return_d['input_ids'] =token_ids
    return_d['token_type_ids'] = token_type_ids
    return_d['attention_mask'] = attn_masks
    label = data['label']
    return {'inputs': return_d, 'label':label}

def get_output_dict(model, inputs, label):
    original_label = deepcopy(label)
    logit_label = torch.zeros(len(label), 2).cuda(1)
    logit_label[range(len(label)), original_label]=1
    
    output_label = model(**inputs)
    pred = output_label.logits
    loss = bce_classification(pred, logit_label)
    output_dict={'logits': pred, 'loss': loss}
    return output_dict
    
def evaluate_batch(model, data, trig, tokenizer):
    input_dict = get_inputs(deepcopy(data), tokenizer, trig)
    output_dict = get_output_dict(model, input_dict['inputs'], input_dict['label'])
    return output_dict


# -

def bce_classification(preds, orig):
    loss = torch.nn.BCEWithLogitsLoss()
    return loss(preds, orig)


# +
# def get_accuracy(model, data, tokenizer):
#     input_dict = get_inputs(deepcopy(data), tokenizer)
#     output_label = model(**input_dict['inputs'])
#     preds = output_label.logits.detach().cpu().numpy()
#     preds = np.argmax(preds, axis=1)[0]
#     orig = input_dict['label']
#     return accuracy_score([orig], [preds])
# -

def get_average_grad(model, batch, trigger_token_ids, trig_len, tokenizer):
    optimizer = optim.Adam(model.parameters())
    optimizer.zero_grad()
    global extracted_grads
    extracted_grads = [] # clear existing stored grads
    loss = evaluate_batch(model, batch, trigger_token_ids, tokenizer)['loss']
    loss.backward()
    grads = extracted_grads[0].cpu()
    averaged_grad = torch.sum(grads, dim=0)
    averaged_grad_sent1 = averaged_grad[1:1+trig_len] # return just trigger grads
    averaged_grad_sent2 = averaged_grad[trig_len+2:-1] # return just trigger grads
    return averaged_grad_sent1, averaged_grad_sent2

def get_best_candidates(model, x_batch, trigger_ids, candidates, tokenizer, beam_size=1,\
                        increase_loss=False, field='sentence1'):
    if increase_loss:
        beamer = heapq.nlargest
    else:
        beamer = heapq.nsmallest
    
    loss_per_candidate = get_loss_per_candidate(0, model, x_batch, trigger_ids, candidates, tokenizer, field)
    rand_ind = random.randint(0,beam_size-1)
    top_candidates = [beamer(beam_size, loss_per_candidate, key=itemgetter(1))[rand_ind]]
    for idx in range(1, len(trigger_ids)):
        loss_per_candidate = []
        for cand, _ in top_candidates: 
            loss_per_candidate.extend(get_loss_per_candidate(idx, model, x_batch, cand, candidates, tokenizer, field))
        top_candidates = [beamer(beam_size, loss_per_candidate, key=itemgetter(1))[rand_ind]]
    if increase_loss:
        output = max(top_candidates, key=itemgetter(1))
    else:
        output = min(top_candidates, key=itemgetter(1))
    return output[0], output[1] 


def get_loss_per_candidate(index, model, batch, trigger, cand_trigger_token_ids, tokenizer, field):
    """
    For a particular index, the function tries all of the candidate tokens for that index.
    The function returns a list containing the candidate triggers it tried, along with their loss.
    """
    loss_per_candidate = []
#     print(batch)
    curr_loss = evaluate_batch(model, batch, ' '.join(trigger), tokenizer)['loss'].cpu().detach().numpy()
    loss_per_candidate.append((deepcopy(trigger), curr_loss))
    
    for cand_id in range(len(cand_trigger_token_ids[0])):
        trigger_token_ids_one_replaced = deepcopy(trigger)
        trigger_token_ids_one_replaced[index] = tokenizer.decode(int(cand_trigger_token_ids[index][cand_id]))
        loss = evaluate_batch(model, batch, ' '.join(trigger_token_ids_one_replaced), tokenizer)['loss'].cpu().detach().numpy()
        loss_per_candidate.append((deepcopy(trigger_token_ids_one_replaced), loss))
    return loss_per_candidate
