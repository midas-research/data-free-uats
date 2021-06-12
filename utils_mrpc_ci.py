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
# extracted_forward_embs = []

def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])
    
# def extract_forward_hook(module, inputs, outputs):
#     extracted_forward_embs.append(outputs[0].detach().cpu().numpy())
    
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

#         if module[0] == 'albert.pooler':
#             module[1].register_forward_hook(extract_forward_hook)

# +
# def get_inputs(data, tokenizer):
#     sent1 = data['sentence1']
#     sent2 = data['sentence2']
#     inp1 = tokenizer.convert_tokens_to_ids(sent1.split())
#     inp2 = tokenizer.convert_tokens_to_ids(sent2.split())
# #     print(inp1, inp2)
#     return_d = {}
#     return_d['input_ids'] = torch.tensor([[2]+inp1+[3]+inp2+[3]]).to("cuda:1")
#     return_d['token_type_ids'] = torch.tensor([[0]*(len(inp1)+2) + [1]*(len(inp2)+1)]).to("cuda:1")
#     return_d['attention_mask'] = torch.tensor([[1]*(len(inp1)+len(inp2)+3)]).to("cuda:1")
#     label = data['label']
#     return {'inputs': return_d, 'label':label}

def get_inputs(data, tokenizer, length):
    sent1 = data['sentence1']
    sent2 = data['sentence2']
    encoded_pair = tokenizer(sent1, sent2, return_tensors='pt', \
                             padding =True, max_length = (2*length+3), truncation = 'longest_first').to('cuda:1') 

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
    logit_label = torch.zeros(1, 2).to("cuda:1")
    logit_label[range(1), original_label]=1
    
    output_label = model(**inputs)
    pred = output_label.logits
    loss = bce_classification(pred, logit_label)
    output_dict={'logits': pred, 'loss': loss}
    return output_dict
    
def evaluate_batch(model, data, trig, tokenizer, length):
    input_dict = get_inputs(deepcopy(data), tokenizer, length)
    output_dict = get_output_dict(model, input_dict['inputs'], input_dict['label'])
    return output_dict


# -

def bce_classification(preds, orig):
    loss = torch.nn.BCEWithLogitsLoss()
    return loss(preds, orig)


def get_accuracy(model, data, tokenizer, trig_len):
    print(data)
    input_dict = get_inputs(deepcopy(data), tokenizer, trig_len)
    print(input_dict)
    output_label = model(**input_dict['inputs'])
    preds = output_label.logits.detach().cpu().numpy()
    preds = np.argmax(preds, axis=1)[0]
    orig = input_dict['label']
    return accuracy_score([orig], [preds])

def get_average_grad(model, batch, trigger_token_ids, trig_len, tokenizer):
    optimizer = optim.Adam(model.parameters())
    optimizer.zero_grad()
    global extracted_grads
    extracted_grads = [] # clear existing stored grads
    loss = evaluate_batch(model, batch, '', tokenizer, trig_len)['loss']
    loss.backward()
    grads = extracted_grads[0].cpu()
    averaged_grad = torch.sum(grads, dim=0)
    averaged_grad_sent1 = averaged_grad[1:1+trig_len] # return just trigger grads
    averaged_grad_sent2 = averaged_grad[trig_len+2:-1] # return just trigger grads
    return averaged_grad_sent1, averaged_grad_sent2

# +
# def get_forwards(model, dataset):
#     global extracted_forward_embs
#     extracted_forward_embs = [] # clear existing stored grads
#     outs = []
#     for ind in (dataset.data.index):
#         inputs, label = dataset[ind]
#         output_label = model(**inputs)
#         output_dict = {}
# #         output_dict['sent1'] = dataset.data.loc[ind, 'sentence1']
# #         output_dict['sent2'] = dataset.data.loc[ind, 'sentence2']
#         output_dict['embedding'] = extracted_forward_embs[-1]
#         output_dict['label'] = label
        
#         outs.append(output_dict)
#     return outs

# +
def get_best_candidates(model, x_batch, candidates, tokenizer, trig_len, beam_size=1,\
                        increase_loss=False, field='sentence1'):
    if increase_loss:
        beamer = heapq.nlargest
    else:
        beamer = heapq.nsmallest
    
    loss_per_candidate = get_loss_per_candidate(0, model, x_batch, None, candidates, trig_len, tokenizer, field)
    rand_ind = random.randint(0,beam_size-1)
    top_candidates = [beamer(beam_size, loss_per_candidate, key=itemgetter(1))[rand_ind]]
    trigger_token_ids = x_batch[field].split()
    for idx in range(1, len(trigger_token_ids[:trig_len])):
        loss_per_candidate = []
        for cand, _ in top_candidates: 
            loss_per_candidate.extend(get_loss_per_candidate(idx, model, x_batch, cand, candidates, trig_len, tokenizer, field))
        top_candidates = [beamer(beam_size, loss_per_candidate, key=itemgetter(1))[rand_ind]]
#         print(len(top_candidates[0][0]))
    if increase_loss:
        output = max(top_candidates, key=itemgetter(1))
    else:
        output = min(top_candidates, key=itemgetter(1))
    return output[0], output[1] 
    
def gen_adv(data: Dict, trigger="", col = 'sentence1'):
    data[col] = trigger
    return data


# -

def get_loss_per_candidate(index, model, batch, trigger, cand_trigger_token_ids, trig_len, tokenizer, field):
    """
    For a particular index, the function tries all of the candidate tokens for that index.
    The function returns a list containing the candidate triggers it tried, along with their loss.
    """
    loss_per_candidate = []
    if trigger!='' and trigger!=None:
        x_adv = gen_adv(deepcopy(batch), trigger= ' '.join(trigger) , col = field)
    else:
        x_adv = batch
        
    curr_loss = evaluate_batch(model, x_adv, '', tokenizer, trig_len)['loss'].cpu().detach().numpy()
    loss_per_candidate.append((deepcopy(x_adv[field].split()), curr_loss))
    try:
        for cand_id in range(len(cand_trigger_token_ids[0])):
            trigger_token_ids_one_replaced = deepcopy(x_adv[field].split()[:trig_len])
    #         trigger_token_ids_one_replaced[index] = tokenizer.convert_ids_to_tokens(int(cand_trigger_token_ids[index][cand_id]))
            print(index, len(trigger_token_ids_one_replaced), len(cand_trigger_token_ids),  cand_id)
            trigger_token_ids_one_replaced[index] = tokenizer.decode(int(cand_trigger_token_ids[index][cand_id]))
            x_adv = gen_adv(batch, trigger= ' '.join(trigger_token_ids_one_replaced), col = field)
            loss = evaluate_batch(model, x_adv, '', tokenizer, trig_len)['loss'].cpu().detach().numpy()
            loss_per_candidate.append((deepcopy(trigger_token_ids_one_replaced), loss))
    except Exception as e:
        print('paased')
        pass
    return loss_per_candidate
