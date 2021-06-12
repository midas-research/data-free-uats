from operator import itemgetter
from copy import deepcopy
import heapq
import numpy
import torch
import torch.optim as optim
import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score
from keras.preprocessing.sequence import pad_sequences
import random
random.seed(123)
import copy
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


# -

def get_average_grad(model, batch, trigger_token_ids, trig_len, tokenizer):
    optimizer = optim.Adam(model.parameters())
    optimizer.zero_grad()
    global extracted_grads
    extracted_grads = [] # clear existing stored grads
    loss = evaluate_batch(model, batch, trigger_token_ids, tokenizer)['loss']
    print(loss)
    loss.backward()
    grads = extracted_grads[0].cpu()
    print(grads)
    averaged_grad = torch.sum(grads, dim=0)
    print(averaged_grad)
    averaged_grad_sent1 = averaged_grad[1:1+trig_len]
    return averaged_grad_sent1

# +
def bce_classification(preds, orig):
    loss = torch.nn.BCEWithLogitsLoss()
    return loss(preds, orig)

def get_inputs(data, tokenizer, trigger = None):
    if trigger!=None:
        sent1 = trigger+' '+data['sentence1'].astype(str)
    else:
        sent1 = data['sentence1'].astype(str)
    sent2 = data['sentence2'].astype(str)
    return_d = {}
    inputs = []
    att_masks = []
    tt_ids = []
    max_len = 128
    for s1,s2 in zip(sent1.tolist(), sent2.tolist()):
        inp1 = tokenizer.convert_tokens_to_ids(s1.split())
        inp2 = tokenizer.convert_tokens_to_ids(s2.split())
        l = len(inp1)+len(inp2)+3
        inputs.append([2]+inp1+[3]+inp2+[3] + [0]*(max_len-l))
        att_masks.append([0]*(len(inp1)+2) + [1]*(len(inp2)+1) + [0]*(max_len-l))
        tt_ids.append([1]*(len(inp1)+len(inp2)+3) + [0]*(max_len-l))
    
#     inputs = pad_sequences(inputs, maxlen=128, truncating="post", padding="post", dtype="int")
#     attn_masks = pad_sequences(att_masks, maxlen=128, truncating="post", padding="post", dtype="int")
#     tt_ids = pad_sequences(tt_ids, maxlen=128, truncating="post", padding="post", dtype="int")

#     batch = tokenizer(sent1.tolist(), sent2.tolist(), padding=True, truncation=True, return_tensors="pt")
    return_d['input_ids']      = torch.tensor(inputs).to("cuda:1")
    return_d['token_type_ids'] = torch.tensor(tt_ids).to("cuda:1")
    return_d['attention_mask'] = torch.tensor(att_masks).to("cuda:1")

    label = data['label'].tolist()
    return {'inputs': return_d, 'label':label}

def get_output_dict(model, inputs, label):
    original_label = deepcopy(label[0])
    logit_label = torch.zeros(len(label), 2).to("cuda:1")
    logit_label[range(len(label)), original_label]=1
    
    output_label = model(**inputs)
    pred = output_label.logits
    loss = bce_classification(pred, logit_label)
    output_dict={'logits': pred, 'loss': loss}
    return output_dict
    
def evaluate_batch(model, data, trig, tokenizer):
    if trig == None or trig == '':
        input_dict = get_inputs(data, tokenizer, trigger = None)    
    else:
        input_dict = get_inputs(data, tokenizer, trigger = trig)
        
    output_dict = get_output_dict(model, input_dict['inputs'], input_dict['label'])
    return output_dict


# +
def get_best_candidates(model, x_batch, trigger_token_ids, candidates, tokenizer, beam_size=1,\
                        increase_loss=False, field='sentence1'):
    if increase_loss:
        beamer = heapq.nlargest
    else:
        beamer = heapq.nsmallest
    
    loss_per_candidate = get_loss_per_candidate(0, model, copy.deepcopy(x_batch), trigger_token_ids, candidates, tokenizer, field)
    rand_ind = random.randint(0,beam_size-1)
    top_candidates = [beamer(beam_size, loss_per_candidate, key=itemgetter(1))[rand_ind]]
    for idx in range(1, len(trigger_token_ids)):
        loss_per_candidate = []
        for cand, _ in top_candidates: 
            loss_per_candidate.extend(get_loss_per_candidate(idx, model, copy.deepcopy(x_batch), cand, candidates, tokenizer, field))
        top_candidates = [beamer(beam_size, loss_per_candidate, key=itemgetter(1))[rand_ind]]
    if increase_loss:
        output = max(top_candidates, key=itemgetter(1))
    else:
        output = min(top_candidates, key=itemgetter(1))
    return output[0], output[1] 
    
# def gen_adv(data, trigger="", col = 'sentence1'):
#     data[col] = (trigger +" " )+ data[col].astype(str)
#     return data


# -

def get_loss_per_candidate(index, model, batch, trigger_token_ids, cand_trigger_token_ids, tokenizer, field):
    """
    For a particular index, the function tries all of the candidate tokens for that index.
    The function returns a list containing the candidate triggers it tried, along with their loss.
    """
    loss_per_candidate = []

    curr_loss = evaluate_batch(model, batch, ' '.join(trigger_token_ids), tokenizer)['loss'].cpu().detach().numpy()
    loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))
    
    for cand_id in range(len(cand_trigger_token_ids[0])):
        trigger_token_ids_one_replaced = deepcopy(trigger_token_ids)
        trigger_token_ids_one_replaced[index] = tokenizer.convert_ids_to_tokens(int(cand_trigger_token_ids[index][cand_id]))
        loss = evaluate_batch(model, batch, ' '.join(trigger_token_ids_one_replaced), tokenizer)['loss'].cpu().detach().numpy()
        loss_per_candidate.append((deepcopy(trigger_token_ids_one_replaced), loss))
    return loss_per_candidate
