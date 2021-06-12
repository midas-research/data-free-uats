from operator import itemgetter
from copy import deepcopy
import heapq
import numpy
import torch
import random
random.seed(42)
import torch.optim as optim
from allennlp.common.util import lazy_groups_of
from allennlp.data.iterators import BucketIterator
from allennlp.nn.util import move_to_device
from allennlp.modules.text_field_embedders import TextFieldEmbedder

# +
def get_embedding_weight(model):
    """
    Extracts and returns the token embedding weight matrix from the model.
    """
    for module in model.modules():
        if isinstance(module, TextFieldEmbedder):
            for embed in module._token_embedders.keys():
                embedding_weight = module._token_embedders[embed].weight.cpu().detach()
    return embedding_weight

# def get_embedding_weight_DA(model):
#     """
#     Extracts and returns the token embedding weight matrix from the model.
#     """
#     for module in model.modules():
#         if isinstance(module, TextFieldEmbedder):
#             for embed in module._token_embedders.keys():
#                 embedding_weight = module._token_embedders[embed].weight.cpu().detach()
#         return module


# -

# hook used in add_hooks()
extracted_grads = []
def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])

def add_hooks(model):
    """
    Finds the token embedding matrix on the model and registers a hook onto it.
    When loss.backward() is called, extracted_grads list will be filled with
    the gradients w.r.t. the token embeddings
    """
    for module in model.modules():
        if isinstance(module, TextFieldEmbedder):
            for embed in module._token_embedders.keys():
                module._token_embedders[embed].weight.requires_grad = True
            module.register_backward_hook(extract_grad_hook)

def get_accuracy(model, dev_dataset, vocab, trigger_token_ids=None, snli=False):
    """
    When trigger_token_ids is None, gets accuracy on the dev_dataset. Otherwise, gets accuracy with
    triggers prepended for the whole dev_dataset.
    """
    model.get_metrics(reset=True)
    model.eval() # model should be in eval() already, but just in case
    
    iterator = BucketIterator(batch_size=1, sorting_keys=[("premise", "num_tokens")])
    iterator.index_with(vocab)
    if trigger_token_ids is None:
        for batch in lazy_groups_of(iterator(dev_dataset, num_epochs=1, shuffle=False), group_size=1):
            evaluate_batch(model, batch, trigger_token_ids, snli)
        print("Without Triggers: " + str(model.get_metrics()['accuracy']))
    else:
        print_string = ""
        for idx in trigger_token_ids:
            print_string = print_string + vocab.get_token_from_index(idx) + ', '

        for batch in lazy_groups_of(iterator(dev_dataset, num_epochs=1, shuffle=False), group_size=1):
            evaluate_batch(model, batch, trigger_token_ids, snli)
        print("Current Triggers: " + print_string + " : " + str(model.get_metrics()['accuracy']))
    return model.get_metrics()['accuracy']

def evaluate_batch(model, batch, trigger_token_ids=None, snli=False, field = None):
    """
    Takes a batch of classification examples (SNLI or SST), and runs them through the model.
    If trigger_token_ids is not None, then it will append the tokens to the input.
    This funtion is used to get the model's accuracy and/or the loss with/without the trigger.
    """
    batch = move_to_device(batch[0], cuda_device=0)
    trigger_token_ids = move_to_device(trigger_token_ids, cuda_device=0)
    
    if trigger_token_ids is None:
        model(batch['premise'], batch['hypothesis'], batch['label'])
        return None
    else:
        if isinstance(trigger_token_ids, dict):
            original_tokens_hypo = batch['hypothesis']['tokens'].clone()
            original_tokens_premise = batch['premise']['tokens'].clone()
            batch['hypothesis']['tokens'] = deepcopy(trigger_token_ids['hypothesis']['tokens'])
            batch['premise']['tokens'] = deepcopy(trigger_token_ids['premise']['tokens'])
            output_dict = model(batch['premise'], batch['hypothesis'], batch['label'])
            batch['hypothesis']['tokens'] = original_tokens_hypo
            batch['premise']['tokens'] = original_tokens_premise
        
        else:
            original_tokens_hypo = batch['hypothesis']['tokens'].clone()
            original_tokens_premise = batch['premise']['tokens'].clone()
            
            batch['hypothesis']['tokens'] = deepcopy(torch.tensor([trigger_token_ids[0]]).cuda())
            batch['premise']['tokens'] = deepcopy(torch.tensor([trigger_token_ids[1]]).cuda())
            output_dict = model(batch['premise'], batch['hypothesis'], batch['label'])
            batch['hypothesis']['tokens'] = original_tokens_hypo
            batch['premise']['tokens'] = original_tokens_premise
        
        return output_dict

def get_average_grad(model, batch, trigger_token_ids, target_label=None, snli=False, is_df = False):
    """
    Computes the average gradient w.r.t. the trigger tokens when prepended to every example
    in the batch. If target_label is set, that is used as the ground-truth label.
    """
    # create an dummy optimizer for backprop
    optimizer = optim.Adam(model.parameters())
    optimizer.zero_grad()

    # prepend triggers to the batch
    original_labels = batch[0]['label'].clone()
    if target_label is not None:
        # set the labels equal to the target (backprop from the target class, not model prediction)
        batch[0]['label'] = int(target_label) * torch.ones_like(batch[0]['label']).cuda()
    global extracted_grads
    extracted_grads = [] # clear existing stored grads
    loss = evaluate_batch(model, batch, trigger_token_ids, snli)['loss']
    loss.backward()

    # index 0 has the hypothesis grads for SNLI. For SST, the list is of size 1.
    grads_hypo = extracted_grads[0].cpu()
    grads_premise = extracted_grads[1].cpu()
    
    batch[0]['label'] = original_labels # reset labels

    # average grad across batch size, result only makes sense for trigger tokens at the front
    averaged_grad_hypo = torch.sum(grads_hypo, dim=0)
    averaged_grad_premise = torch.sum(grads_premise, dim=0)

    l_hypo, l_premise = get_trigger_length(trigger_token_ids)
    
    averaged_grad_hypo = averaged_grad_hypo[:l_hypo-1] # return just trigger grads
    averaged_grad_premise = averaged_grad_premise[:l_premise-1] # return just trigger grads
    return averaged_grad_hypo, averaged_grad_premise


def get_trigger_length(trigger_token_ids):
    return len(trigger_token_ids['hypothesis']['tokens'][0]), len(trigger_token_ids['premise']['tokens'][0])


# +
def get_best_candidates(model, batch, trigger_token_ids, cand_trigger_token_ids, \
                        snli=True, beam_size=1, increase_loss=False, label_value = 'entailment', field='hypothesis'):
    """"
    Given the list of candidate trigger token ids (of number of trigger words by number of candidates
    per word), it finds the best new candidate trigger.
    This performs beam search in a left to right fashion.
    """
    # first round, no beams, just get the loss for each of the candidates in index 0.
    # (indices 1-end are just the old trigger)
    if increase_loss:
        beamer = heapq.nlargest
    else:
        beamer = heapq.nsmallest
    loss_per_candidate = get_loss_per_candidate(0, model, batch, trigger_token_ids,
                                                cand_trigger_token_ids, snli, label_value, field)
    # maximize the loss
    rand_ind = random.randint(0,beam_size-1)
#     rand_ind = 0
    top_candidates = [beamer(beam_size, loss_per_candidate, key=itemgetter(1))[rand_ind]]
    l_hypo, l_premise = get_trigger_length(trigger_token_ids)
    if field == 'hypothesis':
        l = l_hypo
    if field == 'premise':
        l = l_premise
    # top_candidates now contains beam_size trigger sequences, each with a different 0th token
    for idx in range(1, l-1): # for all trigger tokens, skipping the 0th (we did it above)
        loss_per_candidate = []
        for cand, _ in top_candidates: # for all the beams, try all the candidates at idx
            if field=='hypothesis':
                loss_per_candidate.extend(get_loss_per_candidate(idx, model, batch, [cand, trigger_token_ids['premise']['tokens'][0]],
                                                                 cand_trigger_token_ids, snli, label_value, field))
            else:
                loss_per_candidate.extend(get_loss_per_candidate(idx, model, batch, [trigger_token_ids['hypothesis']['tokens'][0], cand],
                                                                 cand_trigger_token_ids, snli, label_value, field))
#         print(loss_per_candidate)
        top_candidates = [beamer(beam_size, loss_per_candidate, key=itemgetter(1))[rand_ind]]
        
#     print((top_candidates))
    if increase_loss:
        output = max(top_candidates, key=itemgetter(1))
    else:
        output = min(top_candidates, key=itemgetter(1))
    return output[0], output[1]


# -

def get_loss_per_candidate(index, model, batch, trigger_token_ids, cand_trigger_token_ids,\
                           snli=False, label_value = 'entailment', field = 'hypothesis'):
    """
    For a particular index, the function tries all of the candidate tokens for that index.
    The function returns a list containing the candidate triggers it tried, along with their loss.
    """
    if isinstance(cand_trigger_token_ids[0], (numpy.int64, int)):
        print("Only 1 candidate for index detected, not searching")
        return trigger_token_ids
    model.get_metrics(reset=True)
    loss_per_candidate = []
    
    # loss for the trigger without trying the candidates
    if isinstance(trigger_token_ids, dict):
        if field == 'hypothesis':
            trig = trigger_token_ids['hypothesis']['tokens'][0].detach().cpu().numpy()
            other = trigger_token_ids['premise']['tokens'][0].detach().cpu().numpy()
            curr_loss = evaluate_batch(model, batch, [trig, other], snli)['loss'].cpu().detach().numpy()
    
        else:
            trig = trigger_token_ids['premise']['tokens'][0].detach().cpu().numpy()
            other = trigger_token_ids['hypothesis']['tokens'][0].detach().cpu().numpy()
            curr_loss = evaluate_batch(model, batch, [other, trig], snli)['loss'].cpu().detach().numpy()
    else:
        if field == 'hypothesis':
            trig = trigger_token_ids[0]
            other = trigger_token_ids[1].detach().cpu().numpy()
            curr_loss = evaluate_batch(model, batch, [trig, other], snli)['loss'].cpu().detach().numpy()
        else:
            trig = trigger_token_ids[1]
            other = trigger_token_ids[0].detach().cpu().numpy()
            curr_loss = evaluate_batch(model, batch, [other, trig], snli)['loss'].cpu().detach().numpy()
            
    loss_per_candidate.append((deepcopy(trig), curr_loss))
    for cand_id in range(len(cand_trigger_token_ids[0])):
        trigger_token_ids_one_replaced = deepcopy(trig) # copy trigger
        trigger_token_ids_one_replaced[index] = cand_trigger_token_ids[index][cand_id] # replace one token
        if field == 'hypothesis':
            loss = evaluate_batch(model, batch, [trigger_token_ids_one_replaced, other], snli)['loss'].cpu().detach().numpy()
        else:
            loss = evaluate_batch(model, batch, [other, trigger_token_ids_one_replaced], snli)['loss'].cpu().detach().numpy()
            
        loss_per_candidate.append((deepcopy(trigger_token_ids_one_replaced), loss))
    return loss_per_candidate
