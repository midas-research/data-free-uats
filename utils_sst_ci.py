from operator import itemgetter
from copy import deepcopy
import heapq
import numpy
import torch
import torch.optim as optim
from allennlp.common.util import lazy_groups_of
from allennlp.data.iterators import BucketIterator
from allennlp.nn.util import move_to_device
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.modules.text_field_embedders import (
    BasicTextFieldEmbedder,
)
from allennlp.nn.util import move_to_device
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.elmo import _ElmoCharacterEncoder
import random
from allennlp.data.tokenizers import Token
from typing import List, Dict

def get_embedding_weight(model):
    """
    Extracts and returns the token embedding weight matrix from the model.
    """
    for module in model.modules():
#         if isinstance(module, TextFieldEmbedder):
#             for embed in module._token_embedders.keys():
#                 print(module._token_embedders[embed])
#                 embedding_weight = module._token_embedders[embed].weight.cpu().detach()
                
                
        if isinstance(module, TextFieldEmbedder): 
            if isinstance(module, BasicTextFieldEmbedder): 
                if len(module._token_embedders) == 1: 
                    for module_elmo in module._token_embedders['tokens'].modules():
                        if isinstance(module_elmo, _ElmoCharacterEncoder):
                            return module_elmo
#             return embedding_weight

# hook used in add_hooks()
extracted_grads = []
def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])

def make_embedder_input(all_tokens: List[str], vocab, reader) -> Dict[str, torch.Tensor]:
    inputs = {}
    indexers = reader._token_indexers  # type: ignore
    for indexer_name, token_indexer in indexers.items():
        print(token_indexer)
        if isinstance(token_indexer, ELMoTokenCharactersIndexer):
            elmo_tokens = []
            for token in all_tokens:
                elmo_indexed_token = token_indexer.tokens_to_indices(
                    [Token(text=token)], vocab
                )["elmo_tokens"]
                elmo_tokens.append(elmo_indexed_token[0])
            inputs[indexer_name] = {"elmo_tokens": torch.LongTensor(elmo_tokens).unsqueeze(0)}
        else:
            raise RuntimeError("Unsupported token indexer:", token_indexer)

    return move_to_device(inputs, 0)


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

def create_empty_batch(model, batch, trigger_token_ids=None, snli=False):
    batch = move_to_device(batch, cuda_device=0)
    trigger_sequence_tensor = torch.LongTensor(deepcopy(trigger_token_ids))
    trigger_sequence_tensor = trigger_sequence_tensor.repeat(len(batch['label']), 1).cuda()
    batch['tokens']['tokens'] = trigger_sequence_tensor
    return batch


def get_accuracy(model, dev_dataset, vocab, trigger_token_ids=None, snli=False):
    
    model.get_metrics(reset=True)
    model.eval() 
    if trigger_token_ids is None:
        evaluate_batch(model, dev_dataset, trigger_token_ids, snli)
        print("Without Triggers: " + str(model.get_metrics()['accuracy']))
    else:
        print_string = ""
        for idx in trigger_token_ids:
            print_string = print_string + vocab.get_token_from_index(idx) + ', '

        evaluate_batch(model, dev_dataset, trigger_token_ids, snli)
        print("Current Triggers: " + print_string + " : " + str(model.get_metrics()['accuracy']))
    return model.get_metrics()['accuracy']

def evaluate_batch(model, batch, trigger_token_ids=None, snli=False):
    """
    Takes a batch of classification examples (SNLI or SST), and runs them through the model.
    If trigger_token_ids is not None, then it will append the tokens to the input.
    This funtion is used to get the model's accuracy and/or the loss with/without the trigger.
    """
    batch = move_to_device(batch, cuda_device=0)
    if trigger_token_ids is None:
        output_dict = model(batch['tokens'], batch['label'])
        return output_dict
    else:
        if isinstance(trigger_token_ids, dict):
            original_tokens = batch['tokens']['tokens'].clone()
            batch['tokens']['tokens'] = trigger_token_ids['tokens']['tokens']
            output_dict = model(batch['tokens'], batch['label'])
            batch['tokens']['tokens'] = original_tokens
        else:
            original_tokens = batch['tokens']['tokens'].clone()
            batch['tokens']['tokens'] = torch.tensor([trigger_token_ids]).cuda()
            output_dict = model(batch['tokens'], batch['label'])
            batch['tokens']['tokens'] = original_tokens
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
    original_labels = batch['label'].clone()
    if target_label is not None:
        # set the labels equal to the target (backprop from the target class, not model prediction)
        batch['label'] = int(target_label) * torch.ones_like(batch['label']).cuda()
    global extracted_grads
    extracted_grads = [] # clear existing stored grads
    loss = evaluate_batch(model, batch, trigger_token_ids, snli)['loss']
    loss.backward()
#     print(loss)
    # index 0 has the hypothesis grads for SNLI. For SST, the list is of size 1.
    grads = extracted_grads[0].cpu()
    batch['label'] = original_labels # reset labels

    # average grad across batch size, result only makes sense for trigger tokens at the front
    averaged_grad = torch.sum(grads, dim=0)
#     print(averaged_grad.size())
    trig_length = get_trig_length(trigger_token_ids)
    averaged_grad = averaged_grad[0:trig_length] # return just trigger grads
    return averaged_grad

def get_trig_length(trigger):
    all_tokens = trigger['tokens']['tokens'][0].detach().cpu().numpy()
    return len(all_tokens)


def get_best_candidates(model, batch, trigger_token_ids, cand_trigger_token_ids, \
                        snli=False, beam_size=1, increase_loss=False, is_df=False):
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
    
    trigger_tokens = list(trigger_token_ids['tokens']['tokens'][0].cpu().detach().numpy())
    loss_per_candidate = get_loss_per_candidate(0, model, batch, trigger_tokens,
                                                cand_trigger_token_ids, snli, is_df)
    # maximize the loss
    rand_ind = random.randint(0,beam_size-1)
#     rand_ind = 0
    top_candidates = [beamer(beam_size, loss_per_candidate, key=itemgetter(1))[rand_ind]]
    
    # top_candidates now contains beam_size trigger sequences, each with a different 0th token
    for idx in range(1, len(trigger_tokens)): # for all trigger tokens, skipping the 0th (we did it above)
        loss_per_candidate = []
        for cand, _ in top_candidates: # for all the beams, try all the candidates at idx
            loss_per_candidate.extend(get_loss_per_candidate(idx, model, batch, cand,
                                                             cand_trigger_token_ids, snli, is_df))
#         print(loss_per_candidate)
        top_candidates = [beamer(beam_size, loss_per_candidate, key=itemgetter(1))[rand_ind]]
#     print(max(top_candidates, key=itemgetter(1)))
    if increase_loss:
        output = max(top_candidates, key=itemgetter(1))
    else:
        output = min(top_candidates, key=itemgetter(1))
    return output[0], output[1]

def get_loss_per_candidate(index, model, batch, trigger_token_ids, cand_trigger_token_ids, snli=False, is_df=False):
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
    curr_loss = evaluate_batch(model, batch, trigger_token_ids, snli)['loss'].cpu().detach().numpy()
        
    loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))
    for cand_id in range(len(cand_trigger_token_ids[0])):
        trigger_token_ids_one_replaced = deepcopy(trigger_token_ids) # copy trigger
        trigger_token_ids_one_replaced[index] = cand_trigger_token_ids[index][cand_id] # replace one token
        loss = evaluate_batch(model, batch, trigger_token_ids_one_replaced, snli)['loss'].cpu().detach().numpy()
            
        loss_per_candidate.append((deepcopy(trigger_token_ids_one_replaced), loss))
    return loss_per_candidate
