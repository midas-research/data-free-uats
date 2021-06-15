# +
import sys
from sklearn.neighbors import KDTree
from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.common.util import lazy_groups_of
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.models import load_archive
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.iterators import BasicIterator
sys.path.append('..')
import utils
import attacks
import inspect
import numpy as np
import pandas as pd
import argparse
import torch
np.random.seed(123)
torch.manual_seed(123)

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--Ground", help = "Entailment, Contradiction, Neutral")
parser.add_argument("-t", "--Target", help = "0- entailment, 1 - contradiction, 2 - neutral")
parser.add_argument("-r", "--Run", help = "an number")
parser.add_argument("-oh", "--OnlyHypo", help = "0,1")
parser.add_argument("-df", "--IsDF", help = "0,1")
args = parser.parse_args()

# +
def main():
    # Load SNLI dataset
    single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
    tokenizer = WordTokenizer(end_tokens=["@@NULL@@"]) # add @@NULL@@ to the end of sentences
    reader = SnliReader(token_indexers={'tokens': single_id_indexer}, tokenizer=tokenizer)
    dev_dataset = reader.read('../data/snli_1.0_dev.jsonl')
    # Load model and vocab
    model = load_archive('../data/esim-glove-snli-2019.04.23.tar.gz').model
    model.eval().cuda()
    vocab = model.vocab

    # add hooks for embeddings so we can compute gradients w.r.t. to the input tokens
    utils.add_hooks(model)
    embedding_weight = utils.get_embedding_weight(model) # save the word embedding matrix

    # Batches of examples to construct triggers
    universal_perturb_batch_size = 32
    iterator = BasicIterator(batch_size=universal_perturb_batch_size)
    iterator.index_with(vocab)

    dataset_label_filter = args.Ground # only contradiction examples
    subset_dev_dataset = []
    
    oh = args.OnlyHypo
    if oh =='1': 
        impressions = pd.read_csv('results_'+dataset_label_filter+'_onlyhypo.csv')
    else:
        impressions = pd.read_csv('results_'+dataset_label_filter+'.csv')
        
    impressions_list = []
    for index,imp in impressions.iterrows():
        if oh == '1':
            imp['premise'] = ''
        imp_instance = (reader.text_to_instance(imp['premise'], imp['hypothesis'], dataset_label_filter))
        impressions_list.append(imp_instance)
    
#     # get accuracy before adding triggers
#     utils.add_forward_hooks_snli(model)
#     model.eval()
#     outputs = utils.get_forwards_snli(model, impressions_list, vocab)
#     import pickle
#     with open('snli_'+dataset_label_filter+'.pkl', 'wb') as f:
#         pickle.dump(outputs,f)
        
    for instance in dev_dataset:
        if instance['label'].label == dataset_label_filter:
#             l = instance['hypothesis'].sequence_length()
#             hypo = ' '.join([instance['hypothesis'][x].text for x in range(l-1)])
#             print(hypo)
#             imp_instance = (reader.text_to_instance('', hypo, dataset_label_filter))
            subset_dev_dataset.append(instance)
    
    target_label = args.Target
    run = args.Run
    isdf = args.IsDF
    # Get original accuracy before adding universal triggers
    utils.get_accuracy(model, subset_dev_dataset, vocab, trigger_token_ids=None, snli=True)
    utils.get_accuracy(model, impressions_list, vocab, trigger_token_ids=None, snli=True)

    # Initialize triggers
    num_trigger_tokens = 1 # one token prepended
    is_datafree = False
#     print(model, inspect.getfullargspec(model.forward))
    trigger_token_ids = [vocab.get_token_index("the")] * num_trigger_tokens
#     print(is_datafree, dataset_label_filter, target_label)
    sample batches, update the triggers, and repeat
    
    trig_store = []
    if isdf == '1':
        it = impressions_list
        nep = 10
    else:
        it = subset_dev_dataset
        nep = 5
    for batch in lazy_groups_of(iterator(it, num_epochs=nep, shuffle=True), group_size=1):
#     for batch in lazy_groups_of(iterator(subset_dev_dataset, num_epochs=1, shuffle=True), group_size=1):
        # get model accuracy with current triggers
        model.train() # rnn cannot do backwards in train mode

        # get grad of triggers
        averaged_grad = utils.get_average_grad(model, batch, trigger_token_ids, target_label, snli=True, is_df=is_datafree)
        model.eval()
        # find attack candidates using an attack method
        cand_trigger_token_ids = attacks.hotflip_attack(averaged_grad,
                                                        embedding_weight,
                                                        num_candidates=40,increase_loss = True)
        
        trigger_token_ids = utils.get_best_candidates(model,
                                                      batch,
                                                      trigger_token_ids,
                                                      cand_trigger_token_ids,
                                                      snli=True,beam_size = 1,
                                                     increase_loss = True, is_df = is_datafree)
        
        print_string = ' '.join([vocab.get_token_from_index(idx) for idx in trigger_token_ids])
        acc_val = utils.get_accuracy(model, subset_dev_dataset, vocab, trigger_token_ids, snli=True)
        acc_ci = utils.get_accuracy(model, impressions_list, vocab, trigger_token_ids, snli=True)
        
        trig_store.append({'trig':print_string, 'loss_val':acc_val, 'loss_ci':acc_ci})
    
    results = pd.DataFrame(trig_store)
    results.to_csv('triggers_'+oh+"_"+dataset_label_filter+'_'+target_label+'_'+run+'_'+isdf+'.csv', index = False)
# -

if __name__ == '__main__':
    main()
