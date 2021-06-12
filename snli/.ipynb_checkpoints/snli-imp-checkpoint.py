import sys
from sklearn.neighbors import KDTree
from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.common.util import lazy_groups_of
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.models import load_archive
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.iterators import BasicIterator
sys.path.append('..')
import utils_snli as utils
import attacks
import copy
import inspect
import numpy as np
import pandas as pd

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
    universal_perturb_batch_size = 1
    iterator = BasicIterator(batch_size=universal_perturb_batch_size)
    iterator.index_with(vocab)

    # Subsample the dataset to one class to do a universal attack on that class
#     dataset_label_filter = 'entailment' # only entailment examples
#     dataset_label_filter = 'contradiction' # only contradiction examples
    dataset_label_filter = 'neutral' # only neutral examples
    
    target_dict= {'contradiction':'1', 'entailment':'0','neutral':'2' }
    target_label = target_dict[dataset_label_filter] # flip to contradiction

    subset_dev_dataset = []
    for instance in dev_dataset:
        if instance['label'].label == dataset_label_filter:
            subset_dev_dataset.append(instance)
    
    impres_data = []
    for l in range(10, 50, 1):
        impres_data.append(reader.text_to_instance(' ',' '.join(['a']*l) , dataset_label_filter))
        impres_data.append(reader.text_to_instance(' ',' '.join(['an']*l) , dataset_label_filter))

    d_res = []    
    for batch in lazy_groups_of(iterator(impres_data, num_epochs=1, shuffle=False), group_size=1):
        
        new_batch = copy.deepcopy(batch)
        acc_orig = utils.get_accuracy(model, new_batch, vocab, trigger_token_ids=None, snli=True)
        ep = 0
        prev_loss_hypo = float('inf')
        loss_hypo = float('inf')
#         loss_premise = float('inf')
        
        while ep < 5 and loss_hypo>0.:
            d = {}
            ep+=1
            # get grad of triggers
            prev_loss_hypo = loss_hypo
            model.train() # rnn cannot do backwards in train mode
            averaged_grad_hypo, _ = utils.get_average_grad(model, new_batch, None,\
                                                                               target_label, snli=True)
            model.eval()
            # find attack candidates using an attack method
            cand_trigger_token_ids_hypo = attacks.hotflip_attack(averaged_grad_hypo,
                                                            embedding_weight,
                                                            num_candidates=40,increase_loss = False)
#             cand_trigger_token_ids_premise = attacks.hotflip_attack(averaged_grad_premise,
#                                                             embedding_weight,
#                                                             num_candidates=40,increase_loss = False)

            trigger_token_ids_hypo,loss_hypo = utils.get_best_candidates(model,
                                                          new_batch,
                                                          cand_trigger_token_ids_hypo,
                                                          snli=True,term ='hypothesis',beam_size = 1,
                                                         increase_loss = False)
            hypo_text = [vocab.get_token_from_index(idx) for idx in trigger_token_ids_hypo]
            print('hypo', loss_hypo)
            
#             trigger_token_ids_premise, loss_premise = utils.get_best_candidates(model,
#                                                           new_batch,
#                                                           cand_trigger_token_ids_premise,
#                                                           snli=True,term ='premise',beam_size = 1,
#                                                          increase_loss = False)
#             premise_text = [vocab.get_token_from_index(idx) for idx in trigger_token_ids_premise]
#             print('premise',loss_premise)
            
            new_batch = [reader.text_to_instance(' ', ' '.join(hypo_text[:-1]), dataset_label_filter)]
            new_b = []
            for b in lazy_groups_of(iterator(new_batch, num_epochs=1, shuffle=False), group_size=1):
                new_b.append(b[0])
            acc = utils.get_accuracy(model, new_b, vocab, trigger_token_ids=None, snli=True)
            new_batch = new_b
            
            if acc == 1.0 and prev_loss_hypo!=loss_hypo:
                print('in')
                d['trig'] = ' '.join(hypo_text[:-1])
                print(d['trig'] )
#                 d['premise'] = ' '.join(premise_text[:-1])
                d['length'] = len(hypo_text[:-1])
                d['loss'] = loss_hypo
                d_res.append(d)
            print('---------------------------------------------------------')
    
    results = pd.DataFrame(d_res)
    results.to_csv('results_'+dataset_label_filter+'.csv')


# -

if __name__ == '__main__':
    main()
