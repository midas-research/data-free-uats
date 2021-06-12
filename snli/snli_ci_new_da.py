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
import utils_snli_new as utils
import attacks
import inspect
import numpy as np
import pandas as pd
import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--Type", help = "Entailment, Contradiction, Neutral")
args = parser.parse_args()


# -

def main():
    # Load SNLI dataset
    single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
    tokenizer = WordTokenizer(end_tokens=["@@NULL@@"]) # add @@NULL@@ to the end of sentences
    reader = SnliReader(token_indexers={'tokens': single_id_indexer}, tokenizer=tokenizer)
    dev_dataset = reader.read('../data/snli_1.0_dev.jsonl')
    # Load model and vocab
#     model = load_archive('../data/esim-glove-snli-2019.04.23.tar.gz').model
    model = load_archive('../data/decomposable-attention-2017.09.04.tar.gz').model
    print(model)
    model.eval().cuda()
    vocab = model.vocab

    # add hooks for embeddings so we can compute gradients w.r.t. to the input tokens
    utils.add_hooks(model)
    embedding_weight = utils.get_embedding_weight_DA(model) # save the word embedding matrix

    # Batches of examples to construct triggers
    universal_perturb_batch_size = 1
    iterator = BasicIterator(batch_size=universal_perturb_batch_size)
    iterator.index_with(vocab)

    # Subsample the dataset to one class to do a universal attack on that class
    dataset_label_filter = args.Type # only entailment examples
    
    impres_data = []
    for l in range(10, 30, 1):
        impres_data.append(reader.text_to_instance(' '.join(['a']*l),' '.join(['a']*l) , dataset_label_filter))
        impres_data.append(reader.text_to_instance(' '.join(['an']*l),' '.join(['an']*l) , dataset_label_filter))
        impres_data.append(reader.text_to_instance(' '.join(['a']*(l-1)),' '.join(['a']*l) , dataset_label_filter))
        impres_data.append(reader.text_to_instance(' '.join(['an']*(l-1)),' '.join(['an']*l) , dataset_label_filter))
        impres_data.append(reader.text_to_instance(' '.join(['a']*(l+1)),' '.join(['a']*l) , dataset_label_filter))
        impres_data.append(reader.text_to_instance(' '.join(['an']*(l+1)),' '.join(['an']*l) , dataset_label_filter))
        impres_data.append(reader.text_to_instance(' '.join(['a']*(l+2)),' '.join(['a']*l) , dataset_label_filter))
        impres_data.append(reader.text_to_instance(' '.join(['an']*(l+2)),' '.join(['an']*l) , dataset_label_filter))
        impres_data.append(reader.text_to_instance(' '.join(['a']*(l-2)),' '.join(['a']*l) , dataset_label_filter))
        impres_data.append(reader.text_to_instance(' '.join(['an']*(l-2)),' '.join(['an']*l) , dataset_label_filter))
#         impres_data.append(reader.text_to_instance(' ',' '.join(['a']*l) , dataset_label_filter))
#         impres_data.append(reader.text_to_instance(' ',' '.join(['an']*l) , dataset_label_filter))
#         impres_data.append(reader.text_to_instance(' ',' '.join(['the']*l) , dataset_label_filter))
        
    subset_dev_dataset = []
    for instance in dev_dataset:
        if instance['label'].label == dataset_label_filter:
            subset_dev_dataset.append(instance)

    target_dict= {'contradiction':'1', 'entailment':'0','neutral':'2' }
    target_label = target_dict[dataset_label_filter] # flip to contradiction

    # sample batches, update the triggers, and repeat
    d_res = []
    for batch in lazy_groups_of(iterator(subset_dev_dataset, num_epochs=1, shuffle=False), group_size=1):
        i = 0
        for trigger_batch in lazy_groups_of(iterator(impres_data, num_epochs=1, shuffle=False), group_size=1):
            new_batch = copy.deepcopy(trigger_batch)
            acc_orig = utils.get_accuracy(model, [impres_data[i]], vocab, trigger_token_ids=None, snli=True)
            i+=1
            ep = 0
            prev_loss_hypo = float('inf')
            loss_hypo = float('inf')
            loss_premise = float('inf')
            
            while ep < 5:
                d = {}
                prev_loss_hypo = loss_hypo
#                 model.train() # rnn cannot do backwards in train mode
                # get grad of triggers
                averaged_grad_hypo, averaged_grad_premise = utils.get_average_grad(model, batch, new_batch[0], target_label, snli=True)
                model.eval()
                model.zero_grad()
                # find attack candidates using an attack method
                print(averaged_grad_hypo.size(), embedding_weight.size())
                cand_trigger_token_ids_hypo = attacks.hotflip_attack(averaged_grad_hypo,
                                                                embedding_weight,
                                                                num_candidates=30,increase_loss = False)
                cand_trigger_token_ids_premise = attacks.hotflip_attack(averaged_grad_premise,
                                                                embedding_weight,
                                                                num_candidates=30,increase_loss = False)
    
                trigger_token_ids_hypo, loss_hypo = utils.get_best_candidates(model,
                                                                              batch, new_batch[0], 
                                                                              cand_trigger_token_ids_hypo, snli=True,
                                                                              beam_size = 5, increase_loss = False, 
                                                                              label_value = dataset_label_filter, field='hypothesis')
                hypo_text = [vocab.get_token_from_index(idx) for idx in trigger_token_ids_hypo]
                print('hypo:', ' '.join(hypo_text[:-1]), loss_hypo)
                
                trigger_token_ids_premise, loss_premise = utils.get_best_candidates(model,
                                                                              batch, new_batch[0], 
                                                                              cand_trigger_token_ids_premise, snli=True,
                                                                              beam_size = 5, increase_loss = False, 
                                                                              label_value = dataset_label_filter, field='premise')
                
                premise_text = [vocab.get_token_from_index(idx) for idx in trigger_token_ids_premise]
                print('premise:', ' '.join(premise_text[:-1]), loss_premise)
                
                new_batch = [reader.text_to_instance(' '.join(premise_text[:-1]), ' '.join(hypo_text[:-1]), dataset_label_filter)]
#                 new_batch = [reader.text_to_instance(' ', ' '.join(hypo_text[:-1]), dataset_label_filter)]
                new_b = []
                for b in lazy_groups_of(iterator(new_batch, num_epochs=1, shuffle=False), group_size=1):
                    new_b.append(b[0])
                acc = utils.get_accuracy(model, new_batch, vocab, trigger_token_ids=None, snli=True)
                new_batch = new_b
                ep+=1
                
                if acc == 1.0:
                    d['hypothesis'] = ' '.join(hypo_text[:-1])
                    d['premise'] = ' '.join(premise_text[:-1])
                    d['loss_hypo'] = loss_hypo
                    d['loss_premise'] = loss_premise
                    d_res.append(d)
                print('---------------------------------------------------------')
        break
    results = pd.DataFrame(d_res)
    results.to_csv('results_'+dataset_label_filter+'_da.csv', index = False)


if __name__ == '__main__':
    main()
