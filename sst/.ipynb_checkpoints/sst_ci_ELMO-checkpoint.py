# +
import sys
import os.path
from sklearn.neighbors import KDTree
import torch
import torch.optim as optim
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader

from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.token_indexers import SingleIdTokenIndexer

# from allennlp.data.samplers import BucketBatchSampler
from allennlp.modules.token_embedders import ElmoTokenEmbedder

from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.data import Instance, Token
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer
from allennlp.common.util import lazy_groups_of
sys.path.append('..')
import utils_sst_ci as utils
import attacks
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--Type", help = "Positive,Negative")
args = parser.parse_args()


# -

# Simple LSTM classifier that uses the final hidden state to classify Sentiment. Based on AllenNLP
class LstmClassifier(Model):
    def __init__(self, word_embeddings, encoder, vocab):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.linear = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                      out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, tokens, label):
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)
        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)
        return output

    def get_metrics(self, reset=False):
        return {'accuracy': self.accuracy.get_metric(reset)}

EMBEDDING_TYPE = "elmo" # what type of word embeddings to use

# +
def main():
    elmo_token_indexer = ELMoTokenCharactersIndexer()
    reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
                                                    token_indexers={"tokens": elmo_token_indexer},
                                                    use_subtrees=True)
    train_data = reader.read('../data/train.txt')
    reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
                                                    token_indexers={"tokens": elmo_token_indexer})
    dev_data = reader.read('../data/dev.txt')

    vocab = Vocabulary.from_instances(train_data)
    
    options_file = "../data/elmo_2x1024_128_2048cnn_1xhighway_options.json"
    weight_file =  "../data/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
    elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
    elmo_embedding_dim = 256

    word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})
    encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(elmo_embedding_dim,
                                                  hidden_size=512,
                                                  num_layers=2,
                                                  batch_first=True))
    model = LstmClassifier(word_embeddings, encoder, vocab)
    model.cuda()
    print(train_data[0])
    # where to save the model
    model_path = "/tmp/" + EMBEDDING_TYPE + "_" + "model_test.th"
    vocab_path = "/tmp/" + EMBEDDING_TYPE + "_" + "vocab_test"
    # if the model already exists (its been trained), load the pre-trained weights and vocabulary
    if os.path.isfile(model_path):
        vocab = Vocabulary.from_files(vocab_path)
        model = LstmClassifier(word_embeddings, encoder, vocab)
        with open(model_path, 'rb') as f:
            model.load_state_dict(torch.load(f))
    # otherwise train model from scratch and save its weights
    else:
        iterator = BucketIterator(batch_size=64, sorting_keys=[("tokens", "num_tokens")])
        iterator.index_with(vocab)
        optimizer = optim.Adam(model.parameters())
        trainer = Trainer(model=model,
                          optimizer=optimizer,
                          iterator=iterator,
                          train_dataset=train_data,
                          validation_dataset=dev_data,
                          num_epochs=1,
                          patience=1,
                          cuda_device=0)
        trainer.train()
        with open(model_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        vocab.save_to_files(vocab_path)
    
#     model.train().cuda() # rnn cannot do backwards in train mode
    print('Starting trigger generation')
    
    embedding_layer = utils.get_embedding_weight(model) # also save the word embedding matrix
    vocab_namespace = ''.join([chr(x) for x in range(33,127)])
#     print(vocab_namespace)
#     all_tokens = list([vocab.get_token_from_index(x) for x in vocab_namespace])
    tokens = [Token(word) for word in sentence.split()]
    all_tokens = vocab.get_vocab_size('*label')
    print(all_tokens)
    
    inputs = utils.make_embedder_input(all_tokens, vocab, reader)
    print(inputs)
    embedding_matrix = embedding_layer(inputs).squeeze()
    
    # Register a gradient hook on the embeddings. This saves the gradient w.r.t. the word embeddings.
    # We use the gradient later in the attack.
#     print(model)
    print(embedding_weight)
    utils.add_hooks(model)
    
#     # Use batches of size universal_perturb_batch_size for the attacks.
    universal_perturb_batch_size = 1
    iterator = BasicIterator(batch_size=universal_perturb_batch_size)
    iterator.index_with(vocab)

    # filter the dataset to only positive or negative examples
    # (the trigger will cause the opposite prediction)
    dataset_label_filter = args.Type
    targeted_dev_data = []
    for instance in dev_data:
        if instance['label'].label == dataset_label_filter:
            targeted_dev_data.append(instance)
            
    d_res = []
    l = 10
    step_size = 2
    hidden_size = 512
    for batch in lazy_groups_of(iterator(targeted_dev_data, num_epochs=5), group_size=1):
        # get accuracy with current triggers
        batch = batch[0]
        if l>200:
            break
        for init_token in ['the','a', 'an']:
            trigger_token_ids = [vocab.get_token_index(init_token)] *l
            new_batch = utils.create_empty_batch(model, batch, trigger_token_ids, False)  
            acc_orig = utils.get_accuracy(model, new_batch, vocab, None)
            print(l, init_token, acc_orig)
            print('--------------------------------------------------------------------------')
            loss = float('inf')
            ep = 0
            while loss>0. and ep<2:
                d = {}
                new_batch = utils.create_empty_batch(model, batch, trigger_token_ids, False)
#                 print(new_batch)
                prev_loss =  loss
                model.train() # rnn cannot do backwards in train mode
                
                # get gradient w.r.t. trigger embeddings for current batch
                averaged_grad = utils.get_average_grad(model, batch, new_batch)
                model.eval()
                
                # pass the gradients to a particular attack to generate token candidates for each token.
                cand_trigger_token_ids = attacks.hotflip_attack(averaged_grad,
                                                                embedding_weight,
                                                                num_candidates=30,
                                                                increase_loss=False)
            
                # Tries all of the candidates and returns the trigger sequence with highest loss.
                trigger_token_ids,loss = utils.get_best_candidates(model,
                                                                batch,
                                                                new_batch,
                                                                cand_trigger_token_ids = cand_trigger_token_ids,
                                                                beam_size = 5,
                                                                increase_loss = False)
                print(l, trigger_token_ids, loss)
                acc_new = utils.get_accuracy(model, batch, vocab, trigger_token_ids)
                
                if acc_new ==1.0 and prev_loss!=loss:
                    d['trig'] = ' '.join([vocab.get_token_from_index(x) for x in trigger_token_ids])
                    d['loss'] = loss
                    d_res.append(d)
                
                ep+=1
                print('--------------------------------------------------------------------------')
                
        l+=step_size
        
    results = pd.DataFrame(d_res)
    results.to_csv('results_'+dataset_label_filter+'_new.csv')


# -
if __name__ == '__main__':
    main()
