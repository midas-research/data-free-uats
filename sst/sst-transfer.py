# +
import sys
import os.path
from sklearn.neighbors import KDTree
import torch
import torch.optim as optim
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.data import Instance, Token
from allennlp.modules.token_embedders import Embedding
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer
from allennlp.common.util import lazy_groups_of
from allennlp.data.token_indexers import SingleIdTokenIndexer
sys.path.append('..')
import utils
import attacks
import pandas as pd
import argparse
import numpy as np
np.random.seed(123)
torch.manual_seed(123)

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--Type", help = "0, 1")
parser.add_argument("-e", "--EMB", help = "w2v or elmo")
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

EMBEDDING_TYPE = args.EMB # what type of word embeddings to use

# +
def main():
    # load the binary SST dataset.
    # use_subtrees gives us a bit of extra data by breaking down each example into sub sentences.
    if EMBEDDING_TYPE== "w2v":
        s_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
    else:
        s_indexer = ELMoTokenCharactersIndexer()
    
    reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
                                                    token_indexers={"tokens": s_indexer},
                                                    use_subtrees=True)
    train_data = reader.read('../data/train.txt')
    reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
                                                    token_indexers={"tokens": s_indexer})
    dev_data = reader.read('../data/dev.txt')
    
    vocab = Vocabulary.from_instances(train_data)

    # Randomly initialize vectors
    if EMBEDDING_TYPE == "None":
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=300)
        word_embedding_dim = 300

    # Load word2vec vectors
    elif EMBEDDING_TYPE == "w2v":
        embedding_path = "../data/crawl-300d-2M.vec"
        weight = _read_pretrained_embeddings_file(embedding_path,
                                                  embedding_dim=300,
                                                  vocab=vocab,
                                                  namespace="tokens")
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                    embedding_dim=300,
                                    weight=weight,
                                    trainable=False)
        word_embedding_dim = 300
    else:
        options_file = "../data/elmo_2x1024_128_2048cnn_1xhighway_options.json"
        weight_file =  "../data/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
        token_embedding = ElmoTokenEmbedder(options_file, weight_file)
        word_embedding_dim = 256

    # Initialize model, cuda(), and optimizer
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(word_embedding_dim,
                                                  hidden_size=512,
                                                  num_layers=2,
                                                  batch_first=True))
    model = LstmClassifier(word_embeddings, encoder, vocab)
    model.cuda()

    # where to save the model
    model_path = "/tmp/" + EMBEDDING_TYPE + "_" + "model.th"
    vocab_path = "/tmp/" + EMBEDDING_TYPE + "_" + "vocab"
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
                          num_epochs=5,
                          patience=1,
                          cuda_device=0)
        trainer.train()
        with open(model_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        vocab.save_to_files(vocab_path)
    model.cuda() # rnn cannot do backwards in train mode
    print('Starting trigger generation')
    # Register a gradient hook on the embeddings. This saves the gradient w.r.t. the word embeddings.
    # We use the gradient later in the attack.
#     utils.add_hooks(model)
#     embedding_weight = utils.get_embedding_weight(model) # also save the word embedding matrix

    # Use batches of size universal_perturb_batch_size for the attacks.
    universal_perturb_batch_size = 32
    iterator = BasicIterator(batch_size=universal_perturb_batch_size)
    iterator.index_with(vocab)
    
    dataset_label_filter = (args.Type)

    targeted_dev_data = []
    
    trigger = ""
    for instance in dev_data:
        if instance['label'].label == dataset_label_filter:
            targeted_dev_data.append(instance)
    
#     impressions = pd.read_csv('results_'+dataset_label_filter+'_new.csv')
    
#     impress = impressions['trig'].tolist()
    
#     impressions_list = []
#     for imp in impress:
#         tks = imp.split(' ')
#         imp_instance = reader.text_to_instance(tks)
#         imp_instance.add_field('label',targeted_dev_data[0]['label'])
#         impressions_list.append(imp_instance)
#     triggs = 'useless endurance useless'
    triggs = 'compassionately compassionately gaghan'
    # get accuracy before adding triggers
    acc = utils.get_accuracy_elmo(model, targeted_dev_data, vocab, trigger_token_ids=triggs)
    print(acc)
    
#     model.train() # rnn cannot do backwards in train mode
    
#     # initialize triggers which are concatenated to the input
#     num_trigger_tokens = 3
#     trigger_token_ids = [vocab.get_token_index("the")] * num_trigger_tokens
#     is_datafree = False
#     print(is_datafree, dataset_label_filter)
#     trig_store=[]
#     # sample batches, update the triggers, and repeat
#     for batch in lazy_groups_of(iterator(impressions_list, num_epochs=10, shuffle=True), group_size=1):
# #     for batch in lazy_groups_of(iterator(targeted_dev_data, num_epochs=5, shuffle=True), group_size=1):
#         # get accuracy with current triggers
#         model.train() # rnn cannot do backwards in train mode

#         # get gradient w.r.t. trigger embeddings for current batch
#         averaged_grad = utils.get_average_grad(model, batch, trigger_token_ids, is_df=is_datafree)
#         model.eval()
#         # pass the gradients to a particular attack to generate token candidates for each token.
#         cand_trigger_token_ids = attacks.hotflip_attack(averaged_grad,
#                                                         embedding_weight,
#                                                         num_candidates=40,
#                                                         increase_loss=True)
#         # cand_trigger_token_ids = attacks.random_attack(embedding_weight,
#         #                                                trigger_token_ids,
#         #                                                num_candidates=40)
#         # cand_trigger_token_ids = attacks.nearest_neighbor_grad(averaged_grad,
#         #                                                        embedding_weight,
#         #                                                        trigger_token_ids,
#         #                                                        tree,
#         #                                                        100,
#         #                                                        num_candidates=40,
#         #                                                        increase_loss=True)

#         # Tries all of the candidates and returns the trigger sequence with highest loss.
#         trigger_token_ids = utils.get_best_candidates(model,
#                                                       batch,
#                                                       trigger_token_ids,
#                                                       cand_trigger_token_ids,beam_size = 1,
#                                                      increase_loss = True, is_df=is_datafree)
        
#         print_string = ' '.join([vocab.get_token_from_index(x) for x in trigger_token_ids])
#         acc_val = utils.get_accuracy(model, targeted_dev_data, vocab, trigger_token_ids)
#         acc_ci = utils.get_accuracy(model, impressions_list, vocab, trigger_token_ids)
#         trig_store.append({'trig':print_string, 'loss_val':acc_val, 'loss_ci':acc_ci})
    
#     results = pd.DataFrame(trig_store)
#     results.to_csv('triggers_'+dataset_label_filter+'.csv', index = False)


# -

if __name__ == '__main__':
    main()
