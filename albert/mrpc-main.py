# +
import sys
import os
sys.path.append('..')
import utils_mrpc as utils
import attacks
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.utils.data import Dataset
import torch
from typing import Dict
from transformers import (EvalPrediction, InputFeatures,
                          Trainer, TrainingArguments, glue_compute_metrics)
from sklearn.metrics import accuracy_score
import argparse
import copy

np.random.seed(123)
torch.manual_seed(123)

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--Type", help = "0, 1")
args = parser.parse_args()
# -

raw_datasets = load_dataset('glue', 'mrpc')
tokenizer = AutoTokenizer.from_pretrained("textattack/albert-base-v2-MRPC")
model = AutoModelForSequenceClassification.from_pretrained("textattack/albert-base-v2-MRPC")
model.eval().to('cuda:1')


# +
class CustomDataset(Dataset):
    def __init__(self, data, with_labels=True, bert_model='textattack/albert-base-v2-MRPC'):

        self.data = data  # pandas dataframe
        #Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)  
        self.with_labels = with_labels 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if isinstance(index, int):

            # Selecting sentence1 and sentence2 at the specified index in the data frame
            sent1 = str(self.data.loc[index, 'sentence1'])
            sent2 = str(self.data.loc[index, 'sentence2'])
            inp1 = tokenizer.convert_tokens_to_ids(sent1.split())
            inp2 = tokenizer.convert_tokens_to_ids(sent2.split())
#             print(inp1, inp2)
            # Tokenize the pair of sentences to get token ids, attention masks and token type ids
#             encoded_pair = self.tokenizer(sent1, sent2, 
#                                           return_tensors='pt', padding =True, truncation = True).to('cuda:1')  # Return torch.Tensor objects

#             token_ids = encoded_pair['input_ids']  # tensor of token ids
#             attn_masks = encoded_pair['attention_mask'] # binary tensor with "0" for padded values and "1" for the other values
#             token_type_ids = encoded_pair['token_type_ids']  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens
            l = len(inp1) + len(inp2) + 3
            token_ids = torch.tensor([[2]+inp1+[3]+inp2+[3] + [0]*(128-l)]).to("cuda:1")
            token_type_ids  = torch.tensor([[0]*(len(inp1)+2) + [1]*(len(inp2)+1) + [0]*(128-l)]).to("cuda:1")
            attn_masks = torch.tensor([[1]*(len(inp1)+len(inp2)+3) + [0]*(128-l)]).to("cuda:1")
            if self.with_labels:  # True if the dataset has labels
                label = self.data.loc[index, 'label']
                return {'input_ids': token_ids, 'attention_mask': attn_masks, 'token_type_ids': token_type_ids}, label  
            else:
                return {'input_ids': token_ids, 'attention_mask': attn_masks, 'token_type_ids': token_type_ids}
        else:
        
            # Selecting sentence1 and sentence2 at the specified index in the data frame
            sent1 = self.data.ix[index]['sentence1'].tolist()
            sent2 = self.data.ix[index]['sentence2'].tolist()
            labels = self.data.ix[index]['label'].tolist()
            
            max_len = 128
            inps = []
            tts = []
            atts = []
            for s1,s2 in zip(sent1, sent2):
                inp1 = tokenizer.convert_tokens_to_ids(s1.split())
                inp2 = tokenizer.convert_tokens_to_ids(s2.split())
                l = len(inp1) + len(inp2) + 3
                inps.append([[2]+inp1+[3]+inp2+[3] + [0]*(128-l)])
                tts.append([[0]*(len(inp1)+2) + [1]*(len(inp2)+1) + [0]*(128-l)])
                atts.append([[1]*(len(inp1)+len(inp2)+3) + [0]*(128-l)])
            
            token_ids = torch.tensor(inps).to("cuda:1")
            token_type_ids  = torch.tensor(tts).to("cuda:1")
            attn_masks = torch.tensor(atts).to("cuda:1")
       
            if self.with_labels:  # True if the dataset has labels
                return {'input_ids': token_ids, 'attention_mask': attn_masks, 'token_type_ids': token_type_ids}, label  
            else:
                return {'input_ids': token_ids, 'attention_mask': attn_masks, 'token_type_ids': token_type_ids}


# +
train = raw_datasets['train']  # 90 % of the original training data
test = raw_datasets['validation']  # the original validation data is used as test data because the test labels are not available with the datasets library

df_train = pd.DataFrame(train)
df_test = pd.DataFrame(test)

train_dataset = CustomDataset(df_train)
test_dataset = CustomDataset(df_test)


# +
def get_orig_preds(dataset):
    pred = []
    orig = []
    for ind in (dataset.data.index):
        inputs, label = dataset[ind]
        output_label = model(**inputs)
        pred.append(output_label.logits.cpu().detach().numpy())
        orig.append(label)
    return orig, pred
    
def get_accuracy(orig, preds):
    new_preds = []
    for x in preds:
        new_preds.append(np.argmax(x, axis=1)[0])
    return accuracy_score(orig, new_preds)

def validate_trigger_acc(df, trigger):
    df['sentence1'] = trigger + df['sentence1'].astype(str)
    new_eval_dataset = CustomDataset(df)
    orig, preds = get_orig_preds(new_eval_dataset)
    acc = get_accuracy(orig, preds)
    print("Accuracy: {}".format(acc))


# -

dataset_label_filter = int(args.Type)
df_targeted = df_test.loc[df_test['label'] == dataset_label_filter]

import pandas as pd
df_imps = pd.read_csv('results_'+str(dataset_label_filter)+'_prelim.csv')
df_imps['label'] = dataset_label_filter
df_imps = df_imps.rename(columns={'sent1':'sentence1', 'sent2':'sentence2'})
# df_imps['sentence1'] = df_imps['sent1']
# df_imps['sentence2'] = df_imps['sent2']
# imps_dataset = CustomDataset(df_imps)
validate_trigger_acc(copy.deepcopy(df_imps), '')
validate_trigger_acc(copy.deepcopy(df_targeted), '')


# +
def main():
    df = copy.deepcopy(df_imps)
#     df.sample(frac=1)
    utils.add_hooks(model)
    embedding_weight = utils.get_embedding_weight(model) # also save the word embedding matrix
    batch =1
    ep = 0
    trigger_token_ids = 'the the the'
    while ep<5:
        print('ep',ep)
        for i in range(df.shape[0]//batch -1):
            print('batch',i)
            data = df[i*batch:(i*batch)+batch]
            averaged_grad_sent1 = utils.get_average_grad(model, copy.deepcopy(data), trigger_token_ids, 3, tokenizer)
            print(averaged_grad_sent1)
            break
#             cand_trigger_token_ids_sent1 = attacks.hotflip_attack(averaged_grad_sent1,
#                                                             embedding_weight,
#                                                             num_candidates=40,increase_loss = True)
            
#             trigger_token_ids, loss = utils.get_best_candidates(model,
#                                                                   copy.deepcopy(data), trigger_token_ids.split(),
#                                                                   cand_trigger_token_ids_sent1, tokenizer,
#                                                                   beam_size = 1, increase_loss = True, 
#                                                                   field='sentence1')
#             trigger_token_ids = ' '.join(trigger_token_ids)
#             print('trig:', trigger_token_ids, loss)
            
#             validate_trigger_acc(copy.deepcopy(df_imps), trigger_token_ids)
#             validate_trigger_acc(copy.deepcopy(df_targeted), trigger_token_ids)
#         ep+=1
        break


# -

if __name__ == '__main__':
    main()
