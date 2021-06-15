# Data Free Universal Adversarial Triggers

This is the official code for the ICDM submission paper, Data Free Universal Adversarial Triggers.

The repository is broken down by task: 
+ `sst` attacks sentiment analysis using the SST dataset (AllenNLP-based).
+ `snli` attacks natural language inference models on the SNLI dataset (AllenNLP-based).
+ `albert` attacks ALBERT and MRPC dataset

## Get the data and models:

! cd data-free-uats

! mkdir data

! cd data

! wget https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt

! wget https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt

! wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip

! wget https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_dev.jsonl

! wget https://allennlp.s3-us-west-2.amazonaws.com/models/esim-glove-snli-2019.04.23.tar.gz

! wget https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-2017.09.04.tar.gz


References:
Highly inspried by the work by Eric Wallace: https://github.com/Eric-Wallace/universal-triggers
