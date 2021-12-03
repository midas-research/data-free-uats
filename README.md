# MINIMAL: Mining Models for Data Free Universal Adversarial Triggers

This is the official code for the AAAI-2022 paper, MINIMAL: Mining Models for Data Free Universal Adversarial Triggers (https://arxiv.org/abs/2109.12406)

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

! wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json

! wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5

SST MODELS:

1)  LSTM - GLOVE
2)  LSTM - ELMO

Pl download the models from here: https://drive.google.com/drive/folders/1lj_6Tq5FL79cnhJzzn9jjg-rPgln0cZx?usp=sharing and add them to the data folder

RUNNING EXPERIMENTS:
Pl check individual folders for instructions

# Citation:
If you use the code, please cite the paper as:
```
@article{singla2021minimal,
  title={MINIMAL: Mining Models for Data Free Universal Adversarial Triggers},
  author={Singla, Yaman Kumar and Parekh, Swapnil and Singh, Somesh and Chen, Changyou and Krishnamurthy, Balaji and Shah, Rajiv Ratn},
  journal={arXiv preprint arXiv:2109.12406},
  year={2021}
}
```

References:
Highly inspried by the work by Eric Wallace: https://github.com/Eric-Wallace/universal-triggers
