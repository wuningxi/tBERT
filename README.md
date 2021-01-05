# tBERT 

![Alt text](tBERT.jpg?raw=true "tBERT model")

This repository provides code for the paper "tBERT: Topic Models and BERT Joining Forces for Semantic Similarity Detection" (https://www.aclweb.org/anthology/2020.acl-main.630/).

## Setup


### Download pretrained BERT

- Create cache folder in home directory:
```
cd ~
mkdir tf-hub-cache
cd tf-hub-cache
```
- Download pretrained BERT model and unzip:
```
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
```

### Download preprocessed data

- Go to the tBERT repository:
```
cd /path/to/tBERT/
```
- Download topic models and original datasets from dropbox:
```
wget "https://www.dropbox.com/s/6p26mkwv62677zt/original_data.tar.gz"
```
- Uncompress original_data.tar.gz:
```
tar zxvf original_data.tar.gz &
```
- Your tBERT directory should now have the following content (the Semeval folder is empty because data will be automatically downloaded by the script):
```
.
├── data
│   ├── cache
│   ├── logs
│   ├── models
│   ├── MSRP
│   │   └── MSRParaphraseCorpus
│   │       ├── msr-para-test.tsv
│   │       ├── msr-para-train.tsv
│   │       └── msr-para-val.tsv
│   ├── Quora
│   │   └── Quora_question_pair_partition
│   │       ├── dev.tsv
│   │       ├── test.tsv
│   │       └── train.tsv
│   ├── Semeval
│   └── topic_models
│       ├── basic
│       │   ├── MSRP_alpha1_80
│       │   ├── Quora_alpha1_90
│       │   ├── Semeval_alpha10_70
│       │   ├── Semeval_alpha10_80
│       │   └── Semeval_alpha50_70
│       ├── basic_gsdmm
│       │   ├── MSRP_alpha0.1_80
│       │   ├── Quora_alpha0.1_90
│       │   ├── Semeval_alpha0.1_70
│       │   └── Semeval_alpha0.1_80
│       └── mallet-2.0.8.zip
└── src
    ├── evaluation
    │   ├── difficulty
    │   ├── metrics
    ├── experiments
    ├── loaders
    │   ├── MSRP
    │   ├── PAWS
    │   ├── Quora
    │   └── Semeval
    ├── logs
    ├── models
    │   ├── forward
    │   ├── helpers
    ├── preprocessing
    ├── ShortTextTopic
    │   ├── gsdmm
    │   └── test
    └── topic_model
```

### Requirements

- This code has been tested with Python 3.6 and Tensorflow 1.11.
- Install the required Python packages as defined in requirements.txt:
```
pip install -r requirements.txt
```

## Usage

- You can try out if everything works by training a model on a small portion of the data (you can play around with different model options by changing the opt dictionary). Please make sure you are in the top tBERT directory when executing the following commands (`ls` should show `data  data.tar.gz  README.md  requirements.txt  src` as output):
```
python src/models/base_model_bert.py
```
- This should produce the following output:
```
['m_train_B', 'm_dev_B', 'm_test_B']
['data/MSRP/m_train_B.txt', 'data/MSRP/m_dev_B.txt', 'data/MSRP/m_test_B.txt']
data/cache/m_train_B.pickle
Loading cached input for m_train_B
data/cache/m_dev_B.pickle
Loading cached input for m_dev_B
data/cache/m_test_B.pickle
Loading cached input for m_test_B
Mapping words to BERT ids...
Finished word id mapping.
Done.
{'topic_type': 'ldamallet', 'load_ids': True, 'topic': 'doc', 'minibatch_size': 10, 'seed': 1, 'max_m': 10, 'bert_large': False, 'num_topics': 80, 'num_epochs': 1, 'model': 'bert_simple_topic', 'max_length': 'minimum', 'simple_padding': True, 'padding': False, 'bert_update': True, 'L2': 0, 'dropout': 0.1, 'bert_cased': False, 'speedup_new_layers': False, 'unk_topic': 'zero', 'stopping_criterion': 'F1', 'tasks': ['B'], 'learning_rate': 0.3, 'hidden_layer': 1, 'gpu': -1, 'optimizer': 'Adadelta', 'datapath': 'data/', 'unflat_topics': False, 'sparse_labels': True, 'freeze_thaw_tune': False, 'dataset': 'MSRP', 'topic_alpha': 1, 'predict_every_epoch': False, 'unk_sub': False, 'subsets': ['train', 'dev', 'test'], 'topic_update': True}
Topic scope: doc
input ids shape: (?, ?)
Loading pretrained model from https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1
---
Model: tBERT
---
D_T1 shape: (?, 80)
D_T2 shape: (?, 80)
pooled BERT shape: (?, 768)
combined shape: (?, 928)
hidden 1 shape: (?, 464)
output layer shape: (?, 2)
reading logs...
No file found at data/logs/test.json. Creating new log.
get new id
Model 0
2020-07-03 19:02:31.946114: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-03 19:02:37.224331: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1411] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 6a05:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-07-03 19:02:37.224390: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1490] Adding visible gpu devices: 0
logfile: test.json
Finetune...
Epoch 1
Dev F1 after epoch 1: 0.75
data/models/model_0/model_epoch1.ckpt
Maximum number of epochs reached during early stopping.
Finished training.
Load best model from epoch 1
reading logs...
Finished training after 0.55 min
Dev F1: 0.75
Test F1: 0.75
reading logs...
Wrote predictions for model_0.
```
- The model will be saved under data/models/model_0/ and the training log is available under data/logs/test.json
- You can also run an experiment on the complete dataset and alter different commandline flags, e.g.:
```
python src/experiments/tbert.py -dataset MSRP -layers 1 -topic doc -topic_type ldamallet -learning_rate 5e-5 --early_stopping -seed 3 -gpu 0
```
- This should give you the following output:
```
Starting experiment 1 of 1
tbert_1_seed_early_stopping.json
{'dropout': 0.1, 'model': 'bert_simple_topic', 'bert_cased': False, 'max_m': None, 'tasks': ['B'], 'padding': False, 'dataset': 'MSRP',
 'L2': 0, 'subsets': ['train', 'dev', 'test'], 'unk_sub': False, 'hidden_layer': 1, 'datapath': 'data/', 'predict_every_epoch': False,
'num_epochs': 3, 'simple_padding': True, 'patience': 2, 'speedup_new_layers': False, 'minibatch_size': 32, 'max_length': 'minimum', 'lo
ad_ids': True, 'topic_type': 'ldamallet', 'unk_topic': 'uniform', 'topic_update': False, 'sparse_labels': True, 'num_topics': 80, 'topi
c_alpha': 1, 'seed': 1, 'gpu': 0, 'stopping_criterion': 'F1', 'bert_update': True, 'learning_rate': 3e-05, 'optimizer': 'Adam', 'topic'
: 'doc'}
['m_train_B', 'm_dev_B', 'm_test_B']
['data/MSRP/m_train_B.txt', 'data/MSRP/m_dev_B.txt', 'data/MSRP/m_test_B.txt']
data/cache/m_train_B.pickle
Loading cached input for m_train_B
data/cache/m_dev_B.pickle
Loading cached input for m_dev_B
data/cache/m_test_B.pickle
Loading cached input for m_test_B
Mapping words to BERT ids...
Finished word id mapping.
Done.
{'dropout': 0.1, 'model': 'bert_simple_topic', 'bert_cased': False, 'max_m': None, 'tasks': ['B'], 'padding': False, 'dataset': 'MSRP',
 'unflat_topics': False, 'L2': 0, 'subsets': ['train', 'dev', 'test'], 'unk_sub': False, 'hidden_layer': 1, 'datapath': 'data/', 'bert_
large': False, 'predict_every_epoch': False, 'num_epochs': 3, 'simple_padding': True, 'patience': 2, 'speedup_new_layers': False, 'mini
batch_size': 32, 'max_length': 'minimum', 'load_ids': True, 'topic_type': 'ldamallet', 'unk_topic': 'uniform', 'topic_update': False, '
sparse_labels': True, 'num_topics': 80, 'topic_alpha': 1, 'seed': 1, 'gpu': 0, 'stopping_criterion': 'F1', 'bert_update': True, 'learni
ng_rate': 3e-05, 'optimizer': 'Adam', 'topic': 'doc'}
Running on GPU: 0
Topic scope: doc
input ids shape: (?, ?)
Loading pretrained model from https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1
---
Model: tBERT
---
D_T1 shape: (?, 80)
D_T2 shape: (?, 80)
pooled BERT shape: (?, 768)
combined shape: (?, 928)
hidden 1 shape: (?, 464)
output layer shape: (?, 2)
reading logs...
get new id
Model 1
2020-07-03 19:51:34.629485: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow bi
nary was not compiled to use: AVX2 FMA
2020-07-03 19:51:39.501180: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1411] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 6a05:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-07-03 19:51:39.501233: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1490] Adding visible gpu devices: 0
2020-07-03 19:51:39.769183: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-07-03 19:51:39.769242: I tensorflow/core/common_runtime/gpu/gpu_device.cc:977]      0
2020-07-03 19:51:39.769263: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990] 0:   N
2020-07-03 19:51:39.769368: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1103] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10761 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 6a05:00:00.0, compute capability: 3.7)
logfile: tbert_1_seed_early_stopping.json
Finetune...
Epoch 1
Dev F1 after epoch 1: 0.8917378783226013
data/models/model_1/model_epoch1.ckpt
Epoch 2
Dev F1 after epoch 2: 0.9058663249015808
data/models/model_1/model_epoch2.ckpt
Epoch 3
Dev F1 after epoch 3: 0.8959999680519104
Maximum number of epochs reached during early stopping.
Finished training.
Load best model from epoch 2
reading logs...
Finished training after 10.74 min
Dev F1: 0.9059
Test F1: 0.8841
reading logs...
Wrote predictions for model_1.
```

pip-install-ob_rbe5r/MarkupSafe/setup.py
```
pip install --upgrade pip setuptools==45.2.0
```
