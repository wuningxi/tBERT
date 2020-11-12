from src.loaders.load_data import load_data
from src.models.base_model_bert import model,test_opt
import argparse

# run tbert with different learning rates on a certain dataset
# example usage: python src/experiments/tbert.py -learning_rate 5e-05 -gpu 0 -topic_type ldamallet -topic word -dataset MSRP --debug

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument('-dataset', action="store", dest="dataset", type=str, default='MSRP')
parser.add_argument('-learning_rate', action="store", dest="learning_rate", type=str, default='3e-5') # learning_rates = [5e-5, 2e-5, 4e-5, 3e-5]
parser.add_argument('-layers', action="store", dest="hidden_layers", type=str, default='0')
parser.add_argument('-topic', action="store", dest="topic", type=str, default='word')
parser.add_argument('-gpu', action="store", dest="gpu", type=int, default=-1)
parser.add_argument("--speedup_new_layers",type="bool",nargs="?",const=True,default=False,help="Use 100 times higher learning rate for new layers.")
parser.add_argument("--debug",type="bool",nargs="?",const=True,default=False,help="Try to use small number of examples for troubleshooting")
parser.add_argument("--train_longer",type="bool",nargs="?",const=True,default=False,help="Train for 9 epochs")
parser.add_argument("--early_stopping",type="bool",nargs="?",const=True,default=False)
parser.add_argument("--unk_topic_zero",type="bool",nargs="?",const=True,default=False)
parser.add_argument('-seed', action="store", dest="seed", type=str, default='fixed')
parser.add_argument('-topic_type', action="store", dest="topic_type", type=str, default='ldamallet')

FLAGS, unparsed = parser.parse_known_args()

# sanity check command line arguments
if len(unparsed)>0:
    parser.print_help()
    raise ValueError('Unidentified command line arguments passed: {}\n'.format(str(unparsed)))

# setting model options based on flags

dataset = FLAGS.dataset
assert dataset in ['MSRP','Semeval_A','Semeval_B','Semeval_C','Quora']

hidden_layers = [int(h) for h in FLAGS.hidden_layers.split(',')]
for h in hidden_layers:
    assert h in [0,1,2]

topics = FLAGS.topic.split(',')
for t in topics:
    assert t in ['word','doc']

priority = []
todo = []
last = []

stopping_criterion = None #'F1'
patience = None
batch_size = 32 # standard minibatch size
if 'Semeval' in dataset:
    dataset, task = dataset.split('_')
    subsets = ['train_large', 'test2016', 'test2017']
    if task in ['A']:
        batch_size = 16 # need smaller minibatch to fit on GPU due to long sentences
        num_topics = 70
        if FLAGS.topic_type=='gsdmm':
            alpha = 0.1
        else:
            alpha = 50
    elif task == 'B':
        num_topics = 80
        if FLAGS.topic_type=='gsdmm':
            alpha = 0.1
        else:
            alpha = 10
    elif task == 'C':
        batch_size = 16 # need smaller minibatch to fit on GPU due to long sentences
        num_topics = 70
        if FLAGS.topic_type=='gsdmm':
            alpha = 0.1
        else:
            alpha = 10
else:
    task = 'B'
    if dataset== 'Quora':
        subsets = ['train', 'dev', 'test'] 
        num_topics = 90
        if FLAGS.topic_type=='gsdmm':
            alpha = 0.1
        else:
            alpha = 1
        task = 'B'
    else:
        subsets = ['train', 'dev', 'test'] # MSRP
        num_topics = 80
        if FLAGS.topic_type=='gsdmm':
            alpha = 0.1
        else:
            alpha = 1
        task = 'B'

if FLAGS.debug:
    max_m = 100
else:
    max_m = None

if FLAGS.train_longer:
    epochs = 9
    predict_every_epoch = True
else:
    epochs = 3
    predict_every_epoch = False

if FLAGS.early_stopping:
    patience = 2
    stopping_criterion = 'F1'

try:
    seed = int(FLAGS.seed)
except:
    seed = None

if FLAGS.unk_topic_zero:
    unk_topic = 'zero'
else:
    unk_topic = 'uniform'
for topic_scope in topics:

    for hidden_layer in hidden_layers:

        opt = {'dataset': dataset, 'datapath': 'data/',
                             'model': 'bert_simple_topic','bert_update':True,'bert_cased':False,
                             'tasks': [task],
                             'subsets': subsets,'seed':seed,
                             'minibatch_size': batch_size, 'L2': 0,
                             'max_m': max_m, 'load_ids': True,
                            'topic':topic_scope,'topic_update':False,
                            'num_topics':num_topics, 'topic_alpha':alpha,
               'unk_topic': unk_topic, 'topic_type':FLAGS.topic_type,
               'unk_sub': False, 'padding': False, 'simple_padding': True,
               'learning_rate': float(FLAGS.learning_rate),
               'num_epochs': epochs, 'hidden_layer':hidden_layer,
               'sparse_labels': True, 'max_length': 'minimum',
               'optimizer': 'Adam', 'dropout':0.1,
               'gpu': FLAGS.gpu,
               'speedup_new_layers':FLAGS.speedup_new_layers,
               'predict_every_epoch': predict_every_epoch,
               'stopping_criterion':stopping_criterion, 'patience':patience
               }
        todo.append(opt)

tasks = todo

if __name__ == '__main__':

    for i,opt in enumerate(tasks):
        print('Starting experiment {} of {}'.format(i+1,len(tasks)))
        l_rate = str(opt['learning_rate']).replace('-0','-')
        if FLAGS.speedup_new_layers:
            log = 'tbert_{}_seed_speedup_new_layers.json'.format(str(seed))
        # elif FLAGS.freeze_thaw_tune:
        #     log = 'tbert_{}_seed_freeze_thaw_tune.json'.format(str(seed))
        elif FLAGS.train_longer:
            log = 'tbert_{}_seed_train_longer.json'.format(str(seed))
        else:
            log = 'tbert_{}_seed.json'.format(str(seed))
        if FLAGS.early_stopping:
            log = log.replace('.json','_early_stopping.json')

        print(log)
        print(opt)
        test_opt(opt)
        data = load_data(opt, cache=True, write_vocab=False)
        if FLAGS.debug:
            # print(data[''])
            print(data['E1'][0].shape)
            print(data['E1'][1].shape)
            print(data['E1'][2].shape)

            print(data['E1_mask'][0])
            print(data['E1_seg'][0])
        opt = model(data, opt, logfile=log, print_dim=True)
