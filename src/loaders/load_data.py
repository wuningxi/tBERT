import importlib
import os
import pickle

import numpy as np

from src.loaders.Quora.build import build
from src.loaders.augment_data import create_large_train, double_task_training_data
from src.preprocessing.Preprocessor import Preprocessor, get_onehot_encoding, reduce_embd_id_len
from src.topic_model.topic_loader import load_document_topics, load_word_topics
from src.topic_model.topic_visualiser import read_topic_key_table


def get_filenames(opt):
    filenames = [] # name of cache files
    for s in opt['subsets']:
        for t in opt['tasks']:
            prefix = ''
            if opt['dataset'] == 'Quora':
                if s.startswith('p_'):
                    prefix = ''
                else:
                    prefix = 'q_'
            if opt['dataset'] == 'PAWS':
                prefix = 'p_'
            if opt['dataset'] == 'MSRP':
                prefix = 'm_'
            filenames.append(prefix+s+'_'+t)
    return filenames

def get_filepath(opt):
    filepaths = []
    for name in get_filenames(opt):
        if 'quora' in name:
            filepaths.append(os.path.join(opt['datapath'], 'Quora', name + '.txt'))
            print('quora in filename')
        else:
            filepaths.append(os.path.join(opt['datapath'], opt['dataset'], name + '.txt'))
    return filepaths

def load_file(filename,onehot=True):
    """
    Reads file and returns tuple of (ID1, ID2, D1, D2, L) if ids=False
    """
    # todo: return dictionary
    ID1 = []
    ID2 = []
    D1 = []
    D2 = []
    L = []
    with open(filename,'r',encoding='utf-8') as read:
        for i,line in enumerate(read):
            if not len(line.split('\t'))==5:
                print(line.split('\t'))
            id1, id2, d1, d2, label = line.rstrip().split('\t')
            ID1.append(id1)
            ID2.append(id2)
            D1.append(d1)
            D2.append(d2)
            if 's_' in filename:
                if float(label)>=4:
                    label = 1
                elif float(label)<4:
                    label = 0
                else:
                    ValueError()
            L.append(int(label))
    L = np.array(L)
    # L = L.reshape(len(D1),1)
    if onehot:
        classes = L.shape[1] + 1
        L = get_onehot_encoding(L)
        print('Encoding labels as one hot vector.')
    return (ID1, ID2, D1, D2, L)

def get_dataset_max_length(opt):
    '''
    Determine maximum number of tokens in both sentences, as well as highes max length for current task
    :param opt: 
    :return: [maximum length of sentence in tokens,should first sentence be shortened?]
    '''
    tasks = opt['tasks']
    if opt['dataset'] in ['Quora','PAWS','GlueQuora']:
        cutoff = opt.get('max_length', 24)
        if cutoff == 'minimum':
            cutoff = 24
        s1_len, s2_len = cutoff, cutoff
    elif opt['dataset']=='MSRP':
        cutoff = opt.get('max_length', 40)
        if cutoff == 'minimum':
            cutoff = 40
        s1_len, s2_len = cutoff, cutoff
    elif 'B' in tasks:
        cutoff = opt.get('max_length', 100)
        if cutoff == 'minimum':
            cutoff = 100
        s1_len, s2_len = cutoff, cutoff
    elif 'A' in tasks or 'C' in tasks:
        cutoff = opt.get('max_length', 200)
        if cutoff == 'minimum':
            s1_len = 100
            s2_len = 200
        else:
            s1_len, s2_len = cutoff,cutoff
    return s1_len,s2_len,max([s1_len,s2_len])

def reduce_examples(matrices, m):
    '''
    Reduces the size of matrices
    :param matrices: 
    :param m: maximum number of examples
    :return: 
    '''
    return [matrix[:m] for matrix in matrices]

def create_missing_datafiles(opt,datafile,datapath):
    if not os.path.exists(datapath) and 'large' in datafile:
        create_large_train()
    if not os.path.exists(datapath) and 'double' in datafile:
        double_task_training_data()
    if not os.path.exists(datapath) and 'quora' in datafile:
        quora_opt = opt
        quora_opt['dataset'] = 'Quora'
        build(quora_opt)

def get_cache_folder(opt):
    return opt['datapath'] + 'cache/'

def load_cache_or_process(opt, cache, onehot):
    ID1 = []
    ID2 = []
    R1 = []
    R2 = []
    T1 = []
    T2 = []
    L = []
    filenames = get_filenames(opt)
    print(filenames)
    filepaths = get_filepath(opt)
    print(filepaths)
    for datafile,datapath in zip(filenames,filepaths):
        create_missing_datafiles(opt,datafile,datapath) # if necessary
        cache_folder = get_cache_folder(opt)
        if not os.path.exists(cache_folder):
            os.mkdir(cache_folder)
        # separate cache for n-gram / no n-gram replacement
        cached_path = cache_folder + datafile + '.pickle'
        # load preprocessed cache
        print(cached_path)
        if cache and os.path.isfile(cached_path):
            print("Loading cached input for " + datafile)
            try:
                with open(cached_path, 'rb') as f:
                    id1, id2, r1, r2, t1, t2, l = pickle.load(f)
            except ValueError:
                Warning('No ids loaded from cache: {}.'.format(cached_path))
                with open(cached_path, 'rb') as f:
                    r1, r2, l = pickle.load(f)
                    id1 = None
                    id2 = None

        # do preprocessing if cache not available
        else:
            print('Creating cache...')
            load_ids = opt.get('load_ids',True)
            if not load_ids:
                DeprecationWarning('Load_ids is deprecated setting. Now loaded automatically.')
            id1, id2, r1, r2, l = load_file(datapath,onehot)
            t1 = Preprocessor.basic_pipeline(r1)
            t2 = Preprocessor.basic_pipeline(r2)
            if cache: # don't overwrite existing data if cache=False
                pickle.dump((id1, id2, r1, r2, t1, t2, l), open(cached_path, "wb")) # Store the new data as the current cache
        ID1.append(id1)
        ID2.append(id2)
        R1.append(r1)
        R2.append(r2)
        L.append(l)
        T1.append(t1)
        T2.append(t2)
    return {'ID1': ID1, 'ID2': ID2, 'R1': R1, 'R2': R2,'T1': T1, 'T2': T2, 'L': L}


def load_data(opt,cache=True,numerical=True,onehot=False, write_vocab=False):
    """
    Reads data and does preprocessing based on options file and returns a data dictionary.
    Tokens will always be loaded, other keys depend on settings and will contain None if not available.
    :param opt: option dictionary, containing task and dataset info
    :param numerical: map tokens to embedding ids or not
    :param onehot: load labels as one hot representation or not
    :param write_vocab: write vocabulary to file or not
    :param cache: try to use cached preprocessed data or not
    :return: 
        { # essential:
        'ID1': ID1, 'ID2': ID2, # doc ids
        'R1': R1, 'R2': R2, # raw text
        'L': L, # labels
          # optional for word embds:
        'E1': E1, 'E2': E2, # embedding ids
        'embd': embd, # word embedding matrix
        'mapping_rates': mapping_rates,  
          # optional for topics:
        'D_T1':D_T1,'D_T2':D_T2, # document topics
        'word_topics':word_topics, # word topic matrix
        'topic_keys':topic_key_table} # key word explanation for topics
    """
    E1 = None
    E1_mask = None
    E1_seg = None
    E2 = None
    D_T1 = None
    D_T2 = None
    W_T1 = None
    W_T2 = None
    W_T = None
    topic_key_table = None
    mapping_rates = None
    embd = None
    word_topics = None
    vocab = []
    word2id = None
    id2word = None

    # get options
    dataset = opt['dataset']
    module_name = "src.loaders.{}.build".format(dataset)
    my_module = importlib.import_module(module_name)
    my_module.build(opt) # download and reformat if not existing
    topic_scope = opt.get('topic','')
    if not  topic_scope=='':
        topic_type = opt['topic_type'] = opt.get('topic_type', 'ldamallet')
    topic_update = opt.get('topic_update', False)
    assert topic_update in [True,False] # no  backward compatibility
    assert topic_scope in ['', 'word', 'doc', 'word+doc','word+avg']
    recover_topic_peaks = opt['unflat_topics'] =opt.get('unflat_topics', False)
    w2v_limit = opt.get('w2v_limit', None)
    assert w2v_limit is None # discontinued
    calculate_mapping_rate = opt.get('mapping_rate', False)
    tasks = opt.get('tasks', '')
    assert len(tasks)>0
    unk_topic = opt['unk_topic'] = opt.get('unk_topic', 'uniform')
    assert unk_topic in ['uniform','zero','min','small']
    s1_max_len,s2_max_len,max_len = get_dataset_max_length(opt) #maximum number of tokens in sentence
    max_m = opt.get('max_m',None) # maximum number of examples
    bert_processing = 'bert' in opt.get('model', '') # special tokens for BERT

    # load or create cache
    cache = load_cache_or_process(opt, cache, onehot)  # load max_m examples
    ID1 = cache['ID1']
    ID2 = cache['ID2']
    R1 = cache['R1']
    R2 = cache['R2']
    T1 = cache['T1']
    T2 = cache['T2']
    L = cache['L']

    # map words to embedding ids
    if numerical:
        print('Mapping words to BERT ids...')
        bert_cased = opt['bert_cased'] = opt.get('bert_cased', False)
        bert_large = opt['bert_large'] = opt.get('bert_large', False)
        # use raw text rather than tokenized text as input due to different preprocessing steps for BERT
        processor_output = Preprocessor.map_files_to_bert_ids(R1, R2, s1_max_len + s2_max_len, calculate_mapping_rate,
                                                              bert_cased=bert_cased, bert_large=bert_large)
        print('Finished word id mapping.')
        E1 = processor_output['E1']
        E1_mask = processor_output['E1_mask']
        E1_seg = processor_output['E1_seg']
        E2 = processor_output['E2']
        word2id = processor_output['word2id']
        id2word = processor_output['id2word']

        mapping_rates = processor_output['mapping_rates']
        if not bert_processing and not s1_max_len==max_len:
            E1 = reduce_embd_id_len(E1, tasks, cutoff=s1_max_len)
        if not bert_processing and not s2_max_len==max_len:
            E2 = reduce_embd_id_len(E2, tasks, cutoff=s2_max_len)

    if 'doc' in topic_scope:
        doc_topics = load_document_topics(opt,recover_topic_peaks=recover_topic_peaks,max_m=None)
        D_T1 = doc_topics['D_T1']
        D_T2 = doc_topics['D_T2']

    # reduce number of examples after mapping words to ids to ensure static mapping regardless of max_m
    if not ID1 is None:
        ID1 = reduce_examples(ID1, max_m)
        ID2 = reduce_examples(ID2, max_m)
    R1 = reduce_examples(R1, max_m)
    R2 = reduce_examples(R2, max_m)
    T1 = reduce_examples(T1, max_m)
    T2 = reduce_examples(T2, max_m)
    if not E1_mask is None:
        # reduce examples for bert
        E1 = reduce_examples(E1, max_m) #[train,dev,test]
        E1_mask = reduce_examples(E1_mask, max_m)
        E1_seg = reduce_examples(E1_seg, max_m)
    elif not E1 is None:
        E1 = reduce_examples(E1, max_m)
        E2 = reduce_examples(E2, max_m)
    if 'doc' in topic_scope:
        # reduce doc topics here after shuffling
        D_T1 = reduce_examples(D_T1, max_m)
        D_T2 = reduce_examples(D_T2, max_m)
    L = reduce_examples(L, max_m)

    # load topic related data
    if 'word' in topic_scope:
        word_topics = load_word_topics(opt,recover_topic_peaks=recover_topic_peaks)
        word2id_dict = word_topics['word_id_dict']
        print('Mapping words to topic ids...')
        W_T1 = [Preprocessor.map_topics_to_id(r,word2id_dict,s1_max_len,opt) for r in T1]
        W_T2 = [Preprocessor.map_topics_to_id(r,word2id_dict,s2_max_len,opt) for r in T2]

    if ('word' in topic_scope) or ('doc' in topic_scope):
        topic_key_table = read_topic_key_table(opt)

    print('Done.')
    data_dict= {'ID1': ID1, 'ID2': ID2, # doc ids
            'R1': R1, 'R2': R2, # raw text
            'T1': T1, 'T2': T2,  # tokenized text
            'E1': E1, 'E2': E2, # embedding ids
            'E1_mask': E1_mask, 'E1_seg': E1_seg,  # embedding ids
            'W_T1': W_T1, 'W_T2': W_T2, # separate word topic ids ()
            'W_T': W_T,  # joined word topic ids ()
            'D_T1':D_T1,'D_T2':D_T2, # document topics
            'L': L, # labels
            # misc
            'mapping_rates': mapping_rates,  # optional
            'id2word':id2word,
            'word2id':word2id,
            'word_topics':word_topics,
            'topic_keys':topic_key_table}
    return data_dict

if __name__ == '__main__':

    # Example usage
    opt = {'dataset': 'MSRP', 'datapath': 'data/',
           'tasks': ['B'],'max_length':'minimum',
           'subsets': ['train','dev','test'],
           'model': 'bert_simple_topic', 'load_ids':True, 'cache':True,
           'w2v_limit': None,#'pretrained_embeddings': 'Deriu',
           'topic':'word','num_topics':50,'topic_alpha':10,'topic_type':'ldamallet'#,'max_m':10000
           }
    data_dict = load_data(opt, cache=True, numerical=True, onehot=False)
