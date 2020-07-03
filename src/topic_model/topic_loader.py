import os
import numpy as np

# -----------------------------------------------
# get functions for consistent model dirs
# -----------------------------------------------

def get_alpha_str(opt):
    return '_alpha{}'.format(opt.get('topic_alpha',50))
def get_topic_root_folder(opt):
    preprocessing = 'basic'
    if opt['topic_type']=='gsdmm':
        return os.path.join(opt['datapath'], 'topic_models', preprocessing+'_gsdmm', '')
    else:
        return os.path.join(opt['datapath'], 'topic_models', preprocessing,'')
def get_topic_parent_folder(opt):
    alpha_str = get_alpha_str(opt)
    return os.path.join(get_topic_root_folder(opt),opt['dataset']+alpha_str+'_'+str(opt['num_topics']),'')
def get_topic_model_folder(opt):
    return os.path.join(get_topic_parent_folder(opt), opt['topic_type'],'')
def get_topic_pred_folder(opt):
    return os.path.join(get_topic_model_folder(opt), 'predictions','')

# -----------------------------------------------
# prediction loaders
# -----------------------------------------------

def load_document_topics(opt,recover_topic_peaks,max_m=None):
    '''
    Loads inferred topic distribution for each sentence in dataset. Dependent on data subset (e.g. Semeval A vs. B).
    :param opt: option dict
    :return: document topics in list corresponding to subsets {'D_T1':T1,'D_T2':T2}
    '''
    # set model paths
    filepaths1 = []
    filepaths2 = []
    topic_model_folder = get_topic_pred_folder(opt)
    task = opt.get('tasks')[0]
    subsets = opt.get('subsets')
    for s in subsets: # train, dev, test
        filepaths1.append(os.path.join(topic_model_folder,task+'_'+s+'_1.npy'))
        filepaths2.append(os.path.join(topic_model_folder,task+'_'+s+'_2.npy'))
    # load
    T1 = [np.load(f) for f in filepaths1]
    T2 = [np.load(f) for f in filepaths2]

    if recover_topic_peaks:
        for split in range(len(T1)):
            for line in range(len(T1[split])):
                T1[split][line] = unflatten_topic(T1[split][line])
                T2[split][line] = unflatten_topic(T2[split][line])

    # reduce number of examples if necessary
    # max_examples = opt.get('max_m',None)
    if not max_m is None:
        T1 = [t[:max_m] for t in T1]
        T2 = [t[:max_m] for t in T2]
    return {'D_T1':T1,'D_T2':T2}

def unflatten_topic(topic_vector):
    # unflatten topic distribution
    min_val = topic_vector.min()
    for j, topic in enumerate(topic_vector):
        if topic == min_val:
            topic_vector[j] = 0
    return topic_vector

def load_word_topics(opt, add_unk = True,recover_topic_peaks=False):
    '''
    Reads word topic vector and dictionary from file
    :param opt: option dictionary containing settings for topic model
    :return: word_topic_dict,id_word_dict
    '''
    complete_word_topic_dict = {}
    id_word_dict = {}
    word_id_dict = {}
    count = 0
    topic_matrix = []
    num_topics = opt.get('num_topics',None)
    unk_topic = opt.get('unk_topic','uniform')
    word_topic_file = os.path.join(get_topic_pred_folder(opt),'word_topics.log')
    # todo:train topic model
    print("Reading word_topic vector from {}".format(word_topic_file))
    if add_unk:
        # add line for UNK word topics
        word = '<nontopic>'
        if unk_topic=='zero':
            topic_vector = np.array([0.0]*num_topics)
        elif unk_topic=='uniform':
            assert not recover_topic_peaks, "Do not use unk_topic='uniform' and 'unflat_topics'=True' together. As it will result in flattened non-topics, but unflattened topics."
            topic_vector = np.array([1/num_topics] * num_topics)
        else:
            raise ValueError
        wordid = 0
        id_word_dict[wordid] = word
        word_id_dict[word] = wordid
        complete_word_topic_dict[word] = topic_vector
        topic_matrix.append(topic_vector)
    # read other word topics
    with open(word_topic_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            count += 1
            if count > 2:
                ldata = line.rstrip().split(' ') # \xa in string caused trouble
                if add_unk:
                    wordid = int(ldata[0])+1
                else:
                    wordid = int(ldata[0])
                word = ldata[1]
                id_word_dict[wordid] = word
                word_id_dict[word] = wordid
                # print(ldata[2:])
                topic_vector = np.array([float(s.replace('[','').replace(']','')) for s in ldata[2:]])
                assert len(topic_vector)==num_topics
                if recover_topic_peaks:
                    topic_vector = unflatten_topic(topic_vector)
                complete_word_topic_dict[word] = topic_vector
                topic_matrix.append(topic_vector)
    topic_matrix = np.array(topic_matrix)
    print('word topic embedding dim: {}'.format(topic_matrix.shape))
    assert len(topic_matrix.shape)==2
    return {'complete_topic_dict':complete_word_topic_dict,'topic_dict':id_word_dict,'word_id_dict':word_id_dict,'topic_matrix':topic_matrix}

if __name__ == '__main__':

    # Example usage

    # load topic predictions for dataset
    opt = {'dataset': 'Semeval', 'datapath': 'data/',
           'tasks': ['A'],
           'subsets': ['train_large','test2016','test2017'],
           'model': 'basic_cnn', 'load_ids':True,
           'num_topics':20, 'topic_type':'ldamallet','unk_topic':'zero'}

    doc_topics = load_document_topics(opt)
    word_topics = load_word_topics(opt)

    word_topics['topic_dict']
    word_topics['topic_matrix']

    word_topics['topic_dict'][0]
    word_topics['topic_matrix'][0]

    word_topics['complete_topic_dict']['trends']
