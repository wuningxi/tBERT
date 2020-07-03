# Enable logging for gensim - optional
import logging
import os
import re
import argparse

import numpy as np

from src.loaders.load_data import load_data
from src.topic_model.topic_loader import get_topic_model_folder, get_topic_pred_folder
from src.topic_model.topic_trainer import infer_topic_dist,lda_preprocess,load_topic_model

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

'''
High level functions to apply based on topic model functionality from src.preprocessing.topic_models to infer, save and 
load topic distribution for data sets
'''

# -----------------------------------------------
# infer, save, load topic distribution functions for dataset
# -----------------------------------------------

# --------  document topic model --------

def infer_and_write_document_topics(opt, topic_model=None, id2word=None):
    '''
    Infer global topic distribution for all documents in data splits (e.g. train, dev, test) mentioned in opt['subsets']
    and save as for both documents separately as numpy arrays.
    :param opt: option dictionary
    :return: Nothing
    '''
    subsets = opt['subsets']
    # try to load topic model
    if topic_model is None or id2word is None:
        topic_model = load_topic_model(opt)
        id2word = topic_model.id2word

    assert len(opt['tasks'])==1
    task = opt.get('tasks')[0]
    topic_dist_list = []

    for subset in subsets: # train, dev, test
        # load data split
        opt['subsets']=[subset]
        data_dict = load_data(opt, numerical=False)

        # use tokenized sentences, not raw strings (to prevent topic inference tokenisation bug which resulted in mapping the whole sentence to UNK -->same doc topic for every sentence)
        sent_1 = data_dict['T1'][0] # undo nested list since we are only dealing with one subset at a time
        sent_2 = data_dict['T2'][0]

        # preprocess and infer topics
        new_corpus_1, _, new_processed_texts_1 = lda_preprocess(sent_1, id2word=id2word, delete_stopwords=True,
                                                                print_steps=False)
        topic_dist_1 = infer_topic_dist(new_corpus_1,new_processed_texts_1, topic_model, opt['topic_type'])

        # preprocess and infer topics
        new_corpus_2, _, new_processed_texts_2 = lda_preprocess(sent_2, id2word=id2word, delete_stopwords=True,
                                                                print_steps=False)
        topic_dist_2 = infer_topic_dist(new_corpus_2,new_processed_texts_2, topic_model, opt['topic_type'])

        # sanity check
        assert(len(sent_1)==len(sent_2)==topic_dist_1.shape[0]==topic_dist_2.shape[0]) # same number of examples
        assert(topic_dist_1.shape[1]==topic_dist_2.shape[1]) # same number of topics

        # set model path
        topic_model_folder = get_topic_pred_folder(opt)
        # make folder if not existing
        if not os.path.exists(topic_model_folder):
            os.mkdir(topic_model_folder)
        topic_dist_path = os.path.join(topic_model_folder,task+'_'+subset)

        # save as separate numpy arrays
        np.save(topic_dist_path+'_1', topic_dist_1)
        np.save(topic_dist_path+'_2', topic_dist_2)
        topic_dist_list.extend([topic_dist_1,topic_dist_2])
    return topic_dist_list

# --------  word topic model --------

def infer_and_write_word_topics(opt, topic_model=None, id2word=None, max_vocab=None):
    '''
    Loads a topic model and writes topic predictions for each word in dictionary to file. Independent of data subset (e.g. Semeval A = B) due to shared dictionary.
    :param opt: option dictionary containing settings for topic model
    :return: void
    '''
    # try to load topic model
    if topic_model is None or id2word is None:
        topic_model = load_topic_model(opt)
    print('Infering and writing word topic distribution ...')
    if max_vocab is None:
        vocab_len = len(topic_model.id2word.keys())
        if not id2word is None:
            assert len(topic_model.id2word.keys()) == len(id2word.keys())
    else:
        vocab_len = max_vocab
    # create one word documents in bag of words format for topic model
    print(vocab_len)
    new_corpus =[[(i, 1)] for i in range(vocab_len)]
    # use [[word1],[word2],...] to prevent gsdmm from splitting them up into individual characters (e.g. ['w', 'o', 'r', 'd', '1'])
    new_processed_texts = [[topic_model.id2word[i]] for i in range(vocab_len)]
    print(new_processed_texts[:10])
    # get topic distribution for each word in topic model dictionary
    # dist_over_topic = lda_model[new_corpus]
    # word_topics = extract_topics_from_prediction(dist_over_topic, opt['topic_type'], lda_model.num_topics)

    word_topics = infer_topic_dist(new_corpus,new_processed_texts, topic_model, opt['topic_type'])
    # if opt['topic_type'] in ['LDA','ldamallet']:
    #     dist_over_topic = infer_lda_topics(topic_model, new_corpus)
    #     # extract topic vectors from prediction
    #     global_topics = extract_topics_from_prediction(dist_over_topic, type, topic_model.num_topics)
    # elif opt['topic_type'] == 'gsdmm':
    #     global_topics = infer_gsdmm_topics(topic_model, new_processed_texts)

    topic_model_folder = get_topic_pred_folder(opt)
    # make folder if not existing
    if not os.path.exists(topic_model_folder):
        os.mkdir(topic_model_folder)
    word_topic_file = os.path.join(topic_model_folder,'word_topics.log')
    with open(word_topic_file, 'w', encoding='utf-8') as outfile:
        # write to file
        model_path = os.path.join(get_topic_model_folder(opt), 'lda_model')
        outfile.writelines('Loading Topic Model from {}\n'.format(model_path))
        outfile.writelines('{}\n'.format(vocab_len))
        for i in range(vocab_len):
        # for i,(k,w) in enumerate(topic_model.id2word.items()):
            w = topic_model.id2word.id2token[i]
            if max_vocab is None or i<max_vocab:
                # replace multiple spaces and new line
                vector_str = re.sub('\s+', ' ', str(word_topics[i]))
                # remove space for last element in case of 'short' float
                vector_str = re.sub('\s+]', ']', str(vector_str))
                line = '{} {} {}'.format(i,w,vector_str)
                # print(line)
                outfile.writelines(line+'\n')
            else:
                break
    print('Done.')

if __name__ == '__main__':

    # Example usage
    # load existing topic model, create predictions and save
    # get command line arguments
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--debug",type="bool",nargs="?",const=True,default=False,help="Use debugger to track down bad values during training. "
             "Mutually exclusive with the --tensorboard_debug_address flag.")
    parser.add_argument('-dataset', action="store", dest="dataset", type=str, default='Quora')
    parser.add_argument('-topics', action="store", dest="topics", type=str, default='word,doc')
    parser.add_argument('-topic_type', action="store", dest="topic_type", type=str, default='ldamallet')
    FLAGS, unparsed = parser.parse_known_args()

    dataset = FLAGS.dataset
    if dataset in ['Quora','MSRP']:
        subsets = ['train', 'dev', 'test']  # evaluate on Quora
        tasks = ['B']
    elif dataset == 'PAWS':
        dataset = 'Quora'
        subsets = ['p_train', 'p_train', 'p_test']  # evaluate on PAWS
        tasks = ['B']
    elif dataset == 'Semeval':
        subsets = ['train_large', 'test2016', 'test2017']
        tasks = ['A','B','C']
    if FLAGS.debug:
        max_m = 100
    else:
        max_m = None


    for alpha in [0.1,1,10,50]: #1,10,50
        for num_topic in [t*10 for t in range(1,11)]: #[t*10 for t in range(1,11)]:
            for t in tasks:
                opt = {'dataset': dataset, 'datapath': 'data/',
                       'tasks': [t],
                       'subsets': subsets,
                        'load_ids': True,
                       'num_topics': num_topic, 'topic_alpha': alpha,
                       'topic_type': FLAGS.topic_type,
                       'max_m':max_m}
                if 'doc' in FLAGS.topics:
                    try:
                        infer_and_write_document_topics(opt)
                    except FileNotFoundError:
                        pass
            if 'word' in FLAGS.topics:
                try:
                    infer_and_write_word_topics(opt)
                except FileNotFoundError:
                    pass
