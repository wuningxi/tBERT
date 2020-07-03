# Enable logging for gensim - optional
import logging
import os

# Gensim
import gensim.corpora as corpora
from src.preprocessing.Preprocessor import Preprocessor
from src.topic_model.topic_loader import get_topic_root_folder
from src.topic_model.lda import train_save_lda_model, train_ldamallet_topic_model, extract_topic_from_lda_prediction, \
    extract_topic_from_ldamallet_prediction, infer_lda_topics, load_lda_model
from src.topic_model.gsdmm import train_save_gsdmm_model,infer_gsdmm_topics,load_gsdmm_model

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

# load data

'''
Low level functions to preprocess corpus, train topic model, select best topic model and infer topic distribution
'''


# -----------------------------------------------
# Preprocessing function
# -----------------------------------------------

def lda_preprocess(data_tokenized, id2word=None, delete_stopwords=True, print_steps=False):
    '''
    Preprocess tokenized text data for LDA (deleting stopwords, recognising ngrams, lemmatisation
    :param data_tokenized: tokenized text data as nested list of tokens
    :param id2word: preexisting id2word object (to map dev or test split to identical ids), otherwise None (train split)
    :param print_steps: print intermediate output examples
    :return: preprocessed corpus as bag of wordids, id2word
    '''
    assert type(data_tokenized)==list
    assert type(data_tokenized[0])==list
    assert type(data_tokenized[0][0])==str
    data_finished = [Preprocessor.removeShortLongWords(s) for s in data_tokenized]
    if delete_stopwords:
        print('removing stopwords')
        data_finished = [Preprocessor.removeStopwords(s) for s in data_finished]
    if print_steps:
        print(data_finished[:1])

    if id2word is None:
        # Create Dictionary
        id2word = corpora.Dictionary(data_finished)

    # Create Corpus
    processed_texts = data_finished

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in processed_texts]

    # View
    if print_steps:
        # bag of ids
        print(corpus[20])
        # Human readable format of corpus (term-frequency)
        print(boids_to_human(corpus[:20], id2word))
    return corpus, id2word, processed_texts


# def get_optimal_model_path(opt):
#     return os.path.join(get_topic_pred_folder(opt), 'optimal')

# -----------------------------------------------
# functions to train and save topic models
# -----------------------------------------------

def train_topic_model(corpus, id2word, processed_texts, opt):
    '''
    Selects the correct training function for topic model dependent on topic_type
    :param corpus: preprocessed corpus
    :param id2word: id2word from preprocessing step
    :param opt: option dictionary
    :return: trained topic model
    '''
    parent_folder = os.path.join(opt['datapath'], 'topic_models')
    if not os.path.exists(parent_folder):
        os.mkdir(parent_folder)
    topic_root_folder = get_topic_root_folder(opt)
    if not os.path.exists(topic_root_folder):
        os.mkdir(topic_root_folder)
    print('Training {} model...'.format(opt['topic_type']))
    if opt['topic_type'] == 'LDA':
        return train_save_lda_model(corpus, id2word, opt)
    elif opt['topic_type'] == 'ldamallet':
        return train_ldamallet_topic_model(corpus, id2word, opt)
    elif opt['topic_type'] == 'gsdmm':
        return train_save_gsdmm_model(processed_texts, id2word, opt)
    else:
        raise ValueError('topic_type should be "LDA", "ldamallet" or "gsdmm".')


def infer_topic_dist(new_corpus, new_processed_texts, topic_model, topic_type):
    '''
    Get global topic distribution of all sentences from dev or test set from lda_model
    '''
    print('Inferring topic distribution...')
    # obtain topic distribution for dev set
    assert type(new_processed_texts) == list # of sentences
    assert type(new_processed_texts[0]) == list # of tokens
    assert type(new_processed_texts[0][0]) == str
    assert type(new_corpus) == list # of sentences
    assert type(new_corpus[0]) == list # of word id/count tuples
    assert type(new_corpus[0][0]) == tuple
    if topic_type in ['LDA', 'ldamallet']:
        dist_over_topic = infer_lda_topics(topic_model, new_corpus)
        # extract topic vectors from prediction
        global_topics = extract_topics_from_prediction(dist_over_topic, topic_type, topic_model.num_topics)
    elif topic_type == 'gsdmm':
        global_topics = infer_gsdmm_topics(topic_model, new_processed_texts)
    # sanity check
    assert (len(global_topics.shape) == 2)
    assert (global_topics.shape[1] == topic_model.num_topics)
    print(global_topics.shape)
    print('Done.')
    return global_topics


def extract_topics_from_prediction(dist_over_topic, type, num_topics):
    '''
    Selects the correct topic extraction function based on topic model type
    :param dist_over_topic: topic inference result from lda_model[new_corpus]
    :param type: 'LDA' or 'ldamallet'
    :return: topic array with (examples, num_topics)
    '''
    if type == 'LDA':
        return extract_topic_from_lda_prediction(dist_over_topic, num_topics)
    elif type == 'ldamallet':
        return extract_topic_from_ldamallet_prediction(dist_over_topic)
    else:
        raise ValueError('Incorrect topic type: {}. Should be "LDA" or "ldamallet"'.format(type))


# -----------------------------------------------
# convenience functions
# -----------------------------------------------

def boids_to_human(corpus, id2word):
    '''
    Convenience function to display bag of words instead of bag of ids
    :param corpus: bag of word ids
    :param id2word: id to word mapping dict
    :return: bag of words
    '''
    if len(corpus[0]) > 0:
        # multiple sentences
        human_format = [[(id2word[id], freq) for id, freq in cp] for cp in corpus]
    else:
        # one sentence
        human_format = [(id2word[id], freq) for id, freq in corpus]
    return human_format


def load_sentences_for_topic_model(opt):
    '''
    Loads dataset
    :param opt:
    :return:
    '''
    # for s in opt['subsets']:
    #     assert ('train' in s)  # only use for training data
    # load dataset
    from src.loaders.load_data import load_data
    data_dict = load_data(opt, numerical=False)
    R1 = data_dict['T1']
    R2 = data_dict['T2']

    # select sentences from dataset (avoid duplication in Semeval)
    if opt['dataset'] == 'Semeval':
        assert opt['tasks'] == ['A', 'B', 'C']
        # combine data from all subtasks (A,B,C)
        Asent_1 = [sent for i, sent in enumerate(R1[0]) if i % 10 == 0]  # only once
        Asent_2 = R2[0]
        Bsent_1 = [sent for i, sent in enumerate(R1[1]) if i % 10 == 0]  # only once
        Bsent_2 = R2[1]
        # Csent_1 = [sent for i,sent in enumerate(dataset[12]) if i%100==0] # same as Bsent_!
        Csent_2 = R2[2]
        sentences = Asent_1 + Asent_2 + Bsent_1 + Bsent_2 + Csent_2
        print(len(sentences))
    else:
        sentences = [s for s in R1[0]] + [s for s in R2[0]]
    return sentences


def train_one_topic_model_on_data(opt):
    '''
    Reads data for topic model,  preprocesses input, trains multiple topic models based on opt specifications, evaluates and saves topic model evaluation log.
    :param opt:
    :return:
    '''
    # load input data
    data = load_sentences_for_topic_model(opt)
    # preprocess input for LDA
    corpus, id2word, data_lemmatized, processed_texts = lda_preprocess(data, id2word=None, delete_stopwords=True,
                                                                       print_steps=False)
    # train model
    topic_model = train_topic_model(corpus, id2word, processed_texts, opt)  # saves model automatically
    return topic_model

def load_topic_model(opt):
    if opt['topic_type'] in ['LDA', 'ldamallet']:
        topic_model = load_lda_model(opt)
    elif opt['topic_type'] == 'gsdmm':
        topic_model = load_gsdmm_model(opt)
    return topic_model

def read_topic_model_log():
    '''

    :return: settings for best topic model ('num_topics','topic_type')
    '''
    NotImplementedError()
    log_path = os.path.join(get_topic_root_folder(opt), 'eval_log.txt')


if __name__ == '__main__':
    # Import Dataset
    opt = {'dataset': 'MSRP', 'datapath': 'data/','topic_type': 'gsdmm',
           'tasks': ['B',], 'n_gram_embd': False, 'numerical': False,
           'num_topics': 10,'topic_alpha':0.1,
           'subsets': ['train'], 'cache': True}
    # load input data
    train_data = load_sentences_for_topic_model(opt)
    # preprocess input for LDA
    corpus, id2word, processed_texts = lda_preprocess(train_data, id2word=None, delete_stopwords=True, print_steps=True)
    # # train model
    # topic_model = train_topic_model(corpus, id2word, processed_texts, opt)
    topic_model = load_topic_model(opt)
    topic_model.id2word

