import os

import gensim
import numpy as np

from gensim.models import CoherenceModel
from src.topic_model.topic_loader import get_topic_model_folder, get_topic_parent_folder

# -----------------------------------------------
# save and load functions for lda topic models
# -----------------------------------------------

def train_save_lda_model(corpus, id2word, opt):
    '''
    Trains and saves original LDA model from Gensim implementation
    :param corpus: preprocessed corpus
    :param id2word: id2word from preprocessing step
    :param opt: option dictionary
    :return: trained topic model
    '''
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=opt['num_topics'],
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    # save
    save_topic_model(lda_model, opt, 'LDA')
    return lda_model


def train_ldamallet_topic_model(corpus, id2word, opt):
    '''
    Trains and saves LDA model from ldamallet implementation (tends to be better)
    :param corpus: preprocessed corpus
    :param id2word: id2word from preprocessing step
    :param opt: option dictionary
    :return: trained topic model
    '''
    # Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
    mallet_path = os.path.join(opt['datapath'], 'topic_models', 'mallet-2.0.8', 'bin', 'mallet')
    prefix = get_topic_model_folder(opt)  # saves model here, e.g.: 'data/topic_models/Semeval_15/ldamallet'
    parent_folder = get_topic_parent_folder(opt)  # e.g.: 'data/topic_models/Semeval_15'
    alpha = opt.get('topic_alpha', 50)
    if not os.path.exists(parent_folder):
        os.mkdir(parent_folder)
    os.mkdir(prefix)
    lda_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=opt['num_topics'],
                                                 id2word=id2word, prefix=prefix, alpha=alpha)  # try to change alpha
    # Save model
    save_topic_model(lda_model, opt)
    return lda_model


def save_topic_model(topic_model, opt):
    model_path = os.path.join(get_topic_model_folder(opt), 'lda_model')
    print('Saving Topic Model to {}...'.format(model_path))
    topic_model.save(model_path)


def load_lda_model(opt):
    model_path = os.path.join(get_topic_model_folder(opt), 'lda_model')
    print('Loading Topic Model from {}...'.format(model_path))
    lda_model = gensim.models.LdaModel.load(model_path)
    # update path in case topic model was trained with old code on VM (starting with /data-disk/...)
    lda_model.mallet_path = 'data/topic_models/mallet-2.0.8/bin/mallet'
    print('Done.')
    return lda_model

# -----------------------------------------------
# functions to infer topic distribution
# -----------------------------------------------

def infer_lda_topics(lda_model, new_corpus):
    assert type(new_corpus) == list # of sentences
    assert type(new_corpus[0]) == list # of word id/count tuples
    assert type(new_corpus[0][0]) == tuple
    return lda_model[new_corpus]

# -----------------------------------------------
# functions to explore topic distribution
# -----------------------------------------------

def extract_topic_from_lda_prediction(dist_over_topic, num_topics):
    '''
    Extracts topic vectors from prediction
    :param dist_over_topic: nested topic distribution object
    :param num_topics: number of topics
    :return: topic array with (examples, num_topics)
    '''
    # iterate through nested representation and extract topic distribution as single vector with length=num_topics for each document
    topic_array = []
    for j, example in enumerate(dist_over_topic):
        i = 0
        topics_per_doc = []
        for topic_num, prop_topic in example[0]:
            # print(i)
            # print(topic_num)
            while not i == topic_num:
                topics_per_doc.append(0)  # fill in 'missing' topics with probabilites < threshold as P=0
                # print('missing')
                i = i + 1
            topics_per_doc.append(prop_topic)
            i = i + 1
        while len(topics_per_doc) < num_topics:
            topics_per_doc.append(0)  # fill in last 'missing' topics
        topic_array.append(np.array(topics_per_doc))
    global_topics = np.array(topic_array)
    # sanity check
    if not ((len(global_topics.shape) == 2) and (global_topics.shape[1] == num_topics)):
        print('Inconsistent topic vector length detected:')
        i = 0
        for dist, example in zip(global_topics, dist_over_topic):
            if len(dist) != num_topics:
                print('{}th example with length {}: {}'.format(i, len(dist), dist))
                print('from: {}'.format(example[0]))
                print('--')
            i += 1
    return global_topics

def extract_topic_from_ldamallet_prediction(dist_over_topic):
    '''
    Extracts topic vectors from prediction
    :param dist_over_topic: nested topic distribution object
    :return: topic array with (examples, num_topics)
    '''
    global_topics = np.array([np.array([prop_topic for j, (topic_num, prop_topic) in enumerate(example)]) for example in
                              dist_over_topic])  # doesn't work with missing topics
    return global_topics

def evaluate_lda_topic_model(lda_model, corpus, processed_texts, id2word):
    '''
    Performs intrinsic evaluation by computing perplexity and coherence
    :param lda_model:
    :param corpus:
    :param processed_texts:
    :param id2word:
    :return: dictionary with perplexity and coherence
    '''
    try:
        model_perplexity = lda_model.log_perplexity(corpus)
        print('\nPerplexity: ', model_perplexity)  # a measure of how good the model is. lower the better.
        results = {'perplexity': model_perplexity}
    except AttributeError:
        results = {}
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_texts, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()  # the higher the better
    print('\nCoherence Score: ', coherence_lda)
    results['coherence'] = coherence_lda
    return results
if __name__ == '__main__':
    from src.topic_model.topic_trainer import load_sentences_for_topic_model,lda_preprocess
    # Import Dataset
    opt = {'dataset': 'Semeval', 'datapath': 'data/', 'num_topics': 10, 'topic_type': 'ldamallet',
           'tasks': ['A','B','C'], 'n_gram_embd': False, 'numerical': False,
           'subsets': ['train_large'], 'cache': True, 'stem':False,
           'max_m':100}
    # load input data
    data = load_sentences_for_topic_model(opt)
    # preprocess input for LDA
    corpus, id2word, processed_texts = lda_preprocess(data, id2word=None, delete_stopwords=True, print_steps=True)
    lda_model = load_lda_model(opt)
    # dist_over_topic = infer_lda_topics(lda_model, corpus)
    # global_topics = extract_topic_from_ldamallet_prediction(dist_over_topic)
