import os

import gensim
import numpy as np
import pandas as pd
import pickle

from src.ShortTextTopic.gsdmm import MovieGroupProcess
from src.topic_model.topic_loader import get_topic_model_folder, get_topic_parent_folder,get_topic_root_folder

# -----------------------------------------------
# save and load functions for gsdmm topic model
# -----------------------------------------------

def train_save_gsdmm_model(processed_texts,id2word, opt):
    '''
    Trains and saves Gibbs Sampling Dirichlet Multinomial Mixture model (topic model for short text)
    :param processed_texts: preprocessed texts
    :param opt: option dictionary
    :return: trained topic model
    '''
    alpha = opt.get('topic_alpha', 0.1)
    # It's important to always choose K to be larger than the number of clusters you expect exist in your data, as the algorithm can never return more than K clusters.
    gsdmm_model = MovieGroupProcess(K=opt['num_topics'], alpha=alpha, beta=alpha, n_iters=30)
    # corpus, id2word, data_lemmatized, texts = lda_preprocess(data, id2word=None, print_steps=True, lemmatize=False)
    vocab = set(x for doc in processed_texts for x in doc)
    n_terms = len(vocab)
    y = gsdmm_model.fit(processed_texts, n_terms)
    # save additional information for later
    gsdmm_model.id2word = id2word
    gsdmm_model.num_topics = opt['num_topics']
    save_gsdmm_model(gsdmm_model, opt)
    return gsdmm_model

def save_gsdmm_model(gsdmm_model, opt):
    root_folder = get_topic_root_folder(opt)
    if not os.path.exists(root_folder):
        os.mkdir(root_folder)
    model_folder = get_topic_parent_folder(opt)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    model_path = get_topic_model_folder(opt)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    print('Saving Topic Model to {}...'.format(model_path))
    with open(model_path + 'gsdmm.model', 'wb') as f:
        pickle.dump(gsdmm_model, f)
        f.close()
    write_keywords(gsdmm_model, opt)

def load_gsdmm_model(opt):
    model_path = os.path.join(get_topic_model_folder(opt), 'gsdmm.model')
    print('Loading Topic Model from {}...'.format(model_path))
    filehandler = open(model_path, 'rb')
    gsdmm_model = pickle.load(filehandler)
    print('Done.')
    return gsdmm_model

# -----------------------------------------------
# functions to infer topic distribution
# -----------------------------------------------

def infer_gsdmm_topics(gsdmm_model, texts):
    '''
    Predicts topic distribution for tokenized sentences
    :param gsdmm_model:
    :param texts: nested list of tokenized sentences
    :return:
    '''
    assert type(texts)==list
    assert type(texts[0])==list
    assert type(texts[0][0])==str
    dist_over_topic = [gsdmm_model.score(t) for t in texts]  # probablility distribution over topics
    global_topics = extract_topic_from_gsdmm_prediction(dist_over_topic)
    return global_topics

# -----------------------------------------------
# functions to explore topic distribution
# -----------------------------------------------

def extract_topic_from_gsdmm_prediction(dist_over_topic):
    '''
    Extracts topic vectors from prediction
    :param dist_over_topic: nested topic distribution object
    :return: topic array with (examples, num_topics)
    '''
    global_topics = np.array(dist_over_topic)
    return global_topics

def top_words(cluster_word_distribution, top_cluster, values):
    for cluster in top_cluster:
        sort_dicts =sorted(cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:values]
        print('Cluster %s : %s'%(cluster,sort_dicts))
        print(' — — — — — — — — — ')

def write_keywords(gsdmm_model, opt, cutoff=20):
    clusters = [i for i in range(len(gsdmm_model.cluster_doc_count))]
    importance = [doc_count/sum(gsdmm_model.cluster_doc_count) for doc_count in gsdmm_model.cluster_doc_count]
    topickey_path = os.path.join(get_topic_model_folder(opt), 'topickeys.txt')
    with open(topickey_path, 'w', encoding="utf-8") as outfile:
        for i in clusters:
            sort_dicts = sorted(gsdmm_model.cluster_word_distribution[i].items(), key=lambda k: k[1], reverse=True)[:cutoff]
            keywords = ' '.join([w for w,c in sort_dicts])
            outfile.writelines('{}\t{}\t{}\n'.format(i,importance[i],keywords))

# def show_topic_dist(topic_dist, lda_model):
#     '''
#     Print topic distribution with labels and scores
#     :param topic_dist:
#     :param lda_model:
#     :return:
#     '''
#     total = 0
#     props = []
#     topics = []
#     for topic_num, prop_topic in enumerate(topic_dist):
#         # [(self.id2word[id], value) for id, value in self.get_topic_terms(topicid, topn)]
#         wp = lda_model.show_topic(topic_num)
#         topic_keywords = ", ".join([word for word, prop in wp])
#         #     print(topic_keywords + ' {}'.format(prop_topic))
#         topics.append(topic_keywords)
#         props.append(prop_topic)
#         total += prop_topic
#     if 1 - total > 0.01:
#         raise ValueError('Should add up to 1, but are {}'.format(total))
#     sent_topics_df = pd.DataFrame(props, topics)
#     return sent_topics_df.sort_values(0, ascending=False)

if __name__ == '__main__':
    from src.topic_model.topic_trainer import load_sentences_for_topic_model,lda_preprocess
    # Import Dataset
    gsdmm_opt = {'dataset': 'Semeval', 'datapath': 'data/', 'num_topics': 10, 'topic_type': 'gsdmm',
           'tasks': ['A','B','C'], 'n_gram_embd': False, 'numerical': False, 'topic_alpha': 0.1,
           'subsets': ['train_large'], 'cache': True, 'stem':True,
           'max_m':100}
    # load input data
    data = load_sentences_for_topic_model(gsdmm_opt)
    # preprocess input for LDA

    corpus, id2word, processed_texts = lda_preprocess(data, id2word=None, delete_stopwords=True, print_steps=True)
    # # train model
    # gsdmm_model = train_save_gsdmm_model(processed_texts, id2word, gsdmm_opt)
    gsdmm_model = load_gsdmm_model(gsdmm_opt)
    # gsdmm_predictions = infer_gsdmm_topics(gsdmm_model, processed_texts)
    # todo: compare format with lda




    # # # # evaluate
    # results = evaluate_topic_model(lda_model, corpus, processed_texts, id2word)
    # from src.topic_model.topic_predictor import infer_and_write_word_topics
    # infer_and_write_word_topics(opt)