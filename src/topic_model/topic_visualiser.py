import gzip
import os

import pandas as pd
import pyLDAvis
import pyLDAvis.gensim  # don't skip this

from src.topic_model.topic_loader  import get_topic_model_folder
import matplotlib.pyplot as plt


def visualise_topics(opt,lda_model=None, corpus=None, id2word=None):
    '''
    Generic topic visualisation function that calls specific visualisation based on topic model type
    :param opt: 
    :param lda_model: 
    :param corpus: 
    :param id2word: 
    :return: 
    '''
    print('Visualising {} topic model...'.format(opt['topic_type']))
    if opt['topic_type'] == 'ldamallet':
        return visualise_ldamallet_topics(opt['dataset'],opt.get('topic_alpha',50),opt['num_topics'])
    elif opt['topic_type'] == 'LDA':
        return visualise_lda_topics(lda_model, corpus, id2word)
    else:
        ValueError('Topic model type not supported. Choose "ldamallet" or "LDA".')


def visualise_lda_topics(lda_model, corpus, id2word):
    '''
    Visualizes the topics for Gensim's LDA implementation
    :param lda_model: 
    :param corpus: 
    :param id2word: 
    :return: visualisation
    '''
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    return vis

def visualise_ldamallet_topics(dataset,alpha,num_topic):
    '''
    Extracts relevant information form ldamallet's LDA model and visualizes the topics with Gensim's LDA visualisation
    :return: visualisation
    '''
    ldamallet_dir = 'data/topic_models/basic/{}_alpha{}_{}/ldamallet'.format(dataset,alpha,num_topic) # e.g. Semeval_alpha50_20
    convertedLDAmallet = convertLDAmallet(dataDir=ldamallet_dir, filename='state.mallet.gz')
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.prepare(**convertedLDAmallet)
    # pyLDAvis.display(vis)
    return vis

# from http://jeriwieringa.com/2018/07/17/pyLDAviz-and-Mallet/#comment-4018495276
def convertLDAmallet(dataDir='data/topic_models/SemevalA/', filename='state.mallet.gz'):
    def extract_params(statefile):
        """Extract the alpha and beta values from the statefile.

        Args:
            statefile (str): Path to statefile produced by MALLET.
        Returns:
            tuple: alpha (list), beta    
        """
        with gzip.open(statefile, 'r') as state:
            params = [x.decode('utf8').strip() for x in state.readlines()[1:3]]
        return (list(params[0].split(":")[1].split(" ")), float(params[1].split(":")[1]))

    def state_to_df(statefile):
        """Transform state file into pandas dataframe.
        The MALLET statefile is tab-separated, and the first two rows contain the alpha and beta hypterparamters.

        Args:
            statefile (str): Path to statefile produced by MALLET.
        Returns:
            datframe: topic assignment for each token in each document of the model
        """
        return pd.read_csv(statefile,
                           compression='gzip',
                           sep=' ',
                           skiprows=[1, 2]
                           )

    params = extract_params(os.path.join(dataDir, filename))
    alpha = [float(x) for x in params[0][1:]]
    beta = params[1]
    # print("{}, {}".format(alpha, beta))

    df = state_to_df(os.path.join(dataDir, filename))
    df['type'] = df.type.astype(str)
    # df[:10]

    # Get document lengths from statefile
    docs = df.groupby('#doc')['type'].count().reset_index(name='doc_length')
    # docs[:10]

    # Get vocab and term frequencies from statefile
    vocab = df['type'].value_counts().reset_index()
    vocab.columns = ['type', 'term_freq']
    vocab = vocab.sort_values(by='type', ascending=True)
    # vocab[:10]

    # Topic-term matrix from state file
    # https://ldavis.cpsievert.me/reviews/reviews.html
    import sklearn.preprocessing
    def pivot_and_smooth(df, smooth_value, rows_variable, cols_variable, values_variable):
        """
        Turns the pandas dataframe into a data matrix.
        Args:
            df (dataframe): aggregated dataframe 
            smooth_value (float): value to add to the matrix to account for the priors
            rows_variable (str): name of dataframe column to use as the rows in the matrix
            cols_variable (str): name of dataframe column to use as the columns in the matrix
            values_variable(str): name of the dataframe column to use as the values in the matrix
        Returns:
            dataframe: pandas matrix that has been normalized on the rows.
        """
        matrix = df.pivot(index=rows_variable, columns=cols_variable, values=values_variable).fillna(value=0)
        matrix = matrix.values + smooth_value
        normed = sklearn.preprocessing.normalize(matrix, norm='l1', axis=1)
        return pd.DataFrame(normed)

    phi_df = df.groupby(['topic', 'type'])['type'].count().reset_index(name='token_count')
    phi_df = phi_df.sort_values(by='type', ascending=True)
    # phi_df[:10]
    phi = pivot_and_smooth(phi_df, beta, 'topic', 'type', 'token_count')
    # phi[:10]
    theta_df = df.groupby(['#doc', 'topic'])['topic'].count().reset_index(name='topic_count')
    # theta_df[:10]
    theta = pivot_and_smooth(theta_df, alpha, '#doc', 'topic', 'topic_count')
    data = {'topic_term_dists': phi,
            'doc_topic_dists': theta,
            'doc_lengths': list(docs['doc_length']),
            'vocab': list(vocab['type']),
            'term_frequency': list(vocab['term_freq'])
            }
    return data

# Visualisation for topic prediction
def read_topic_key_table(opt):
    keyfile = get_topic_model_folder(opt) + 'topickeys.txt'
    topic_table = pd.read_csv(keyfile, sep='\t', header=None, names=['topic','ratio','keywords'])
    return reformat_topic_key_table(topic_table)

def reformat_topic_key_table(topic_table):
    topic_table['keywords'] = ['T{}: {}'.format(i, k) for i, k in zip(topic_table['topic'], topic_table['keywords'])]
    return topic_table

def visualise_topic_pred(sentence,topic_vector,topic_table,figsize=None):
    '''
    Produces horizontal bar plot for predicted topics
    :param sentence: str
    :param topic_vector: vector with length=num_topics 
    :param topic_table: table with topic key words
    :return: plot
    '''
    print(sentence)
    print(topic_vector)
    topic_table['pred'] = topic_vector
    return topic_table.plot.barh('keywords','pred',figsize=figsize)

def shorten_topic_keywords(topic_table,n_keywords=None):
    topic_keywords = [t for t in topic_table['keywords']]
    if not n_keywords is None:
        topic_keywords = [' '.join(t.split(' ')[:n_keywords + 1]) for t in topic_keywords]
    return topic_keywords

def visualise_doc_topic_pair_pred(sentence1, sentence2, topic_vector1,topic_vector2,topic_table,figsize=None,print_vectors=False,n_keywords=None,alpha=None):
    '''
    Produces horizontal bar plot for predicted topics
    :param sentence: str
    :param topic_vector: vector with length=num_topics 
    :param topic_table: table with topic key words
    :return: plot
    '''
    if print_vectors:
        print('doc1: {} {}'.format(sentence1,topic_vector1))
        print('doc2: {} {}'.format(sentence2,topic_vector2))
    else:
        print('doc1: {}'.format(sentence1))
        print('doc2: {}'.format(sentence2   ))
    plt.figure(figsize=figsize)
    topic_keywords = shorten_topic_keywords(topic_table,n_keywords)
    title = 'doc1: {}\ndoc2: {}'.format(sentence1, sentence2)
    if not alpha is None:
        title = '{}\nalpha: {}'.format(title,alpha)
    plt.title(title)
    plt.barh(topic_keywords, topic_vector1, alpha=0.1, label='doc1', color='r')
    plt.barh(topic_keywords, topic_vector2, alpha=0.1, label='doc2', color='b')
    plt.legend(loc='best')
    plt.show()

def visualise_word_topic_pair_pred(sentence1, sentence2, topic_vector1,topic_vector2,topic_table,figsize=None,n_keywords=None,alpha=None):
    '''
    Produces horizontal bar plot for predicted topics
    :param sentence: str
    :param topic_vector: vector with length=num_topics 
    :param topic_table: table with topic key words
    :return: plot
    '''
    plt.figure(figsize=figsize)
    print('doc1: {}'.format(sentence1))
    print('doc2: {}'.format(sentence2))
    topic_keywords = shorten_topic_keywords(topic_table,n_keywords)
    title = 'doc1: {}\ndoc2: {}'.format(sentence1, sentence2)
    if not alpha is None:
        title = '{}\nalpha: {}'.format(title,alpha)
    plt.title(title)
    for i in range(len(topic_vector1)):
        if i > 0:
            label1 = None
            label2 = None
        else:
            label1 = 'doc1'
            label2 = 'doc2'
        plt.barh(topic_keywords, topic_vector1[i], alpha=0.1, label=label1, color='r')
        plt.barh(topic_keywords, topic_vector2[i], alpha=0.1, label=label2, color='b')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':

    from src.loaders.load_data import load_data

    opt = {'dataset': 'Quora', 'datapath': 'data/',
           'tasks': ['B'],
           'subsets': ['train',
                       'dev',
                       'test'],
                         'max_m':100,
           'num_topics': 50, 'topic_type': 'ldamallet','topic':'word+doc',
           'topic_alpha':10#,'unk_topic':'zero',#'unflat_topics':True #,'topic_update':True
           }
    data_dict = load_data(opt)


    sentence1 = data_dict['R1'][0][0]
    sentence2 = data_dict['R2'][0][0]
    topic_vector1 = data_dict['D_T1'][0][0]
    topic_vector2 = data_dict['D_T2'][0][0]
    topic_table = data_dict['topic_keys']
    visualise_doc_topic_pair_pred(sentence1, sentence2, topic_vector1, topic_vector2, topic_table, figsize=(5,10),n_keywords=None, alpha=opt['topic_alpha'])