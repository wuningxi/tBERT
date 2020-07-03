import numpy as np
import math

def mini_batches_bert(X, X_mask, X_seg, Y, mini_batch_size=64, seed=0, sparse=True,random=True):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    # sanity check shapes
    assert type(X)==np.ndarray
    assert type(X_mask)==np.ndarray
    assert type(X_seg)==np.ndarray
    assert X.shape==X_mask.shape==X_seg.shape

    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    if random:
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation, :]
        shuffled_X_mask = X_mask[permutation, :] # would have raised an error for if m(X_mask)>m(X)
        shuffled_X_seg = X_seg[permutation, :]
        if sparse:
            shuffled_Y = Y[permutation,]
        else:
            shuffled_Y = Y[permutation, :]
    else:
        shuffled_X = X
        shuffled_X_mask = X_mask
        shuffled_X_seg = X_seg
        shuffled_Y = Y

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_mask = shuffled_X_mask[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_seg = shuffled_X_seg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        if sparse:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, ]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = {'E1':mini_batch_X, 'E1_mask':mini_batch_X_mask, 'E1_seg':mini_batch_X_seg, 'Y':mini_batch_Y}
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_mask = shuffled_X_mask[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_seg = shuffled_X_seg[num_complete_minibatches * mini_batch_size: m, :]
        if sparse:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m,]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = {'E1': mini_batch_X, 'E1_mask': mini_batch_X_mask, 'E1_seg': mini_batch_X_seg, 'Y': mini_batch_Y}
        mini_batches.append(mini_batch)
    return mini_batches

# minibatching for document topics +BERT
def doc_topic_mini_batches_bert(X, X_mask, X_seg, D_T1, D_T2, Y, mini_batch_size=64, seed=0, sparse=True, random=True):
    '''
    Creates a list of not random minibatches from (X, Y) for prediction in document topic models

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    sparse -- is label representation sparse or not
    shuffle -- should data be shuffled? True for training, False for prediction

    Returns:
    mini_batches -- list of synchronous (mini_batch_X1, mini_batch_X2, mini_batch_T1, mini_batch_T2,mini_batch_Y)
    :param X1: input data with embedding ids with shape (batch, sent_len_1)
    :param X2: input data with embedding ids with shape (batch, sent_len_2)
    :param D_T1: document topic with shape (batch, num_topics)
    :param D_T2: document topic with shape (batch, num_topics)
    :param Y: labels with shape (batch, 1)
    :param mini_batch_size:
    :param seed:
    :param sparse:
    :param random:
    :return:
    '''
    # sanity check shapes
    assert type(X)==np.ndarray
    assert type(X_mask)==np.ndarray
    assert type(X_seg)==np.ndarray
    assert X.shape==X_mask.shape==X_seg.shape
    assert(type(D_T1) == np.ndarray)
    assert(type(D_T2) == np.ndarray)
    assert(D_T1.shape[0] == D_T2.shape[0]) # identical number of examples
    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y) if random
    if random:
        permutation = list(np.random.permutation(m))
        # documents
        shuffled_X = X[permutation, :]
        shuffled_X_mask = X_mask[permutation, :]
        shuffled_X_seg = X_seg[permutation, :]
        # topics
        shuffled_D_T1 = D_T1[permutation, :]
        shuffled_D_T2 = D_T2[permutation, :]
        if sparse:
            shuffled_Y = Y[permutation]
        else:
            shuffled_Y = Y[permutation, :]
    else:
        shuffled_X = X
        shuffled_X_mask = X_mask
        shuffled_X_seg = X_seg
        shuffled_D_T1 = D_T1
        shuffled_D_T2 = D_T2
        shuffled_Y = Y

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    # print(mini_batch_size)
    num_complete_minibatches = math.floor(m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        # documents
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_mask = shuffled_X_mask[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_seg = shuffled_X_seg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        # topics
        mini_batch_D_T1 = shuffled_D_T1[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_D_T2 = shuffled_D_T2[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        if sparse:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = {'E1':mini_batch_X, 'E1_mask':mini_batch_X_mask, 'E1_seg':mini_batch_X_seg, 'D_T1':mini_batch_D_T1, 'D_T2':mini_batch_D_T2, 'Y':mini_batch_Y}
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        # documents
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_mask = shuffled_X_mask[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_seg = shuffled_X_seg[num_complete_minibatches * mini_batch_size: m, :]
        # topics
        mini_batch_D_T1 = shuffled_D_T1[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_D_T2 = shuffled_D_T2[num_complete_minibatches * mini_batch_size: m, :]
        if sparse:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = {'E1':mini_batch_X, 'E1_mask':mini_batch_X_mask, 'E1_seg':mini_batch_X_seg, 'D_T1':mini_batch_D_T1, 'D_T2':mini_batch_D_T2, 'Y':mini_batch_Y}
        mini_batches.append(mini_batch)

    return mini_batches

# minibatching for word topics + BERT
def word_topic_mini_batches_bert(X, X_mask, X_seg, W_T1, W_T2, Y, mini_batch_size=64, seed=0, sparse=True, random=True):
    '''
    Creates a list of not random minibatches from (X, Y) for prediction in word topic models

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    sparse -- is label representation sparse or not
    shuffle -- should data be shuffled? True for training, False for prediction

    Returns:
    mini_batches -- list of synchronous (mini_batch_X1, mini_batch_X2, mini_batch_T1, mini_batch_T2,mini_batch_Y)
    :param X1: input data with embedding ids with shape (batch, sent_len_1)
    :param X2: input data with embedding ids with shape (batch, sent_len_2)
    :param W_T1: word topics with shape (batch, sent_len_1)
    :param W_T2: word topics with shape (batch, sent_len_2)
    :param Y: labels with shape (batch, 1)
    :param mini_batch_size:
    :param seed:
    :param sparse:
    :param random:
    :return:
    '''
    # sanity check shapes
    assert type(X)==type(X_mask)==type(X_seg)==np.ndarray
    assert X.shape==X_mask.shape==X_seg.shape
    assert type(W_T1) == type(W_T2) == np.ndarray
    assert len(W_T1.shape) == len(W_T2.shape) == 2, 'Word topics should be 2dim, but were {}, {}.'.format(W_T1.shape, W_T2.shape)  # word topics have 3 dims!
    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y) if random
    if random:
        permutation = list(np.random.permutation(m))
        # documents
        shuffled_X = X[permutation, :]
        shuffled_X_mask = X_mask[permutation, :]
        shuffled_X_seg = X_seg[permutation, :]
        # topics
        shuffled_W_T1 = W_T1[permutation, :]
        shuffled_W_T2 = W_T2[permutation, :]
        if sparse:
            shuffled_Y = Y[permutation]
        else:
            shuffled_Y = Y[permutation, :]
    else:
        shuffled_X = X
        shuffled_X_mask = X_mask
        shuffled_X_seg = X_seg
        shuffled_W_T1 = W_T1
        shuffled_W_T2 = W_T2
        shuffled_Y = Y

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    # print(mini_batch_size)
    num_complete_minibatches = math.floor(m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        # documents
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_mask = shuffled_X_mask[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_seg = shuffled_X_seg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        # topics
        mini_batch_W_T1 = shuffled_W_T1[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_W_T2 = shuffled_W_T2[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        if sparse:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = {'E1':mini_batch_X, 'E1_mask':mini_batch_X_mask,'E1_seg':mini_batch_X_seg,'W_T1':mini_batch_W_T1, 'W_T2':mini_batch_W_T2, 'Y':mini_batch_Y}
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        # documents
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_mask = shuffled_X_mask[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_seg = shuffled_X_seg[num_complete_minibatches * mini_batch_size: m, :]
        # topics
        mini_batch_W_T1 = shuffled_W_T1[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_W_T2 = shuffled_W_T2[num_complete_minibatches * mini_batch_size: m, :]
        if sparse:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = {'E1':mini_batch_X, 'E1_mask':mini_batch_X_mask,'E1_seg':mini_batch_X_seg,'W_T1':mini_batch_W_T1, 'W_T2':mini_batch_W_T2, 'Y':mini_batch_Y}
        mini_batches.append(mini_batch)

    return mini_batches

def create_minibatches(subset, mini_batch_size=64,seed=0, sparse=True, random=True, topic_scope=''):
    '''
    Distinguishes which minibatch function is needed and passes arguments on to specific functions
    :param X1: document 1
    :param X2: document 2
    :param T1: topic distribution for document 1 or None
    :param T2: topic distribution for document 1 or None
    :param Y: labels
    :param mini_batch_size: number of examples in each minibatch
    :param seed: seed for shuffling
    :param sparse: sparse label representation?
    :param random: random minibatches? (True for training, False for prediction)
    :param dim: dimensions of word topics, should be 2, 3 or None
    :return: minibatches of X1,X2,(T1,T2),Y
    '''
    # X1, X2, T1, T2, Y
    # additional token input for elmo
    if 'E1_mask' in subset: # BERT
        if topic_scope=='':
            return mini_batches_bert(subset['E1'],subset['E1_mask'],subset['E1_seg'], subset['Y'], mini_batch_size, seed, sparse, random)
        elif 'doc' in topic_scope:
            return doc_topic_mini_batches_bert(subset['E1'],subset['E1_mask'],subset['E1_seg'], subset['D_T1'], subset['D_T2'], subset['Y'], mini_batch_size, seed, sparse, random)
        elif 'word' in topic_scope:
            return word_topic_mini_batches_bert(subset['E1'],subset['E1_mask'],subset['E1_seg'], subset['W_T1'], subset['W_T2'], subset['Y'], mini_batch_size, seed, sparse, random)
        else:
            raise NotImplementedError()
            raise ValueError('topic_scope should be "","doc" or "word".')
    else:
        raise ValueError("Minibatch should contain 'E1_mask'")

if __name__=='__main__':

    from src.loaders.load_data import load_data
    from src.models.helpers.base import extract_data

    # Example usage
    opt = {'dataset': 'Semeval', 'datapath': 'data/',
           'tasks': ['B'],'max_length':'minimum',
           'subsets': ['train_large','test2016','test2017'],
           'model': 'bert', 'load_ids':True,'lemmatize':False,
           # 'topic':'word','num_topics':20,'topic_type':'ldamallet'
           }

    data_dict = load_data(opt)

    pretrain,train,dev,test = extract_data(data_dict, topic_scope=opt.get('topic', ''), extra_test=False)

    train_batches = create_minibatches(train, mini_batch_size=64, seed=0, sparse=True, random=True, topic_scope=opt.get('topic',''))

    for batch in train_batches:
        for k in batch.keys():
            print('{} shape: {}\n{}'.format(k,batch[k].shape,batch[k]))
        break