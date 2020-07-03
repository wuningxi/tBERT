import tensorflow as tf
import numpy as np

def maybe_print(elements, names, test_print):
    if test_print:
        for e, n in zip(elements, names):
            print(n + " shape: " + str(e.get_shape()))

def compute_vocabulary_size(files):
    """
    Counts number of distinct vocabulary indices
    :param files: only X, not Y
    :return: size of vocabulary
    """
    vocabulary = set()
    for f in files:
        for row in f:
            for integer in row:
                if integer not in vocabulary:
                    vocabulary.add(integer)
    return max(vocabulary)+1

def create_placeholders(sentence_lengths, classes, bicnn=False, sparse=True, bert=False):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    sentence length -- scalar, width of sentence matrix 
    classes -- scalar, number of classes

    Returns:
    X -- placeholder for the data input, of shape [None, sentence_length] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, classes] and dtype "float"
    """
    sentence_length = sentence_lengths[0]
    if sparse:
        Y = tf.placeholder(tf.int64, [None, ], name='labels')
    else:
        Y = tf.placeholder(tf.int32, [None, classes], name='labels')
    if bert:
        # BERT Placeholders # no names!!
        X1 = tf.placeholder(dtype=tf.int32, shape=[None, None]) # input ids
        X1_mask = tf.placeholder(dtype=tf.int32, shape=[None, None]) # input masks
        X1_seg = tf.placeholder(dtype=tf.int32, shape=[None, None]) # segment ids
        return X1,X1_mask,X1_seg,Y
    else:
        X = tf.placeholder(tf.int32, [None, sentence_length], name='XL')
        if bicnn:
            sentence_length2 = sentence_lengths[1]
            X2 = tf.placeholder(tf.int32, [None, sentence_length2], name='XR')
            return X, X2, Y
        else:
            return X, Y

def create_text_placeholders(sentence_lengths):
    T1 = tf.placeholder(tf.string, [None, sentence_lengths[0]], name='TL')
    T2 = tf.placeholder(tf.string, [None, sentence_lengths[1]], name='TR')
    return T1,T2

def create_word_topic_placeholders(sentence_lengths):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    :param sentence lengths: scalar, width of sentence matrix 
    :param num_topics: number of topics for topic model
    :param dim: dimensions of word topics, should be 2, 3 or None

    Returns:
    X -- placeholder for the data input, of shape [None, sentence_length] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, classes] and dtype "float"
    """
    T1 = tf.placeholder(tf.int32, [None, sentence_lengths[0]], name='W_TL')
    T2 = tf.placeholder(tf.int32, [None, sentence_lengths[1]], name='W_TR')
    return T1,T2

def create_word_topic_placeholder(sentence_length):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    :param sentence lengths: scalar, width of sentence matrix
    :param num_topics: number of topics for topic model
    :param dim: dimensions of word topics, should be 2, 3 or None

    Returns:
    X -- placeholder for the data input, of shape [None, sentence_length] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, classes] and dtype "float"
    """
    WT = tf.placeholder(tf.int32, [None, sentence_length], name='W_T')
    return WT

def create_doc_topic_placeholders(num_topics):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    sentence length -- scalar, width of sentence matrix 
    classes -- scalar, number of classes

    Returns:
    X -- placeholder for the data input, of shape [None, sentence_length] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, classes] and dtype "float"
    """
    T1 = tf.placeholder(tf.float32, [None, num_topics], name='D_TL')
    T2 = tf.placeholder(tf.float32, [None, num_topics], name='D_TR')
    return T1,T2

def create_embedding(vocab_size, embedding_dim,name='embedding'):
    with tf.name_scope(name):
        embedding_matrix = tf.Variable(
            tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0),
            name="W", trainable=True)
    return embedding_matrix

def create_embedding_placeholder(doc_vocab_size, embedding_dim):
    # use this or load data directly as variable?
    embedding_placeholder = tf.placeholder(tf.float32, [doc_vocab_size, embedding_dim], name='embd_placeholder')
    return embedding_placeholder

def initialise_pretrained_embedding(doc_vocab_size, embedding_dim, embedding_placeholder, name='embedding',trainable=True):
    with tf.name_scope(name):
        if trainable:
            print('init pretrained embds')
            embedding_matrix = tf.Variable(embedding_placeholder, trainable=True, name="W",dtype=tf.float32)
        else:
            W = tf.Variable(tf.constant(0.0, shape=[doc_vocab_size, embedding_dim]), trainable=False, name="W")
            embedding_matrix = W.assign(embedding_placeholder)
    return embedding_matrix

def lookup_embedding(X, embedding_matrix,expand=True,transpose=True,name='embedding_lookup'):
    '''
    Looks up embeddings based on word ids
    :param X: word id matrix with shape (m, sentence_length)
    :param embedding_matrix: embedding matrix with shape (vocab_size, embedding_dim)
    :param expand: add dimension to embedded matrix or not
    :param transpose: switch dimensions of embedding matrix or not
    :param name: name used in TF graph
    :return: embedded_matrix
    '''
    embedded_matrix = tf.nn.embedding_lookup(embedding_matrix, X, name=name) # dim [m, sentence_length, embedding_dim]
    if transpose:
        embedded_matrix = tf.transpose(embedded_matrix, perm=[0, 2, 1]) # dim [m, embedding_dim, sentence_length]
    if expand:
        embedded_matrix = tf.expand_dims(embedded_matrix, -1) # dim [m, embedding_dim, sentence_length, 1]
    return embedded_matrix

def compute_cost(logits, Y, loss_fn='cross_entropy', name='main_cost'):
    """
    Computes the cost

    Arguments:
    logits -- output of forward propagation (output of the last LINEAR unit of shape (batch, classes)
    Y -- "true" labels vector of shape (batch,)

    Returns:
    cost - Tensor of the cost function
    """
    # multi class classification (binary classification as special case)
    with tf.name_scope(name):
        if loss_fn=='cross_entropy':
            # maybe_print([logits,Y], ['logits','Y'], True)
            cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y),name='cost')
        elif loss_fn=='bert':
            # from https://github.com/google-research/bert/blob/master/run_classifier_with_tfhub.py
            # probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(Y, depth=2, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            cost = tf.reduce_mean(per_example_loss)
        else:
            raise NotImplemented()
    return cost



