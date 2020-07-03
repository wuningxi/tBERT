import tensorflow as tf
from src.models.tf_helpers import maybe_print

def forward_propagation(input_dict, classes, hidden_layer=0, reduction_factor=2, dropout=0, seed_list=[], print_dim=False):
    """
    Defines forward pass for tBERT model

    Returns: logits
    """
    if print_dim:
        print('---')
        print('Model: tBERT')
        print('---')

    # word topics
    if input_dict['D_T1'] is None:
        W_T1 = input_dict['W_T1'] # (batch, sent_len_1, num_topics)
        W_T2 = input_dict['W_T2'] # (batch, sent_len_2, num_topics)
        maybe_print([W_T1, W_T2], ['W_T1', 'W_T2'], print_dim)
        # compute mean of word topics
        D_T1 = tf.reduce_mean(W_T1,axis=1) # (batch, num_topics)
        D_T2 = tf.reduce_mean(W_T2,axis=1) # (batch, num_topics)

    # document topics
    elif input_dict['W_T1'] is None:
        D_T1 = input_dict['D_T1'] # (batch, num_topics)
        D_T2 = input_dict['D_T2'] # (batch, num_topics)

    else:
        ValueError('Word or document topics need to be provided for bert_simple_topic.')
    maybe_print([D_T1,D_T2], ['D_T1','D_T2'], print_dim)

    # bert representation
    bert = input_dict['E1']
    # bert has 2 keys: sequence_output which is output embedding for each token and pooled_output which is output embedding for the entire sequence.
    with tf.name_scope('bert_rep'):
        # pooled output (containing extra dense layer)
        bert_rep = bert['pooled_output'] # pooled output over entire sequence
        maybe_print([bert_rep], ['pooled BERT'], print_dim)

        # C vector from last layer corresponding to CLS token
        # bert_rep = bert['sequence_output'][:, 0, :]  # shape (batch, BERT_hidden)
        # maybe_print([bert_rep], ['BERT C vector'], print_dim)
        bert_rep = tf.layers.dropout(inputs=bert_rep, rate=dropout, seed=seed_list.pop(0))

    # combine BERT with document topics
    combined = tf.concat([bert_rep, D_T1, D_T2], -1)
    maybe_print([combined], ['combined'], print_dim)

    if hidden_layer>0:
        with tf.name_scope('hidden_1'):
            hidden_size = combined.shape[-1].value/reduction_factor
            combined = tf.layers.dense(
                combined,
                hidden_size,
                activation=tf.tanh,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02, seed=seed_list.pop(0)))
        maybe_print([combined], ['hidden 1'], print_dim)
        combined = tf.layers.dropout(inputs=combined, rate=dropout, seed=seed_list.pop(0))

    if hidden_layer>1:
        with tf.name_scope('hidden_2'):
            hidden_size = combined.shape[-1].value/reduction_factor
            combined = tf.layers.dense(
                combined,
                hidden_size,
                activation=tf.tanh,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02, seed=seed_list.pop(0)))
        maybe_print([combined], ['hidden 2'], print_dim)
        combined = tf.layers.dropout(inputs=combined, rate=dropout, seed=seed_list.pop(0))

    if hidden_layer>2:
        raise ValueError('Only 2 hidden layers supported.')

    with tf.name_scope('output_layer'):
        hidden_size = combined.shape[-1].value
        output_weights = tf.get_variable(
            "output_weights", [classes, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02,seed=seed_list.pop(0)))
        output_bias = tf.get_variable(
            "output_bias", [classes], initializer=tf.zeros_initializer())
        logits = tf.matmul(combined, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
    maybe_print([logits], ['output layer'], print_dim)

    output = {'logits':logits}

    return output