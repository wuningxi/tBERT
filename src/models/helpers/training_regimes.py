import tensorflow as tf

def standard_training_regime(optimizer_choice, cost, learning_rate, epsilon, rho):
    # normal setting with only one learning rate and optimizer for all variables
    with tf.name_scope('train'):
        if optimizer_choice == 'Adam':
            train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        elif optimizer_choice == 'Adadelta':
            train_step = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, epsilon=epsilon, rho=rho).minimize(cost)
        else:
            raise NotImplementedError()
    return train_step

def layer_specific_regime(optimizer_choice, cost, learning_rate_old_layers, learning_rate_new_layers, epsilon, rho):
    '''
    Using layer-specific learning rates for BERT vs. newer layers which can be changed during training (e.g. freeze --> unfreeze)
    :param optimizer_choice: Adam or Adadelta
    :param cost: cost tensor
    :param learning_rate_old_layers: placeholder
    :param learning_rate_new_layers: placeholder
    :param epsilon:
    :param rho:
    :return: update op (combining bert optimizer and new layer optimizer)
    '''
    # based on https://stackoverflow.com/questions/34945554/how-to-set-layer-wise-learning-rate-in-tensorflow
    trainable_vars = tf.trainable_variables() # huge list of trainable model variables
    # separate existing and new variables based on name (not position as previously)
    bert_vars = []
    new_vars = []
    for t in trainable_vars:
        if t.name.startswith('bert_lookup/'):
            bert_vars.append(t)
        else:
            new_vars.append(t)
    # create optimizers with different learning rates
    with tf.name_scope('train'):
        if optimizer_choice == 'Adam':
            old_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_old_layers, name='old_optimizer')
            new_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_new_layers, name='new_optimizer')
        elif optimizer_choice == 'Adadelta':
            old_optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate_old_layers, epsilon=epsilon, rho=rho,name='old_optimizer')
            new_optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate_new_layers, epsilon=epsilon, rho=rho,name='new_optimizer')
        else:
            raise NotImplementedError()
        # only compute gradients once
        grads = tf.gradients(cost, bert_vars + new_vars)
        # separate gradients from pretrained and new layers
        bert_grads = grads[:len(bert_vars)]
        new_grads = grads[len(bert_vars):]
        # apply optimisers to respective variables and gradients
        train_step_bert = old_optimizer.apply_gradients(zip(bert_grads, bert_vars))
        train_step_new = new_optimizer.apply_gradients(zip(new_grads, new_vars))
        # combine to one operation
        train_step = tf.group(train_step_bert, train_step_new)
    return train_step
