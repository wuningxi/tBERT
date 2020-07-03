import math
import traceback
import numpy as np
import sys
import importlib
import argparse
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.framework import ops
from src.models.tf_helpers import lookup_embedding,initialise_pretrained_embedding
from src.models.tf_helpers import compute_cost
from src.models.tf_helpers import create_placeholders,create_doc_topic_placeholders,create_word_topic_placeholders
from src.models.helpers.minibatching import create_minibatches
from src.models.tf_helpers import maybe_print
import tensorflow_hub as hub

import random

np.random.seed(1)


from src.loaders.load_data import load_data
from src.logs.training_logs import write_log_entry, start_timer, end_timer, get_new_id
from src.models.save_load import save_model, load_model, create_saver, get_model_dir, create_model_folder, \
    delete_all_checkpoints_but_best
from src.evaluation.evaluate import output_predictions, get_confidence_scores, save_eval_metrics
from src.models.helpers.base import add_git_version,skip_MAP,extract_data
from src.models.helpers.bert import get_bert_version
from src.models.helpers.training_regimes import standard_training_regime,layer_specific_regime

import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
# warnings.filterwarnings('error')


"""
Implements Base Model with early stopping and auxiliary loss
"""

predefined_opts = [
    # loader related keywords
    'dataset', 'datapath', 'tasks', 'subsets', 'w2v_limit', 'max_m', 'max_length', 'n_gram_embd', 'padding', 'unk_sub',
    'simple_padding',
    # model related keywords
    'model', 'load_ids', 'learning_rate', 'num_epochs', 'minibatch_size',
    'bert_update','bert_cased','bert_large','optimizer', 'epsilon', 'rho', 'L2',
    'sparse_labels', 'dropout',  'hidden_reduce',
    'hidden_layer', 'checkpoints', 'stopping_criterion','model',
    # topid related keywords
    'patience','unk_topic','topic_encoder','unflat_topics','topic_update','topic_alpha','num_topics',
    'topic_type','topic','injection_location',
    # other keywords
    'speedup_new_layers','freeze_thaw_tune','predict_every_epoch','gpu','git','seed']

def test_opt(opt):
    '''
    Test if opt contains unused or wrong
    :param opt:
    :return:
    '''
    for k in opt.keys():
        assert k in predefined_opts, '{} not in accepted options'.format(k)

def model(data_dict, opt, logfile=None, print_dim=False):
    """
    Creates and executes Tensorflow graph for BERT-based models

    Arguments:
    data_dict -- contains all necessary data for model

    opt -- option log, contains learning_rate, num_epochs, minibatch_size, ...
    logfile -- path of file to save opt and results
    print_dim -- print dimensions for debugging purposes

    Returns:
    opt -- updated option log
    parameters -- trained parameters of model
    """

    #####
    # Read options, set defaults and update log
    #####

    try:
        # check input options
        print(opt)
        test_opt(opt)
        if opt.get('git',None) is None:
            add_git_version(opt) # keep track of git SHA

        # assign variables
        opt['model'] = opt.get('model','bert')
        assert 'bert' in opt['model']
        learning_rate = opt['learning_rate'] = opt.get('learning_rate', 5e-5) # small learning rate for pretrained BERT layers
        speedup_new_layers = opt['speedup_new_layers'] = opt.get('speedup_new_layers', False)
        freeze_thaw_tune = opt['freeze_thaw_tune'] = opt.get('freeze_thaw_tune', False)
        layer_specific_lr = speedup_new_layers or freeze_thaw_tune
        num_epochs = opt.get('num_epochs', None) # get num of planned epochs
        opt['num_epochs'] = 0 # use this to keep track of finished epochs
        minibatch_size = opt['minibatch_size'] = opt.get('minibatch_size', 64)
        bert_embd = True
        bert_update = opt['bert_update'] = opt.get('bert_update', False)
        bert_large = opt['bert_large'] = opt.get('bert_large', False)
        cased = opt['bert_cased'] = opt.get('bert_cased', False)
        starter_seed = opt['seed'] = opt.get('seed', None)
        if not type(starter_seed) == int:
            assert starter_seed == None
        # layers = opt['layers'] = opt.get('layers', 1)
        hidden_layer = opt['hidden_layer'] = opt.get('hidden_layer', 0)  # add hidden layer before softmax layer?
        assert hidden_layer in [0,1,2]
        topic_encoder = opt['topic_encoder'] = opt.get('topic_encoder', None)
        L_R_unk = opt.get('unk_sub',False)
        assert L_R_unk is False
        # assert encoder in ['word', 'ffn', 'cnn', 'lstm', 'bilstm', 'word+cnn', 'word+ffn', 'word+lstm', 'word+bilstm']
        assert topic_encoder in [None, 'ffn', 'cnn', 'lstm', 'bilstm']
        optimizer_choice = opt['optimizer'] = opt.get('optimizer', 'Adadelta')  # which optimiser to use?
        assert optimizer_choice in ['Adam','Adadelta']
        epsilon = opt['epsilon'] = opt.get('epsilon', 1e-08)
        rho = opt['rho'] = opt.get('rho', 0.95)
        L2 = opt['L2'] = opt.get('L2', 0)  # L2 regularisation
        dropout = opt['dropout'] = opt.get('dropout', 0)
        assert not (L2 > 0 and dropout > 0), 'Use dropout or L2 regularisation, not both. Current settings: L2={}, dropout={}.'.format(L2, dropout)
        sparse = opt['sparse_labels'] = opt.get('sparse_labels', True)  # are labels encoded as sparse?
        save_checkpoints = opt.get('checkpoints', False)  # save all checkpoints?
        stopping_criterion = opt['stopping_criterion'] = opt.get('stopping_criterion',None)  # which metric should be used as early stopping criterion?
        assert stopping_criterion in [None, 'cost', 'MAP', 'F1', 'Accuracy']
        if stopping_criterion is None and num_epochs is None:
            raise ValueError('Invalid parameter combination. Stopping criterion and number of epochs cannot both be None.')
        early_stopping = stopping_criterion in ['F1', 'cost', 'MAP', 'Accuracy']
        predict_every_epoch = opt['predict_every_epoch'] = opt.get('predict_every_epoch', False)
        reduction_factor = opt['hidden_reduce'] = opt.get('hidden_reduce', 2)
        patience = opt['patience'] = opt.get('patience',20)
        # topic models
        topic_scope = opt['topic'] = opt.get('topic', '')
        if opt['model']=='bert_simple_topic':
            assert topic_scope in ['word', 'doc']
        elif opt['model']=='bert':
            topic_scope = ''
        else:
            raise NotImplementedError()
        module_name = "src.models.forward.{}".format(opt['model'])
        model = importlib.import_module(module_name)
        if 'word' in topic_scope:
            topic_update = opt['topic_update'] = opt.get('topic_update', False) # None for backward compatibility
        num_topics = opt['num_topics'] = opt.get('num_topics', None)
        topic_type = opt['topic_type'] = opt.get('topic_type', None)
        if not topic_scope == '':
            assert 'topic' in opt['model']
            assert num_topics > 1
            assert topic_type in ['LDA','ldamallet','gsdmm']
            opt['topic_alpha'] = opt.get('topic_alpha',50)
        else:
            assert num_topics is None
            assert topic_type is None
        if opt['dataset']=='Quora' and opt['subsets']==['train','dev','test','p_test']:
            extra_test = True
        else:
            extra_test = False
        injection_location = opt['injection_location'] = opt.get('injection_location',None)
        if 'inject' in opt['model']:
            assert str(injection_location) in ['embd','0','1','2','3','4','5','6','7','8','9','10','11']
        else:
            assert injection_location is None
        # gpu settings
        gpu = opt.get('gpu', -1)

        # general settings
        session_config = tf.ConfigProto()
        if not gpu == -1:
            print('Running on GPU: {}'.format(gpu))
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)  # specifies which GPU to use (if multiple are available)

        ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables

        if not starter_seed == None:
            random.seed(starter_seed) # use starter seed to set seed for random library
        seed_list = [random.randint(1, 100000) for i in range(100)] # generate list of seeds to be used in the model

        np.random.seed(seed_list.pop(0))
        tf.set_random_seed(seed_list.pop(0))  # set tensorflow seed to keep results consistent

        #####
        # unpack data and assign to model variables
        #####

        assert data_dict.get('embd', None) is None# (565852, 200)

        if 'word' in topic_scope:
            topic_embd = data_dict['word_topics'].get('topic_matrix', None) #topic_emb.shape

        # assign word ids
        if extra_test:
            ID1_train, ID1_dev, ID1_test, ID1_test_extra = data_dict['ID1']
            ID2_train, ID2_dev, ID2_test, ID2_test_extra = data_dict['ID2']
        else:
            ID1_train, ID1_dev, ID1_test = data_dict['ID1']
            ID2_train, ID2_dev, ID2_test = data_dict['ID2']
        train_dict,dev_dict,test_dict,test_dict_extra = extract_data(data_dict, topic_scope, extra_test)

        #####
        # check input dimensions
        #####

        if sparse:
            classes = 2
        else:
            classes = train_dict['Y'].shape[1]
        (m, sentence_length_1) = train_dict['E1'].shape
        # (m, sentence_length_2) = train_dict['E2'].shape

        #####
        # Define Tensorflow graph
        #####

        # Create Placeholders and initialise weights of the correct shape
        X1,X1_mask,X1_seg,Y = create_placeholders([sentence_length_1, None], classes, bicnn=True, sparse=sparse, bert=bert_embd)

        # Create topic placeholders
        print('Topic scope: {}'.format(topic_scope))
        if 'doc' in topic_scope:
            D_T1, D_T2 = create_doc_topic_placeholders(num_topics)
        else:
            D_T1,D_T2 = None,None
        if 'word' in topic_scope:
            W_T_embedded = None
            (m, sentence_length_1) = train_dict['W_T1'].shape
            (m, sentence_length_2) = train_dict['W_T2'].shape
            W_T1, W_T2 = create_word_topic_placeholders([sentence_length_1, sentence_length_2])
        else:
            W_T1_embedded, W_T2_embedded, W_T_embedded = None, None, None

        # tensors for feed_dict
        bert_inputs = dict(
            input_ids=X1,
            input_mask=X1_mask,
            segment_ids=X1_seg)
        maybe_print([X1],['input ids'],True)

        dropout_prob = tf.placeholder_with_default(0.0, name='dropout_rate', shape=())

        # load and lookup BERT
        BERT_version = get_bert_version(cased,bert_large)
        BERT_URL = 'https://tfhub.dev/google/bert_{}/1'.format(BERT_version)
        print('Loading pretrained model from {}'.format(BERT_URL))
        bert_lookup = hub.Module(BERT_URL,name='bert_lookup',trainable=bert_update)
        X_embedded = bert_lookup(bert_inputs, signature="tokens", as_dict=True) # important to use tf. 1.11 as tf 1.7 will produce error for sess.run(X_embedded)
        # X_embedded has 2 keys:
        # pooled_output is [batch_size, hidden_size] -->output embedding for each token
        # sequence_output is [batch_size, sequence_length, hidden_size] -->output embedding for the entire sequence

        # Create topic embedding matrix
        if 'word' in topic_scope:
            topic_vocabulary_size, topic_dim = topic_embd.shape
            # assert(topic_vocabulary_size==embd_vocabulary_size) # currently using the same id to index topic and embd matrix
            topic_embedding_matrix = initialise_pretrained_embedding(topic_vocabulary_size, topic_dim, topic_embd,
                                                                     name='word_topics', trainable=topic_update)
            # Lookup topic embedding
            W_T1_embedded = lookup_embedding(W_T1, topic_embedding_matrix, expand=False,transpose=False,name='topic_lookup_L')
            W_T2_embedded = lookup_embedding(W_T2, topic_embedding_matrix, expand=False,transpose=False,name='topic_lookup_R')

        # Forward propagation: Build forward propagation as tensorflow graph
        input_dict = {'E1':X_embedded,'E2':None,'D_T1':D_T1,'D_T2':D_T2,'W_T1':W_T1_embedded,'W_T2':W_T2_embedded,'W_T':W_T_embedded}
        forward_pass = model.forward_propagation(input_dict, classes, hidden_layer, reduction_factor, dropout_prob, seed_list, print_dim)
        logits = forward_pass['logits']

        with tf.name_scope('cost'):
            # Cost function: Add cost function to tensorflow graph
            main_cost = compute_cost(logits, Y,loss_fn='bert')
            cross_entropy_scalar = tf.summary.scalar('cross_entropy', main_cost)
            cost = main_cost
            cost_summary = tf.summary.merge([cross_entropy_scalar])

        # Backpropagation: choose training regime (creates tensorflow optimizer which minimizes the cost).

        if layer_specific_lr:
            learning_rate_old_layers = tf.placeholder_with_default(0.0, name='learning_rate_old', shape=())
            learning_rate_new_layers = tf.placeholder_with_default(0.0, name='learning_rate_new', shape=())
            train_step = layer_specific_regime(optimizer_choice, cost, learning_rate_old_layers, learning_rate_new_layers, epsilon, rho)
        else:
            learning_rate_old_layers = tf.placeholder_with_default(0.0, name='learning_rate', shape=())
            learning_rate_new_layers = None
            train_step = standard_training_regime(optimizer_choice, cost, learning_rate_old_layers, epsilon, rho)

        # Prediction and Evaluation tensors

        with tf.name_scope('evaluation_metrics'):
            predicted_label = tf.argmax(logits, 1, name='predict')  # which column is the one with the highest activation value?
            if sparse:
                actual_label = Y
            else:
                actual_label = tf.argmax(Y, 1)
            conf_scores = get_confidence_scores(logits, False)
            maybe_print([predicted_label, actual_label, conf_scores],['Predicted label', 'Actual label', 'Confidence Scores'], False)

            # create streaming metrics: http://ronny.rest/blog/post_2017_09_11_tf_metrics/
            streaming_accuracy, streaming_accuracy_update = tf.metrics.accuracy(labels=actual_label, predictions=predicted_label)
            label_idx = tf.expand_dims(tf.where(tf.not_equal(Y, 0))[:, 0], 0,name='label_idx')
            rank_scores = tf.expand_dims(get_confidence_scores(logits), 0,name='rank_scores')
            maybe_print([label_idx, rank_scores], ['Label index', 'Rank scores'], False)
            streaming_map, streaming_map_update = tf.metrics.average_precision_at_k(label_idx, rank_scores, 10)
            # fixed NaN for examples without relevant docs by editing .virtualenvs/tensorflow/lib/python3.6/site-packages/tensorflow/python/ops/metrics_impl.py line 2796
            # return math_ops.div(precision_sum, num_relevant_items + 1e-11, name=scope)
            streaming_recall, streaming_recall_update = tf.contrib.metrics.streaming_recall(predictions=predicted_label,labels=actual_label)
            streaming_precision, streaming_precision_update = tf.contrib.metrics.streaming_precision(predictions=predicted_label, labels=actual_label)
            eps = 1e-11  # prevent division by zero
            streaming_f1 = 2 * (streaming_precision * streaming_recall) / (
            streaming_precision + streaming_recall + eps)
            # create and merge summaries
            accuracy_scalar = tf.summary.scalar('Accuracy', streaming_accuracy)
            recall_scalar = tf.summary.scalar('Recall', streaming_recall)
            precision_scalar = tf.summary.scalar('Precision', streaming_precision)
            f1_scalar = tf.summary.scalar('F1', streaming_f1)
            map_scalar = tf.summary.scalar('MAP', streaming_map)
            eval_summary = tf.summary.merge(
                [accuracy_scalar, recall_scalar, precision_scalar, f1_scalar, map_scalar])

        def predict(sess,subset, writer, epoch, ignore_MAP,topic_scope,layer_specific_lr):
            '''
            Predict in minibatch loop to prevent out of memory error (for large datasets or complex models)
            :param input_X1: document 1
            :param input_X2: document 2
            :param input_T1: topic distributions for document 1 or None
            :param input_T2: topic distributions for document 2 or None
            :param input_Y: labels
            :param writer:
            :param epoch:
            :param ignore_MAP:
            :return: complete prediction results as list [confidence_scores, predictions, minibatch_cost, eval_metrics]
            '''
            # print(input_T1)
            predictions = []
            confidence_scores = []
            minibatch_size = 10
            minibatches = create_minibatches(subset, minibatch_size, sparse=sparse,random=False,topic_scope=topic_scope)
            sess.run(tf.local_variables_initializer())  # for streaming metrics

            for minibatch in minibatches:
                feed_dict = {X1: minibatch['E1'],
                             X1_mask: minibatch['E1_mask'],
                             X1_seg: minibatch['E1_seg'],
                             # X2: minibatch['E2'],
                             Y: minibatch['Y'],
                             learning_rate_old_layers: 0,
                             dropout_prob: 0} # don't use dropout during prediction
                if layer_specific_lr:
                    feed_dict[learning_rate_new_layers] = 0
                if 'doc' in topic_scope:
                    feed_dict[D_T1] = minibatch['D_T1']
                    feed_dict[D_T2] = minibatch['D_T2']
                if 'word' in topic_scope:
                    feed_dict[W_T1] = minibatch['W_T1']
                    feed_dict[W_T2] = minibatch['W_T2']
                # Run the session to execute the prediction and evaluation, the feedict should contain a minibatch for (X,Y).
                pred, conf = sess.run( # evaluating merged_summary will mess up streaming metrics
                     [predicted_label, conf_scores],
                    feed_dict=feed_dict)
                predictions.extend(pred)
                confidence_scores.extend(conf)

            if not ignore_MAP:
                eval_metrics = [None, None,None,None,None]
            else:
                eval_metrics = [None,None,None,None]
            return confidence_scores, predictions, None, eval_metrics


        def predict_eval(sess,subset, writer, epoch, ignore_MAP,topic_scope,layer_specific_lr):
            '''
            Predict in minibatch loop to prevent out of memory error (for large datasets or complex models)
            :param input_X1: document 1
            :param input_X2: document 2
            :param input_T1: topic distributions for document 1 or None
            :param input_T2: topic distributions for document 2 or None
            :param input_Y: labels
            :param writer:
            :param epoch:
            :param ignore_MAP:
            :return: complete prediction results as list [confidence_scores, predictions, minibatch_cost, eval_metrics]
            '''
            # print(input_T1)
            predictions = []
            confidence_scores = []
            minibatch_size = 10
            minibatches = create_minibatches(subset, minibatch_size, sparse=sparse,random=False,topic_scope=topic_scope)
            sess.run(tf.local_variables_initializer())  # for streaming metrics

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set

            for minibatch in minibatches:
                feed_dict = {X1: minibatch['E1'],
                             X1_mask: minibatch['E1_mask'],
                             X1_seg: minibatch['E1_seg'],
                             # X2: minibatch['E2'],
                             Y: minibatch['Y'],
                             learning_rate_old_layers: 0,
                             dropout_prob: 0}  # don't use dropout during prediction
                if layer_specific_lr:
                    feed_dict[learning_rate_new_layers] = 0
                if 'doc' in topic_scope:
                    feed_dict[D_T1] = minibatch['D_T1']
                    feed_dict[D_T2] = minibatch['D_T2']
                if 'word' in topic_scope:
                    feed_dict[W_T1] = minibatch['W_T1']
                    feed_dict[W_T2] = minibatch['W_T2']
                # Run the session to execute the prediction and evaluation, the feeddict should contain a minibatch for (X,Y).
                if not ignore_MAP:
                    # print('with MAP')
                    pred, conf, batch_cost, c, _, _, _, _ = sess.run( # merged_summary will mess up streaming metrics!
                        [predicted_label, conf_scores, cost, cost_summary, streaming_accuracy_update, streaming_recall_update, streaming_precision_update, streaming_map_update],
                        feed_dict=feed_dict)
                else:
                    # print('without MAP')
                    pred, conf, batch_cost, c, _, _, _ = sess.run( # merged_summary will mess up streaming metrics!
                        [predicted_label, conf_scores, cost, cost_summary, streaming_accuracy_update, streaming_recall_update, streaming_precision_update],
                        feed_dict=feed_dict)
                predictions.extend(pred)
                confidence_scores.extend(conf)
                writer.add_summary(c, epoch)
                minibatch_cost += batch_cost / num_minibatches

            if not ignore_MAP:
                eval, acc, rec, prec, f_1, ma_p = sess.run(
                    [eval_summary, streaming_accuracy, streaming_recall, streaming_precision, streaming_f1,streaming_map])
                eval_metrics = [acc, prec, rec, f_1, ma_p]
            else:
                eval, acc, rec, prec, f_1 = sess.run(
                    [eval_summary, streaming_accuracy, streaming_recall, streaming_precision, streaming_f1])
                eval_metrics = [acc, prec, rec, f_1]
            writer.add_summary(eval, epoch)
            return confidence_scores, predictions, minibatch_cost, eval_metrics

        def training_loop(sess, train_dict, dev_dict, test_dict, train_writer, dev_writer, opt,
                          dropout, seed_list, num_epochs, early_stopping, optimizer, lr_bert, lr_new, layer_specific_lr, stopping_criterion='MAP',
                          patience=patience, topic_scope=None, predict_every_epoch=False):
            '''
            Trains the model
            :param X1_train: document 1 (train)
            :param X2_train: document 2 (train)
            :param D_T1_train: topic 1 (train)
            :param D_T2_train: topic 2 (train)
            :param Y_train: labels (train)
            :param X1_dev: document 1 (dev)
            :param X2_dev: document 2 (dev)
            :param D_T1_dev: topic 1 (dev)
            :param D_T2_dev: topic 2 (dev)
            :param Y_dev: labels (dev)
            :param train_writer:
            :param dev_writer:
            :param opt: option dict
            :param dropout:
            :param seed_list:
            :param num_epochs:
            :param early_stopping:
            :param stopping_criterion:
            :param patience:
            :return: [opt, epoch]
            '''
            if predict_every_epoch:
                epoch = opt['num_epochs']
                sess.run(tf.local_variables_initializer())
                _, _, train_cost, train_metrics = predict_eval(sess, train_dict, train_writer, epoch,
                                                               skip_MAP(train_dict['E1']), topic_scope,
                                                               layer_specific_lr)
                dev_scores, dev_pred, dev_cost, dev_metrics = predict_eval(sess, dev_dict, dev_writer, epoch,
                                                                           skip_MAP(dev_dict['E1']), topic_scope,
                                                                           layer_specific_lr)
                print('Predicting for epoch {}'.format(epoch))
                test_scores, test_pred, _, test_metrics = predict_eval(sess, test_dict, test_writer, epoch,
                                                                       skip_MAP(test_dict['E1']), topic_scope,
                                                                       layer_specific_lr)
                output_predictions(ID1_dev, ID2_dev, dev_scores, dev_pred, 'dev_{}'.format(epoch), opt)
                opt = save_eval_metrics(dev_metrics, opt, 'dev', 'score_{}'.format(epoch))  # log dev metrics
                output_predictions(ID1_test, ID2_test, test_scores, test_pred, 'test_{}'.format(epoch), opt)
                opt = save_eval_metrics(test_metrics, opt, 'test', 'score_{}'.format(epoch))  # log test metrics
                write_log_entry(opt, 'data/logs/' + logfile)


            epoch = opt['num_epochs']+1  # continue counting after freeze epochs
            best_dev_value = None
            best_dev_round = 0
            ep = 'num_epochs'

            while True:
                print('Epoch {}'.format(epoch))
                minibatch_cost = 0.
                minibatches = create_minibatches(train_dict, minibatch_size, seed_list.pop(0), sparse=sparse, random=True, topic_scope=topic_scope)

                for minibatch in minibatches:
                    feed_dict = {X1: minibatch['E1'],
                                 X1_mask: minibatch['E1_mask'],
                                 X1_seg: minibatch['E1_seg'],
                                # X2: minibatch['E2'],
                                 Y: minibatch['Y'],
                                 learning_rate_old_layers: lr_bert,
                                 dropout_prob: dropout}
                    if layer_specific_lr:
                        feed_dict[learning_rate_new_layers] = lr_new
                    # print(minibatch.keys())
                    if 'doc' in topic_scope:
                        feed_dict[D_T1] = minibatch['D_T1']
                        feed_dict[D_T2] = minibatch['D_T2']
                    if 'word' in topic_scope:
                        feed_dict[W_T1] = minibatch['W_T1']
                        feed_dict[W_T2] = minibatch['W_T2']
                    # IMPORTANT: The line that runs the graph on a minibatch.
                    # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                    _, temp_cost = sess.run([optimizer, cost], feed_dict=feed_dict)

                # write summaries and checkpoints every few epochs
                if not logfile is None:
                    # print("Train cost after epoch %i: %f" % (epoch, minibatch_cost))
                    sess.run(tf.local_variables_initializer())
                    _, _, train_cost, train_metrics = predict_eval(sess,train_dict, train_writer, epoch, skip_MAP(train_dict['E1']), topic_scope,layer_specific_lr)
                    dev_scores, dev_pred, dev_cost, dev_metrics = predict_eval(sess,dev_dict, dev_writer, epoch, skip_MAP(dev_dict['E1']), topic_scope,layer_specific_lr)
                    if predict_every_epoch and (not epoch==num_epochs):
                        print('Predicting for epoch {}'.format(epoch))
                        test_scores, test_pred, _, test_metrics =  predict_eval(sess,test_dict, test_writer, epoch, skip_MAP(test_dict['E1']), topic_scope,layer_specific_lr)
                        output_predictions(ID1_dev, ID2_dev, dev_scores, dev_pred, 'dev_{}'.format(epoch), opt)
                        opt = save_eval_metrics(dev_metrics, opt, 'dev','score_{}'.format(epoch))  # log dev metrics
                        output_predictions(ID1_test, ID2_test, test_scores, test_pred, 'test_{}'.format(epoch), opt)
                        opt = save_eval_metrics(test_metrics, opt, 'test','score_{}'.format(epoch))  # log test metrics
                        write_log_entry(opt, 'data/logs/' + logfile)
                    # dev_metrics = [acc, prec, rec, f_1, ma_p]
                    # use cost or other metric as early stopping criterion
                    if stopping_criterion == 'cost':
                        stopping_metric = dev_cost
                        print("Dev {} after epoch {}: {}".format(stopping_criterion, epoch, stopping_metric))
                    elif stopping_criterion == 'MAP':
                        assert len(dev_metrics) == 5  # X1_dev must have 10 * x examples
                        current_result = dev_metrics[-1]  # MAP
                        print("Dev {} after epoch {}: {}".format(stopping_criterion, epoch, current_result))
                        stopping_metric = 1 - current_result  # dev error
                    elif stopping_criterion == 'F1':
                        current_result = dev_metrics[3]  # F1
                        print("Dev {} after epoch {}: {}".format(stopping_criterion, epoch, current_result))
                        stopping_metric = 1 - current_result  # dev error
                    elif stopping_criterion == 'Accuracy':
                        current_result = dev_metrics[0]  # Accuracy
                        print("Dev {} after epoch {}: {}".format(stopping_criterion, epoch, current_result))
                        stopping_metric = 1 - current_result  # dev error
                    if early_stopping:
                        # save checkpoint for first or better models
                        if (best_dev_value is None) or (stopping_metric < best_dev_value):
                            best_dev_value = stopping_metric
                            best_dev_round = epoch
                            save_model(opt, saver, sess, epoch)  # save model
                            opt = save_eval_metrics(train_metrics, opt, 'train')  # update train metrics in log
                # check stopping criteria
                # stop training if predefined number of epochs reached
                if (not early_stopping) and (epoch == num_epochs):
                    print('Reached predefined number of training epochs.')
                    save_model(opt, saver, sess, epoch)  # save model
                    break
                if early_stopping and (epoch == num_epochs):
                    print('Maximum number of epochs reached during early stopping.')
                    break
                # stop training if early stopping criterion reached
                if early_stopping and epoch >= best_dev_round + patience:
                    print('Early stopping criterion reached after training for {} epochs.'.format(epoch))
                    break
                # stop training if gradient is vanishing
                if math.isnan(minibatch_cost):
                    print('Cost is Nan at epoch {}!'.format(epoch))
                    break

                epoch += 1

            print('Finished training.')

            # restore weights from saved model in best epoch
            if early_stopping:
                print('Load best model from epoch {}'.format(best_dev_round))
                opt[ep] = best_dev_round
                epoch = best_dev_round  # log final predictions with correct epoch info
                load_model(opt, saver, sess, best_dev_round) # ToDo: fix Too many open files
                # clean up previous checkpoints to save space
                delete_all_checkpoints_but_best(opt, best_dev_round)
            else:
                opt[ep] = epoch
                opt = save_eval_metrics(train_metrics, opt, 'train')  # log train metrics

            return opt, epoch

        # Initialize all the variables globally
        init = tf.global_variables_initializer()
        if (not logfile is None) or (early_stopping):
            saver = create_saver()
        start_time, opt = start_timer(opt, logfile)
        print('Model {}'.format(opt['id']))

        #####
        # Start session to execute Tensorflow graph
        #####

        with tf.Session(config=session_config) as sess: #config=tf.ConfigProto(log_device_placement=True)

            # add debugger (but not for batch experiments)
            if __name__ == '__main__' and FLAGS.debug:
                sess = tf_debug.TensorBoardDebugWrapperSession(sess, "NPMacBook.local:7000")

            # Run the initialization
            sess.run(init)

            if logfile is None:
                train_writer = None
                dev_writer = None
                test_writer = None
                if extra_test:
                    test_writer_extra = None
            else:
                print('logfile: {}'.format(logfile))
                create_model_folder(opt)
                model_dir = get_model_dir(opt)
                train_writer = tf.summary.FileWriter(model_dir + '/train', sess.graph)  # save graph first
                dev_writer = tf.summary.FileWriter(model_dir + '/dev')
                test_writer = tf.summary.FileWriter(model_dir + '/test')
                if extra_test:
                    test_writer_extra = tf.summary.FileWriter(model_dir + '/test_extra')

            # additional input for predict every epoch
            if predict_every_epoch:
                td = test_dict
            else:
                td = None

            # set learning rates per layer
            if speedup_new_layers:
                lr_bert = learning_rate
                lr_new = learning_rate*100
            else:
                lr_bert = learning_rate
                lr_new = learning_rate

            # Freeze BERT and only train new weights
            if freeze_thaw_tune:
                print('Freeze BERT and train new layers...')
                opt, epoch = training_loop(sess, train_dict, dev_dict,td, train_writer, dev_writer,
                                           opt, dropout, seed_list, num_epochs, early_stopping, train_step, 0, lr_new, layer_specific_lr,
                                           stopping_criterion, patience, topic_scope, predict_every_epoch)
                num_epochs += epoch
                lr_new = learning_rate

            # Normal Finetuning
            print('Finetune...')
            opt, epoch = training_loop(sess,train_dict, dev_dict, td, train_writer, dev_writer,
                                   opt, dropout, seed_list, num_epochs, early_stopping, train_step, lr_bert, lr_new, layer_specific_lr,
                                   stopping_criterion, patience,topic_scope, predict_every_epoch)


            # Predict + evaluate on dev and test set

            # train_scores, train_pred, _, train_metrics = predict(X1_train, X2_train, Y_train, train_writer, epoch)
            dev_scores, dev_pred, _, dev_metrics = predict_eval(sess,dev_dict, dev_writer, epoch, skip_MAP(dev_dict['E1']),topic_scope,layer_specific_lr)
            opt = save_eval_metrics(dev_metrics, opt, 'dev')
            if opt['dataset']=='GlueQuora':
                test_scores, test_pred, _, test_metrics = predict(sess, test_dict, test_writer, epoch, skip_MAP(test_dict['E1']), topic_scope,layer_specific_lr)
            else:
                test_scores, test_pred, _, test_metrics = predict_eval(sess,test_dict, test_writer, epoch, skip_MAP(test_dict['E1']),topic_scope,layer_specific_lr)
            opt = save_eval_metrics(test_metrics, opt, 'test')
            opt = end_timer(opt, start_time, logfile)
            if extra_test:
                test_scores_extra, test_pred_extra, _, test_metrics_extra = predict_eval(sess,test_dict_extra, test_writer_extra, epoch, skip_MAP(test_dict['E1']),topic_scope,layer_specific_lr)
                opt = save_eval_metrics(test_metrics_extra, opt, 'PAWS')

            if print_dim:
                if stopping_criterion is None:
                    stopping_criterion = 'Accuracy'
                print('Dev {}: {}'.format(stopping_criterion, opt['score'][stopping_criterion]['dev']))
                print('Test {}: {}'.format(stopping_criterion, opt['score'][stopping_criterion]['test']))

        if not logfile is None:
            # save log
            write_log_entry(opt, 'data/logs/' + logfile)

            # write predictions to file for scorer
            # output_predictions(ID1_train, ID2_train, train_scores, train_pred, 'train', opt)
            output_predictions(ID1_dev, ID2_dev, dev_scores, dev_pred, 'dev', opt)
            output_predictions(ID1_test, ID2_test, test_scores, test_pred, 'test', opt)
            if extra_test:
                output_predictions(ID1_test_extra, ID2_test_extra, test_scores_extra, test_pred_extra, 'PAWS_test', opt)
            print('Wrote predictions for model_{}.'.format(opt['id']))

            # save model
            if save_checkpoints:
                save_model(opt, saver, sess, epoch)  # save disk space

            # close all writers to prevent too many open files error
            train_writer.close()
            dev_writer.close()
            test_writer.close()
            if extra_test:
                test_writer_extra.close()

    except Exception as e:
        print("Error: {0}".format(e.__doc__))
        traceback.print_exc(file=sys.stdout)
        opt['status'] = 'Error'
        write_log_entry(opt, 'data/logs/' + logfile)

    # print('==============')

    return opt


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--debug",
        type="bool",
        nargs="?",
        const=True,
        default=False,
        help="Use debugger to track down bad values during training. "
             "Mutually exclusive with the --tensorboard_debug_address flag.")
    parser.add_argument('-gpu', action="store", dest="gpu", type=int, default=-1)
    FLAGS, unparsed = parser.parse_known_args()

    # adjust settings by changing the option dictionary
    opt = {'dataset': 'MSRP', 'datapath': 'data/',
           'model': 'bert_simple_topic',
           'tasks': ['B'],
           'subsets': [
               'train','dev','test'
               # 'train_large','test2016','test2017' # for Semeval
                       ],
           'minibatch_size': 10,
            'bert_update':True,'bert_large':False,
           'L2': 0, 'load_ids': True,
           'max_m': 10, # only load 10 examples for debugging
           'unk_sub': False, 'padding': False, 'simple_padding': True,
           'hidden_layer': 1,
           'learning_rate': 0.3,
            # 'patience': 2,
           'num_epochs': 1,'bert_cased':False,
           'sparse_labels': True, 'max_length': 'minimum',
           'stopping_criterion': 'F1',
           'optimizer': 'Adadelta', 'dropout': 0.1,
           'gpu':FLAGS.gpu,
          'freeze_thaw_tune':False,
           'speedup_new_layers':False,'seed':1,
          'num_topics': 80, 'topic_type': 'ldamallet','topic':'doc','topic_alpha':1,
            'topic_update':True,
            'unk_topic':'zero',#'unflat_topics':True #
                         'sparse_labels':True,'predict_every_epoch':False
           }

    data_dict = load_data(opt, cache=True, write_vocab=False)
    opt = model(data_dict, opt, logfile='test.json', print_dim=True)

    # T1 = data_dict['T1']
    # T2 = data_dict['T2']
    # E1 = data_dict['E1']
    # E2 = data_dict['E2']
