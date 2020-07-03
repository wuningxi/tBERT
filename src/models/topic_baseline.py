
import math
import traceback
import numpy as np
import sys
import argparse

np.random.seed(1)

from sklearn.metrics import accuracy_score,f1_score
from src.loaders.load_data import load_data
from src.logs.training_logs import write_log_entry, start_timer, end_timer
from src.models.save_load import create_model_folder
from src.evaluation.evaluate import output_predictions
from src.evaluation.evaluate import save_eval_metrics
from src.models.helpers.base import add_git_version,extract_data
import warnings
from src.evaluation.metrics.js_div import js
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
# warnings.filterwarnings('error')


"""
Implements Base Model with early stopping and auxiliary loss
"""

predefined_opts = [
    # loader
    'dataset', 'datapath', 'tasks', 'subsets', 'w2v_limit', 'max_m', 'max_length', 'n_gram_embd', 'padding', 'unk_sub',
    'simple_padding','lemmatize',
    # topic models
    'num_topics', 'topic_type', 'topic','topic_alpha','unflat_topics','unk_topic','stem',
    # model
    'model', 'load_ids', 'embedding_dim', 'embd_update','pretrained_embeddings','model','threshold',
    # auxiliary loss
    'git']

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
    Implements affinity CNN in Tensorflow:

    Arguments:
    data -- train, dev, test data



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
        opt['model'] = opt.get('model','topic_baseline')
        assert opt['model'] == 'topic_baseline'
        # topic models
        topic_scope = opt['topic'] = opt.get('topic', '')
        assert topic_scope in ['','word','doc','word+doc']
        if opt['model'] in ['topic_bi_cnn','topic_affinity_cnn','topic_separate_affinity_cnn']:
            assert topic_scope in ['word+doc','doc','word']
        num_topics = opt['num_topics'] = opt.get('num_topics', None)
        topic_type = opt['topic_type'] = opt.get('topic_type', 'ldamallet')
        threshold = opt['threshold'] = opt.get('threshold', 0.5)
        print('threshold: {}'.format(threshold))
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

        #####
        # unpack data and assign to model variables
        #####

        embd = data_dict.get('embd', None) # (565852, 200)
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

        start_time, opt = start_timer(opt, logfile)
        # if not logfile is None:
        #     print('logfile: {}'.format(logfile))
        #     create_model_folder(opt)
        #     model_dir = get_model_dir(opt)

        def get_mean_word_topics(w_topic_ids, topic_matrix):
            T = []
            for i in range(len(w_topic_ids)):  # loop through sentences
                s_dist = []
                # mean of word topics (todo: ignore non-topic words or not?)
                for w in w_topic_ids[i]:  # loop through words
                    if not w==0:
                        w_dist = topic_matrix[w]
                        s_dist.append(w_dist)
                if len(s_dist)==0:
                    # no word topic vector
                    s_dist=topic_matrix[0]
                elif len(s_dist)==1:
                    # only one word topic vector
                    s_dist = s_dist[0]
                    pass
                else:
                    # multiple word topic vectors
                    s_dist = np.array(s_dist).mean(axis=0)
                T.append(s_dist)
            T = np.array(T)
            assert(len(T.shape)==2)
            return T

        # Predict + evaluate on dev and test set
        def predict(split_dict,topic_scope,threshold):
            # extract topics
            if topic_scope == 'word':
                # get topic ids from dict
                T1_ids = split_dict['W_T1']
                T2_ids = split_dict['W_T2']
                # print(T1_ids.shape)
                # print(T2_ids.shape)
                topic_matrix = data_dict['word_topics']['topic_matrix']
                # lookup actual distributions, reduce dim through mean across word topics
                T1 = get_mean_word_topics(T1_ids, topic_matrix) # shape (m,num_topic)
                T2 = get_mean_word_topics(T2_ids, topic_matrix)
            elif topic_scope == 'doc':
                T1 = split_dict['D_T1'] # shape (m,num_topic)
                T2 = split_dict['D_T2']
            elif 'word+doc':
                # concat
                NotImplementedError()
            # extract labels
            L = split_dict['Y']
            predictions = []
            scores = []

            # calculate JSD between topic distributions
            print(T1.shape)
            print(T2.shape)
            for tl,tr in zip(T1,T2):
                # divergence = distance.jensenshannon(tl,tr) # why does it give nans? zero div
                divergence = js(tl,tr)
                if math.isnan(divergence):
                    Warning('JS divergence is NAN')
                    print(tl)
                    print(tr)
                    print(tl==tr)
                if divergence>threshold:
                    prediction = 0
                else:
                    prediction = 1
                scores.append(divergence)
                predictions.append(prediction)
            predictions = np.array(predictions)

            acc = accuracy_score(L,predictions)
            f_1 = f1_score(L,predictions)
            # todo
            prec = 0
            rec = 0
            ma_p = 0
            metrics = [acc, prec, rec, f_1, ma_p]
            return scores, predictions, metrics

        train_scores, train_pred, train_metrics = predict(train_dict,topic_scope,threshold)
        # print(train_scores)
        # print(train_pred)
        def get_mean_jsd(scores):
            mean_jsd = sum(scores) / len(scores)
            return mean_jsd

        dev_scores, dev_pred, dev_metrics = predict(dev_dict,topic_scope,threshold)
        opt = save_eval_metrics(dev_metrics, opt, 'dev')
        test_scores, test_pred, test_metrics = predict(test_dict,topic_scope,threshold)

        # save mean jsd for splits
        opt['mean_jsd_train'] = get_mean_jsd(train_scores)
        opt['mean_jsd_dev'] = get_mean_jsd(dev_scores)
        opt['mean_jsd_test'] = get_mean_jsd(test_scores)

        print('mean div scores on train set: {}'.format(opt['mean_jsd_train']))
        print('mean div scores on dev set: {}'.format(opt['mean_jsd_dev']))

        opt = save_eval_metrics(test_metrics, opt, 'test')
        opt = end_timer(opt, start_time, logfile)

        if extra_test:
            test_scores_extra, test_pred_extra, test_metrics_extra = predict(test_dict_extra,topic_scope,threshold)
            opt = save_eval_metrics(test_metrics_extra, opt, 'PAWS')

        if print_dim:
            stopping_criterion = 'Accuracy'
            print('Dev {}: {}'.format(stopping_criterion, opt['score'][stopping_criterion]['dev']))
            print('Test {}: {}'.format(stopping_criterion, opt['score'][stopping_criterion]['test']))

        if not logfile is None:

            # # prevent problem with long floats when writing log file
            # for k,v in opt.items():
            #     if type(v)==float:
            #         opt[k] = round(float(v), 6)
            # save log
            print('logfile: {}'.format(logfile))
            create_model_folder(opt)
            write_log_entry(opt, 'data/logs/' + logfile)
            # writer.add_graph(sess.graph)

            # write predictions to file for scorer
            output_predictions(ID1_train, ID2_train, train_scores, train_pred, 'train', opt)
            output_predictions(ID1_dev, ID2_dev, dev_scores, dev_pred, 'dev', opt)
            output_predictions(ID1_test, ID2_test, test_scores, test_pred, 'test', opt)
            if extra_test:
                output_predictions(ID1_test_extra, ID2_test_extra, test_scores_extra, test_pred_extra, 'PAWS_test', opt)
            print('Wrote predictions for model_{}.'.format(opt['id']))

            # todo: compute non-obvious F1 and save in opt

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

    unks = ['zero','zero','uniform','uniform']
    stems = [True,False,True,False]

    for unk_topic,stem in zip(unks,stems):
        opt = {'embd_update': False,
               'load_ids': True, 'max_length': 'minimum', 'max_m': None, #todo: fix max_m=None error
               'model': 'topic_baseline', 'n_gram_embd': False,
               'pretrained_embeddings': None,
               'subsets': ['train', 'dev', 'test'],
               # 'subsets': ['train_large', 'test2016', 'test2017'],
                'topic': 'word', 'topic_type': 'ldamallet',

               'padding': False, 'simple_padding': True,
               'max_length': 'minimum', 'unk_sub': False,
               'lemmatize': False,

               'datapath': 'data/', 'dataset': 'Quora', 'tasks': ['B'],
               'topic_alpha': 1, 'num_topics': 90, 'threshold': 0.090,
               'unk_topic':unk_topic, 'stem':stem}

        # print(opt)
        data_dict = load_data(opt, cache=True, write_vocab=False)
        opt = model(data_dict, opt, logfile='specific_topic_settings.json', print_dim=True)

    # todo: zeros - no stem
    # todo: zeros - stem
    # todo: uniform - stem

    # data_dict['embd']

