from src.models.save_load import get_model_dir
import csv
import numpy as np
import tensorflow as tf
import pandas as pd
from src.loaders.load_data import load_data

def get_confidence_scores(Z3,normalised=False):
    if normalised:
        # normalise logits to get probablities??
        Z3 = tf.nn.softmax(Z3)
    conf_score = tf.gather(Z3,1,axis=1,name='conf_score') # is equal to:  Z3[:,1]
    return conf_score

def save_eval_metrics(metrics, opt, data_split='test',dict_key='score'):
    # dev_metrics = [acc, prec, rec, f_1, ma_p]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'MAP']
    if dict_key not in opt:
        opt[dict_key] = {}
    for i,eval_score in enumerate(metrics):
        metric = metric_names[i]
        if metric not in opt[dict_key]:
            opt[dict_key][metric] = {}
        if eval_score is None:
            opt[dict_key][metric][data_split] = eval_score
        else:
            opt[dict_key][metric][data_split] = round(float(eval_score), 4) # prevent problem when writing log file
    return opt

def output_predictions(query_ids,doc_ids,Z3,Y,subset,opt):
    '''
    Writes an output files with system predictions to be evaluated by official Semeval scorer.
    :param query_ids: list of question ids
    :param doc_ids: list of document ids
    :param Z3: numpy array with ranking scores (m,)
    :param Y: numpy array with True / False (m,)
    :param opt: 
    :param subset: string indicating which subset of the data ('train','dev','test)
    :return: 
    '''
    if 'PAWS' in subset:
        subset = subset.replace('PAWS','p')
    outfile = get_model_dir(opt)+'subtask'+''.join(opt['tasks'])+'.'+subset+'.pred'
    with open(outfile, 'w') as f:
        file_writer = csv.writer(f,delimiter='\t')
        # print(Y)
        label = [str(e==1).lower() for e in Y]
        for i in range(len(query_ids)):
            file_writer.writerow([query_ids[i],doc_ids[i],0,Z3[i],label[i]])

def read_predictions(opt,subset='dev',VM_path=True):
    '''
    Reads prediction file from model directory, extracts pair id, prediction score and predicted label.
    :param opt: option log
    :param subset: ['train','dev','test']
    :param VM_path: was prediction file transferred from VM?
    :return: pandas dataframe
    '''
    if type(opt['id'])==str:
        if opt['dataset']=='Semeval':
            outfile = get_model_dir(opt,VM_copy=VM_path)+'subtask_'+''.join(opt['tasks'])+'_'+subset+'.txt'
        else:
            outfile = get_model_dir(opt, VM_copy=VM_path) + 'subtask' + ''.join(opt['tasks']) + '.' + subset + '.pred'
    else:
        outfile = get_model_dir(opt,VM_copy=VM_path)+'subtask'+''.join(opt['tasks'])+'.'+subset+'.pred'
    print(outfile)
    predictions = []
    with open(outfile, 'r') as f:
        file_reader = csv.reader(f,delimiter='\t')
        for id1,id2,_,score,pred_label in file_reader:
            pairid = id1+'-'+id2
            if pred_label == 'true':
                pred_label=1
            elif pred_label == 'false':
                pred_label=0
            else:
                raise ValueError("Output labels should be 'true' or 'false', but are {}.".format(pred_label))
            predictions.append([pairid,score,pred_label])
    cols = ['pair_id','score','pred_label']
    prediction_df = pd.DataFrame.from_records(predictions,columns=cols)
    return prediction_df

def read_original_data(opt, subset='dev'):
    '''
    Reads original labelled dev file from data directory, extracts get pair_id, gold_label and sentences.
    :param opt: option log
    :param subset: ['train','dev','test']
    :return: pandas dataframe
    '''
    # adjust filenames in case of increased training data
    if 'train_large' in opt['subsets']:
        print('adjusting names')
        if subset=='dev':
            subset='test2016'
        elif subset=='test':
            subset='test2017'
    # adjust loading options:
    opt['subsets'] = [subset] # only specific subset
    opt['load_ids'] = True # with labels
#     print(opt)
    data_dict = load_data(opt,numerical=False)
    ID1 = data_dict['ID1'][0] # unlist, as we are only dealing with one subset
    ID2 = data_dict['ID2'][0]
    R1 = data_dict['R1'][0]
    R2 = data_dict['R2'][0]
    L = data_dict['L'][0]
    # extract get pair_id, gold_label, sentences
    labeled_data = []
    for i in range(len(L)):
        pair_id = ID1[i]+'-'+ID2[i]
        gold_label = L[i]
        s1 = R1[i]
        s2 = R2[i]
        labeled_data.append([pair_id,gold_label,s1,s2])
    # turn into pandas dataframe
    cols = ['pair_id','gold_label','s1','s2']
    label_df = pd.DataFrame.from_records(labeled_data,columns=cols)
    return label_df