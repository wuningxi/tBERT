import pandas as pd

from src.evaluation.evaluate import read_predictions, read_original_data
from src.evaluation.evaluate import read_original_data
from src.logs.training_logs import read_all_training_logs,find_entry
from src.evaluation.difficulty.difficulty_metrics import *

def annotate_difficulty_case(test_df, metric, split_by='median'):
    if split_by == 'median':
        criterion = test_df['overlapping'].median()
    else:
        criterion = split_by
    test_df['difficulty'] =''
    assert metric in ['js-div','jaccard','dice']
    if metric=='js-div':
        test_df.loc[(test_df['overlapping']<=criterion) & (test_df['gold_label']==0), 'difficulty'] = 'Nn'
        test_df.loc[(test_df['overlapping']>criterion) & (test_df['gold_label']==0), 'difficulty'] = 'No'
        test_df.loc[(test_df['overlapping']>criterion) & (test_df['gold_label']==1), 'difficulty'] = 'Pn'
        test_df.loc[(test_df['overlapping']<=criterion) & (test_df['gold_label']==1), 'difficulty'] = 'Po'
    else:
        test_df.loc[(test_df['overlapping']<=criterion) & (test_df['gold_label']==0), 'difficulty'] = 'No'
        test_df.loc[(test_df['overlapping']>criterion) & (test_df['gold_label']==0), 'difficulty'] = 'Nn'
        test_df.loc[(test_df['overlapping']>criterion) & (test_df['gold_label']==1), 'difficulty'] = 'Po'
        test_df.loc[(test_df['overlapping']<=criterion) & (test_df['gold_label']==1), 'difficulty'] = 'Pn'
    return test_df

def gather_statistics(LexSim,metric, todo =['Semeval_A','Semeval_B','Semeval_C','MSRP_B','Quora_B'],split_by='median',subsets=['dev']):
    cases = []
    median = []
    metrics = None
    for t in todo:
        aggregated_subsets = None
        train_median = None
        for s in subsets:
            dataset,task = t.split('_')
            # if dataset=='Semeval':
            #     if s == 'train':
            #         split = 'train_large'
            #     elif s == 'dev':
            #         split = 'test2016'
            #     elif s == 'test':
            #         split = 'test2017'
            # else:
            #     split = s
            # opt = {'dataset': dataset, 'datapath': 'data/',
            #        'tasks': [task],'n_gram_embd':False,
            #         'subsets': [split],
            #        'load_ids': False, 'cache':True}
            # result = read_original_data(opt, subset=split)
            print(s)
            m = LexSim.get_metric(metric,dataset=dataset,task=task,subset=s)
            gold_label = LexSim.get_labels(dataset=dataset, task=task, subset=s)
            pair_id = LexSim.get_ids(dataset=dataset,task=task,subset=s)
            s1 = LexSim.get_s1(dataset=dataset, task=task, subset=s)
            s2 = LexSim.get_s2(dataset=dataset, task=task, subset=s)
            difficulty = LexSim.get_difficulty(dataset=dataset,task=task,metric=metric,subset=s)
            # turn into pandas dataframe
            result = pd.DataFrame({'pair_id': pair_id,'gold_label': gold_label, 's1': s1, 's2': s2,  'overlapping': m, 'difficulty':difficulty})

            print(len(m))
            # result = annotate_difficulty_case(result,metric,split_by)
            print(result['overlapping'].median())
            if aggregated_subsets is None:
                aggregated_subsets = result.groupby('difficulty')['difficulty'].count()
            else:
                aggregated_subsets += result.groupby('difficulty')['difficulty'].count()
            if metrics is None:
                metrics = result['overlapping']
            else:
                metrics = pd.concat([metrics,result['overlapping']])
    #             todo: add mean/median
        median.append(metrics.median())
        cases.append(aggregated_subsets)
    stats = pd.concat(cases,axis=1)
    stats.columns=todo
    obvious = round((stats.loc['No']+stats.loc['Po'])/stats.sum()*100)
    stats.loc['o%'] = obvious
    stats.loc['median'] = median
    return stats

def get_model_info(model_id,VM_path=True):
    if VM_path:
        logs = read_all_training_logs(log_dir='data/VM_logs/')
    else:
        logs = read_all_training_logs(log_dir='data/logs/')
    opt = find_entry(model_id,logs)
    if 'bert_inject' in opt['model']:
        model = 'bert_inject'
    elif 'bert' in opt['model']:
        if 'topic' in opt['model'] and not 'inject' in opt['model']:
            model = 'tbert'
        else:
            model = 'bert'
    elif 'bi_cnn' in opt['model']:
        model = 'Siamese'
    elif opt['model']=='topic_baseline':
        model='baseline'
    else:
        model = 'TASN'
    if opt.get('elmo_embd',False):
        elmo = '_elmo'
    else:
        elmo = ''
    if 'bert' in opt['model']:
        topic = opt.get('topic','')
        topic_type = opt.get('topic_type','ldamallet')
        if topic in ['word','doc','word+doc','word+avg']:
            topic = '_{}_{}_topic'.format(topic_type,topic)
            # topic = '_{}_topic'.format(topic)
        if 'injection_location' in opt.keys() and not opt['injection_location'] is None:
            topic += '_{}'.format(opt['injection_location'])
    elif 'baseline' in opt['model']:
        topic_type = opt.get('topic_type','ldamallet')
        model = '{}_topic_baseline'.format(topic_type)
        topic = ''
    else:
        if opt.get('topic', '') in ['word', 'doc', 'word+doc']:
            topic = '_topic'
        else:
            topic = ''
    if opt.get('freeze_thaw_tune',False):
        freeze = '_freeze'
        if not opt.get('stopping_criterion',None) is None:
            freeze += '_early'
    else:
        freeze = ''
    if opt.get('seed',None) in [None,'fixed']:
        seed = ''
    else:
        seed = '_seed_{}'.format(opt['seed'])
    return '{}{}{}{}{}'.format(model,topic,elmo,freeze,seed)

def get_model_MAP(model_id,VM_path=True):
    if VM_path:
        logs = read_all_training_logs(log_dir='data/VM_logs/')
    else:
        logs = read_all_training_logs(log_dir='data/logs/')
    opt = find_entry(model_id,logs)
    return opt['score']['MAP']['test']
get_model_MAP(3442,True)


def load_subset_pred_overlap(LexSim,metric,split_by, opt, subset,VM_path=True):
    # read dev set predictions
    predictions = read_predictions(opt,VM_path=VM_path,subset=subset)
    # load dev set gold labels
    if subset in ['primary','contrastive1','contrastive2']:
        s = 'test2017'
    elif subset=='dev' and opt['dataset'] == 'Semeval':
        s = 'test2016'
    elif subset=='test' and opt['dataset'] == 'Semeval':
        s = 'test2017'
    else:
        s = subset

    dataset = opt['dataset']
    task = opt['tasks'][0]
    print(subset)
    m = LexSim.get_metric(metric, dataset=dataset, task=task, subset=s)
    gold_label = LexSim.get_labels(dataset=dataset, task=task, subset=s)
    pair_id = LexSim.get_ids(dataset=dataset, task=task, subset=s)
    s1 = LexSim.get_s1(dataset=dataset, task=task, subset=s)
    s2 = LexSim.get_s2(dataset=dataset, task=task, subset=s)
    difficulty = LexSim.get_difficulty(dataset=dataset,task=task,metric=metric,subset=s,split_by=split_by)
    # turn into pandas dataframe
    result = pd.DataFrame({'pair_id': pair_id, 'gold_label': gold_label, 's1': s1, 's2': s2, 'overlapping': m, 'difficulty': difficulty})
    # # turn into pandas dataframe
    # labelled = pd.DataFrame({'pair_id':pair_id,'s1':s1, 's2':s2, 'gold_label':gold_label,'overlapping':m})

    # join dataframes
    result = pd.merge(predictions, result, on='pair_id', how='outer')
    result['correct_pred'] = result['pred_label']==result['gold_label']
    return result

def report_results(split_df,opt,report_obs_number=False,max_char=None,rename_model_ids=False,VM_path=True):
#     difficulty_groups = sorted(list(split_df.difficulty.unique())) # todo: change order of difficulty groups
    difficulty_groups = ['Po','Pn','No','Nn','F1_o','F1_n']
    accuracies = []
    f1s = []
    cases = []
    # accuracy by subgroup
    for case in difficulty_groups:
        if 'F1_' in case:
            case = case[-1]
        [tp,tn,fp,fn] = count_tp_tn_fp_fn(split_df,case=case)
        if 'N' in case:
            accuracies.append(calculate_true_neg_rate(tp,tn,fp,fn))
        elif 'P' in case:
            accuracies.append(calculate_recall(tp,tn,fp,fn))
        elif len(case)==1:
            precision = calculate_precision(tp,tn,fp,fn)
            recall = calculate_recall(tp,tn,fp,fn)
            accuracies.append(calculate_f1(precision,recall))
        model_name = opt['id']
        # rename numerical model ids
        if rename_model_ids:
            if not type(model_name) is str:
                model_name = '{}-{}'.format(get_model_info(opt['id'],VM_path=VM_path),opt['id'])
        # shorten model names
        if not max_char is None:
            if len(str(model_name))>max_char:
                model_name = model_name[:max_char]
        cases.append(tp+tn+fp+fn)
    # total accuracy
    [tp,tn,fp,fn] = count_tp_tn_fp_fn(split_df)
    accuracies.append(calculate_accuracy(tp,tn,fp,fn))
    precision = calculate_precision(tp,tn,fp,fn)
    recall = calculate_recall(tp,tn,fp,fn)
    accuracies.append(calculate_f1(precision,recall))
    difficulty_groups.append('total_accu')
    difficulty_groups.append('total_f1')
    # as dataframe
    df= pd.DataFrame(accuracies,difficulty_groups,columns = [model_name]) # ToDo:fix column name
    if report_obs_number:
        task = opt['tasks'][0]
        cases.append(tp+tn+fp+fn)
        cases.append(tp+tn+fp+fn)
        df['obs_'+task] = cases
    return df
# report_results(split_df,opt,True,rename_model_ids=True,VM_path=VM_path)

def compare_difficulty_cases(opt,LexSim,distance_metric = 'js-div',append_map=True,split_by='median',report_best=True,report_my=True,report_compare=True,report_obs_number=False):
    VM_path = False
    if opt['dataset']=='Semeval':
        if opt['tasks'] == ['A']:
            best = ['KeLP','Beihang-MSRA','IIT-UHH','ECNU','bunji','EICA','SwissAlps','FuRongWang','FA3L','SnowMan']
#             best = ['KeLP','IIT-UHH','ECNU','bunji','EICA','SwissAlps','FuRongWang','FA3L','SnowMan']
            my = [3438,3433,3432,3429]#[1150, 1154]#[3402]
        elif opt['tasks'] == ['B']:
            # best = ['SimBow','LearningToQuestion','KeLP','Talla','Beihang-MSRA','NLM_NIH','UINSUSKA-TiTech','IIT-UHH','SCIR-QA','FA3L','ECNU','EICA']
            best = ['SimBow','LearningToQuestion','KeLP','Talla','Beihang-MSRA','NLM_NIH','UINSUSKA-TiTech','IIT-UHH','SCIR-QA','FA3L']
            my = [3442,3440,3439,3430]#[1156,1155]#[3391]
        elif opt['tasks'] == ['C']:
            # best = ['IIT-UHH','bunji','KeLP','EICA','FuRongWang','ECNU']
            best = ['IIT-UHH','bunji','KeLP','EICA','FuRongWang']
            my = [3447,3445,3444,3431]#[1157]#[3404]
    elif opt['dataset']=='MSRP':
        best = []
        my = []
    elif opt['dataset']=='Quora':
        best = []
        # to copy:
        my = [3509,3472,3471,3454,2221]
    tables = []
    if report_compare:
#         compare = ['random','truly_random','ratio_random']
        compare = ['truly_random']
    else:
        compare = []
    if not report_my:
        my = []
    if not report_best:
        best = []
    models = best+my+compare
    overlapping=LexSim.get_metric(distance_metric,opt['dataset'],opt['tasks'][0],'test')
    maps = []
    for m in models:
        opt['id']=m
        if m in my:
            VM_path = True
            part = 'test'
        else:
            part = 'primary'
        if append_map:
            maps.append(get_model_MAP(m,VM_path))
        test_df = load_subset_pred_overlap(LexSim, opt, part, overlapping, VM_path)
        split_df = annotate_difficulty_case(test_df,distance_metric,split_by)
#         print(split_df)
        report_obs = report_obs_number and (m == models[-1])
        result_df = report_results(split_df,opt,report_obs_number=report_obs,max_char=None,rename_model_ids=True,VM_path=VM_path)
#         print(result_df)
        tables.append(result_df)
    print('{} {}'.format(opt['dataset'],opt['tasks'][0]))
    final_table = pd.concat(tables,1)
    if append_map:
        final_table = final_table.append(pd.Series(maps,name='total_map', index=final_table.columns),ignore_index=False)
    return final_table