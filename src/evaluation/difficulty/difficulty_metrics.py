def count_tp_tn_fp_fn(split_df,case=None):
    if case is None:
        tp = len(split_df.loc[(split_df['gold_label']==1) & (split_df['pred_label']==1)])
        tn = len(split_df.loc[(split_df['gold_label']==0) & (split_df['pred_label']==0)])
        fp = len(split_df.loc[(split_df['gold_label']==0) & (split_df['pred_label']==1)])
        fn = len(split_df.loc[(split_df['gold_label']==1) & (split_df['pred_label']==0)])
    else:
        tp = len(split_df.loc[(split_df['difficulty'].str.contains(case)) & (split_df['gold_label']==1) & (split_df['pred_label']==1)])
        tn = len(split_df.loc[(split_df['difficulty'].str.contains(case)) & (split_df['gold_label']==0) & (split_df['pred_label']==0)])
        fp = len(split_df.loc[(split_df['difficulty'].str.contains(case)) & (split_df['gold_label']==0) & (split_df['pred_label']==1)])
        fn = len(split_df.loc[(split_df['difficulty'].str.contains(case)) & (split_df['gold_label']==1) & (split_df['pred_label']==0)])
    return [tp,tn,fp,fn]

def safe_div(x,y):
    if y == 0:
        return 0
    return x / y
def calculate_accuracy(tp,tn,fp,fn):
    return round(safe_div((tp+tn), (tp+tn+fp+fn)),3)
def calculate_precision(tp,tn,fp,fn):
    return round(safe_div(tp, (tp+fp)),3)
def calculate_recall(tp,tn,fp,fn):
    return round(safe_div(tp, (tp+fn)),3)
def calculate_true_neg_rate(tp,tn,fp,fn):
    return round(safe_div(tn , (tn+fp)),3)
def calculate_f1(precision,recall):
    return round(safe_div(2 * precision * recall, (precision + recall)), 3)


if __name__ == '__main__':
    from src.evaluation.metrics.lexical_similarity import LexicalSimilarity
    from src.evaluation.difficulty.difficulty_cases import annotate_difficulty_case,load_subset_pred_overlap


    VM_path = False
    opt = {'dataset': 'Semeval', 'datapath': 'data/',
           'tasks': ['B'],'n_gram_embd':False,
            'subsets': ['train_large','dev', 'test'],
           'load_ids': False, 'cache':True,'id':1155,
           'simple_padding': True, 'padding': True}
    metric = 'js-div'
    LexSim = LexicalSimilarity()
    overlapping = LexSim.get_metric(metric,opt['dataset'],opt['tasks'][0],'test')
    test_df = load_subset_pred_overlap(opt, 'test', overlapping, VM_path)
    test_df
    split_df = annotate_difficulty_case(test_df,metric,split_by='median')
    count_tp_tn_fp_fn(split_df)
    [tp,tn,fp,fn] = count_tp_tn_fp_fn(split_df)
    print(calculate_accuracy(tp,tn,fp,fn))
    print(calculate_precision(tp,tn,fp,fn))
    print(calculate_recall(tp,tn,fp,fn))
    print(calculate_f1(calculate_precision(tp,tn,fp,fn),calculate_recall(tp,tn,fp,fn)))