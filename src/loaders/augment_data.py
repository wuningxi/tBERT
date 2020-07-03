import os

def create_large_train():
    '''
    Create large training set for each task based on Deriu 2017
    '''
    from src.loaders.load_data import get_filepath, load_file
    print('Creating large training set...')
    for t in ['A','B','C']:
        opt = {'dataset': 'Semeval', 'datapath': 'data/',
               'tasks': [t],
               'subsets': ['train','train2','dev']}
        files = get_filepath(opt)
        outfile = os.path.join(opt['datapath'],opt['dataset'],'train_large_'+t+'.txt')
        large_train = []
        for f in files:
            with open(f,encoding='utf-8') as infile:
                for l in infile:
                    large_train.append(l)
        with open(outfile,'w',encoding='utf-8') as out:
            for l in large_train:
                out.writelines(l)
    print('Done.')

def double_task_training_data():
    '''
    Double existing data by switching side of questions to mitigate data scarcity for task B
    '''
    from src.loaders.load_data import get_filepath, load_file
    print('Creating augmented training files for tasks')
    subsets = ['train', 'train_large']
    for t in ['A','B','C']:
        for s in subsets:
            opt = {'dataset': 'Semeval', 'datapath': 'data/',
                   'tasks': [t],
                   'subsets': [s]}
            f = get_filepath(opt)[0]
            id1,id2,s1,s2,l = load_file(f,True)
            id1_double = id1 + id2
            id2_double = id2 + id1
            s1_double = s1 + s2
            s2_double = s2 + s1
            l_double = list(l) + list(l)
            assert len(id1_double)==len(id2_double)==len(s1_double)==len(s2_double)==len(l_double)
            outfile = os.path.join(os.path.join(opt['datapath'],opt['dataset'])+'/'+s + '_double' + '_'+t+'.txt')
            print(outfile)
            with open(outfile,'w',encoding='utf-8') as out:
                for i in range(len(id1_double)):
                    out.writelines(id1_double[i]+'\t'+id2_double[i]+'\t'+s1_double[i]+'\t'+s2_double[i]+'\t'+str(l_double[i])+'\n')


def augment_task_b_with_():
    '''
    Double existing data by switching side of questions to mitigate data scarcity for task B
    '''
    from src.loaders.load_data import get_filepath, load_file
    print('Creating augmented training files for Task B')
    subsets = ['train', 'train_large']
    for s in subsets:
        opt = {'dataset': 'Semeval', 'datapath': 'data/',
               'tasks': ['B'],
               'subsets': [s]}
        f = get_filepath(opt)[0]
        id1,id2,s1,s2,l = load_file(f,True)
        id1_double = id1 + id2
        id2_double = id2 + id1
        s1_double = s1 + s2
        s2_double = s2 + s1
        l_double = list(l) + list(l)
        assert len(id1_double)==len(id2_double)==len(s1_double)==len(s2_double)==len(l_double)
        outfile = os.path.join(f.split('_B.txt')[0] + '_double' + '_B.txt')
        with open(outfile,'w',encoding='utf-8') as out:
            for i in range(len(id1_double)):
                out.writelines(id1_double[i]+'\t'+id2_double[i]+'\t'+s1_double[i]+'\t'+s2_double[i]+'\t'+str(l_double[i])+'\n')

