import json
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import ParameterGrid
from datetime import datetime
import time
import git

# read and write log #
######################

def get_git_sha():
    '''
    Get current git hash for code version 
    :return: 
    '''
    # pip install gitpython
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return repo.git.rev_parse(sha, short=7)

def read_training_log(log_path='data/logs/Baseline.json'):
    try:
        with open(log_path, 'r') as infile:
            log = json.load(infile)
    except FileExistsError:
        log = []
    return log

def read_all_training_logs(log_dir='data/logs/', print_log_paths=False):
    '''
    Reads all logs in log directory and returns list of entries
    '''
    log_content = []
    logfiles = [f for f in listdir(log_dir) if isfile(join(log_dir, f)) and  f.endswith('.json')]
    for log_path in logfiles:
        if print_log_paths:
            print(log_path)
        try:
            with open(log_dir+'/'+log_path, 'r') as infile:
                log = json.load(infile)
                log_content += log
        except json.decoder.JSONDecodeError:
            print('Error while loading {}'.format(log_path))
        except FileExistsError:
            log_content = []
    return log_content
def read_training_logs(logfiles,log_dir='data/logs/'):
    '''
    Reads predefined list of logs and returns list of entries 
    '''
    log_content = []
    for log_path in logfiles:
        print(log_path)
        try:
            with open(log_dir+'/'+log_path, 'r') as infile:
                log = json.load(infile)
                log_content += log
        except FileExistsError:
            log = []
    return log_content
def write_training_logs(log,log_dir='data/logs/'):
    subs = {}
    for r in log:
        if r['model'] not in subs:
            subs[r['model']]=[r]
        else:
            subs[r['model']].append(r)
    for key in subs.keys():
        log_path = '%s/%s.json' % tuple([log_dir,key])
        try:
            with open(log_path, 'w') as outfile:
                json.dump(subs[key], outfile, indent=4, sort_keys=True)
        except Exception as e:
            print(e)
            print(log)
            raise ValueError('Results not written in json result files.')
def write_training_log(log,log_path):
    try:
        with open(log_path, 'w') as outfile:
            json.dump(log, outfile, indent=4, sort_keys=True)
    except Exception as e:
        print(e)
        print(log)
        raise ValueError('Results not written in json result file.')
def write_log_entry(r,log_path):
    """
    Reads json log file, updates entry or creates new one if no id contained in r
    :param r: dictionary with result values
    :param log_path: path of result file
    :return:
    """
    try:
        print('reading logs...')
        log = read_training_log(log_path)
    except FileNotFoundError:
        print('No file found at {}. Creating new log.'.format(log_path))
        log = []
    # ids = [r['id'] for r in log if 'id' in r]
    # assign new unique id if non existent
    if 'id' not in r:
        r['id'] = get_new_id(1,'data/logs/')[0]
        log.append(r)

    # otherwise use existing id to overwrite old values
    else:
        pos = 0
        for i in range(len(log)):
            if log[i]['id'] == r['id']:
                # print(log['id'][i])
                pos = i
                break
        log[pos] = r  # overwrite old value\
    try:
        with open(log_path, 'w') as outfile:
            json.dump(log, outfile, indent=4, sort_keys=True)
    except Exception as e:
        print(e)
        print(log)
        raise ValueError('Results not written in json result file.')

def find_entry(model_id,log):
    for r in log:
        if 'id' in r.keys():
            if r['id'] == model_id:
                return r
    ValueError('Entry not found for {}.'.format(model_id))

def find_best_experiment_opt(log_path='data/VM_logs/topic_affinity_cnn_Quora_B.json',split='dev',metric='F1'):
    log = read_training_log(log_path)
    best_opt = None
    best_score = 0
    for opt in log:
        if opt['status']=='finished':
            try:
                score = opt['score'][metric][split]
            except KeyError:
                score = 0
            if score>best_score:
                best_score = score
                best_opt = opt
    return best_opt


# def transform_log(log):
#     """
#     Transform the old log style to new one
#     :param log:
#     :return:
#     """
#     n = 0
#     for l in log['results']:
#         r['id'] = n
#         if 'features' not in r['param']:
#             r['param']['features'] = ['head']
#         l['status'] = 'finished'
#         l['training_time'] = None
#         l['param']['algorithm'] = l['algorithm']
#         del l['algorithm']
#         l['score']['accuracy'] = dict(l['score'])
#         del l['score']['dev']
#         del l['score']['test']
#         n += 1
#     return log

# Latex table output #
######################
def report_results(log):
    for r in log:
        if r['status'] == 'finished':
            if r['size'] == None:
                print('%s & %s & %s \\\\' % tuple([r['model']+r['param']['algorithm'],round(r['score']['accuracy']['dev'],4),round(r['score']['accuracy']['test'],4)]))

def insert_model_param():
    log_path = 'scripts/mlearning/results.json'
    log = read_training_log(log_path)
    new_log = []
    for r in log:
        if 'model_param' not in r:
            if r['model'] == 'LogRegression':
                r['model_param']={'C':1.0, 'solver':'liblinear'}
            elif r['model'] == 'SVM':
                r['model_param'] ={'C': 1,'kernel':'rbf'}
            elif r['model'] == 'MLPerceptron':
                r['model_param'] = {'steps' : 200, 'batch_size' : 1000,'hidden_units': [10, 20, 10]}
            elif r['model'] == 'Baseline':
                r['model_param'] = {}
        if 'task' not in r:
            if 'sentence' in r['param']['features'] :
                r['task'] = 2
            else:
                r['task'] = 1
        if 'folder' in r['param']:
            del r['param']['folder']
        if r['size'] in (200000,None):
            new_log.append(r)
    write_training_log(new_log,'scripts/mlearning/results.json')

def reset_log():
    log = read_training_log()
    n= 0
    for r in log:
        r['id']=n
        if 'score' in r:
            del r['score']
        if 'date' in r:
            del r['date']
        if 'time' in r:
            del r['time']
        if r['model'] == 'Baseline':
            r['status']='priority'
        else:
            r['status'] = 'waiting'
        if r['size'] == 856753:
            r['size'] =None
        if 'sentence' in r['param']['features']:
            r['task'] = 'task2'
        else:
            r['task'] = 'task1'
        n+=1
    write_training_log(log,'scripts/mlearning/results.json')

def get_new_id(num,log_dir):
    print('get new id')
    saved_log = read_all_training_logs(log_dir)
    ids = [r['id'] for r in saved_log if 'id' in r]
    res = []
    pos = 0
    while len(res)<num:
        if pos not in ids:
            ids.append(pos)
            res.append(pos)
        else:
            pos += 1
    return res

def build_NN_params(log_dir='scripts/mlearning/training_logs'):
    todo = []
    size = None
    tasks = ['task2']
    algos = ['head_wv','context_wv'] # ,'both_window_wv','head_wv'
    # vectorcombi = ('context_wv','head+context_wv')
    parameter_space = \
        [{'batch_size': 500,
          'dropout': 0.2,
          'hidden_units': [200, 500, 200],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 500,
          'dropout': 0.2,
          'hidden_units': [500, 200, 100],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 500,
          'dropout': 0.2,
          'hidden_units': [500, 500, 500],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 500,
          'dropout': 0.5,
          'hidden_units': [200, 500, 200],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 500,
          'dropout': 0.5,
          'hidden_units': [500, 200, 100],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 500,
          'dropout': 0.5,
          'hidden_units': [500, 500, 500],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 500,
          'dropout': None,
          'hidden_units': [200, 500, 200],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 500,
          'dropout': None,
          'hidden_units': [500, 200, 100],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 500,
          'dropout': None,
          'hidden_units': [500, 500, 500],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 750,
          'dropout': 0.2,
          'hidden_units': [200, 500, 200],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 750,
          'dropout': 0.2,
          'hidden_units': [500, 200, 100],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 750,
          'dropout': 0.2,
          'hidden_units': [500, 500, 500],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 750,
          'dropout': 0.5,
          'hidden_units': [200, 500, 200],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 750,
          'dropout': 0.5,
          'hidden_units': [500, 200, 100],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 750,
          'dropout': 0.5,
          'hidden_units': [500, 500, 500],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 750,
          'dropout': None,
          'hidden_units': [200, 500, 200],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 750,
          'dropout': None,
          'hidden_units': [500, 200, 100],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 750,
          'dropout': None,
          'hidden_units': [500, 500, 500],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 1000,
          'dropout': 0.2,
          'hidden_units': [200, 500, 200],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 1000,
          'dropout': 0.2,
          'hidden_units': [500, 200, 100],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 1000,
          'dropout': 0.2,
          'hidden_units': [500, 500, 500],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 1000,
          'dropout': 0.5,
          'hidden_units': [200, 500, 200],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 1000,
          'dropout': 0.5,
          'hidden_units': [500, 200, 100],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 1000,
          'dropout': 0.5,
          'hidden_units': [500, 500, 500],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 1000,
          'dropout': None,
          'hidden_units': [200, 500, 200],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 1000,
          'dropout': None,
          'hidden_units': [500, 200, 100],
          'optimizer': 'Adam',
          'steps': 200},
         {'batch_size': 1000,
          'dropout': None,
          'hidden_units': [500, 500, 500],
          'optimizer': 'Adam',
          'steps': 200}]
    new_ids = get_new_id(len(parameter_space)*len(algos), log_dir)
    i=0

    for t in tasks:
        for algorithm in algos:
            for model_param in parameter_space:
                # if size == 200000:
                #     status = 'priority'
                # else:
                #     status = 'waiting'
                if t == 'task1':
                    algorithm = 'head_wv'
                    if algorithm.startswith('style'):
                        f = ["head", 'corpus']
                    else:
                        f = ["head"]
                    window = None
                else:
                    # algorithm = 'style+both_window_wv'
                    window = 2
                    if algorithm.startswith('style'):
                        f = ["sentence", 'corpus']
                    else:
                        f = ["sentence"]
                setting = {
                        "task": t,
                        "status": "waiting",
                        "size": size,
                        "model": "DeepNN",
                        "param": {
                            "algorithm": algorithm,
                            "wv": "wiki_300",
                            "features": f,
                            "window": window
                        },
                        "model_param":model_param
                        }
                if setting not in [{key:value for key, value in todo[0].items() if key !='id'} for r in todo]:
                    setting['id'] = new_ids[i]
                    todo.append(setting)
                    i+=1
        write_training_log(todo,'scripts/mlearning/training_logs/DeepNN_head.json')
    return todo

def build_LogReg_params(log_dir='scripts/mlearning/emnlp_norm_training_logs'):
    todo = []
    size = None
    task = 'task2'
    algos = ['head_wv','head+context_wv','head+both_window_wv']
    parameter_space = \
        [{'C': 1, 'solver': 'liblinear'},
         {'C': 1, 'solver': 'sag'},
         {'C': 10, 'solver': 'liblinear'},
         {'C': 10, 'solver': 'sag'},
         {'C': 100, 'solver': 'liblinear'},
         {'C': 100, 'solver': 'sag'},
         {'C': 1000, 'solver': 'liblinear'},
         {'C': 1000, 'solver': 'sag'}]
    new_ids = get_new_id(len(algos)*len(parameter_space), log_dir)
    i=0
    # for size in sizes:
    for algorithm in algos:
        for model_param in parameter_space:
            # if size == 200000:
            #     status = 'priority'
            # else:
            #     status = 'waiting'
            if task == 'task1':
                f = ["head"]
                algorithm = 'style+head_wv'
                window = None
            else:
                f = ["sentence"]
                # algorithm = 'style+head+context_wv'
                # algorithm = 'both_window_wv'

                window = 2
            setting = {
                    "task": task,
                    "status": "server",
                    "size": size,
                    "model": "LogRegression",
                    "param": {
                        "algorithm": algorithm,
                        "wv": "wiki_300",
                        "features": f,
                        "window": window
                    },
                    "model_param":model_param
                    }
            if setting not in [{key:value for key, value in todo[0].items() if key !='id'} for r in todo]:
                setting['id'] = new_ids[i]
                todo.append(setting)
                i+=1
    write_training_log(todo,'scripts/mlearning/emnlp_norm_training_logs/LogRegression.json')

def build_SVM_params(log_dir='scripts/mlearning/emnlp_norm_training_logs'):
    todo = []
    size = None
    status = 'server'
    task = 'task2'
    algos = ['head_wv','head+context_wv','head+both_window_wv']
    parameter_space = \
        [{'C': 1 },
         {'C': 10 },
         {'C': 100},
         {'C': 1000 }]
    # styles = [True,False]
    new_ids = get_new_id(len(algos)*len(parameter_space), log_dir)
    i=0
    for algorithm in algos:
        for model_param in parameter_space:
            if task == 'task1':
                f = ["head"]
                algorithm = 'head_wv'
                window = None
            else:
                f = ["sentence"]
                # algorithm = 'style+head+context_wv'
                window = 2
            setting = {
                "task": task,
                "status": status,
                "size": size,
                "model": "SVM",
                "param": {
                    "algorithm": algorithm,
                    "wv": "wiki_300",
                    "features": f,
                    "window": window
                },
                "model_param": model_param
            }
            if setting not in [{key: value for key, value in todo[0].items() if key != 'id'} for r in todo]:
                setting['id'] = new_ids[i]
                todo.append(setting)
                i += 1
    write_training_log(todo,'scripts/mlearning/emnlp_norm_training_logs/SVM.json')


def build_LSTM_params(log_dir='scripts/mlearning/emnlp_norm_training_logs'):
    todo = []
    hyper_parameters = {'emb_dim': [224, 320, 384,480,576],
                        'dropout_rate': [0.5,0.25,0.0],
                        'batch': [32, 64, 96, 128],
                        'epoch': [20],
                        'padding': [62],
                        'learning_rate': [0.001],
                        'layer':[1],
                        'bidirectional':[True],
                        'keep_anno': [True]
                        }
    parameter_space = list(ParameterGrid(hyper_parameters))
    print('Estimated time: {}h'.format(len(parameter_space)*7/60))
    new_ids = get_new_id(len(parameter_space), log_dir)
    i=0
    # for size in sizes:

    for model_param in parameter_space:
        # if size == 200000:
        #     status = 'priority'
        # else:
        #     status = 'waiting'
        setting = {
                "task": 'task2',
                "status": "waiting",
                "size": None,
                "model": "LSTM",
                "param": {
                    "algorithm": 'sentence_wv',
                    "wv": "wiki_300",
                    "features": ['sentence']
                },
                "model_param":model_param
                }
        if setting not in [{key:value for key, value in todo[0].items() if key !='id'} for r in todo]:
            setting['id'] = new_ids[i]
            todo.append(setting)
            i+=1
    write_training_log(todo,'scripts/mlearning/emnlp_norm_training_logs/LSTM_anno.json')

# def prepare_log(r,logpath):
#     # save session info in log file
#     if not logpath is None:
#         write_log_entry(r, log_path=logpath)
#     print('Current training settings:')
#     print(json.dumps(r, indent=4))

def start_timer(opt, logfile):
    opt['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    opt['status'] = 'training'
    if not logfile is None:
        write_log_entry(opt, 'data/logs/' + logfile)
    start = time.time()  # start timer
    return start, opt

def end_timer(opt, start, logfile):
    end = time.time()  # stop timer
    training_time = round(((end - start) / 60), 2)  # min
    opt['training_time'] = training_time
    opt['status'] = 'finished'
    if not logfile is None:
        write_log_entry(opt, 'data/logs/' + logfile)
    print('Finished training after {} min'.format(training_time))
    return opt