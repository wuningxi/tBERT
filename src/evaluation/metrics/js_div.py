import scipy
import numpy as np
from src.loaders.load_data import load_data
import re, math, collections
from nltk.corpus import stopwords

def js(p, q, print_dist=False):
#     _p = p / norm(p, ord=1)
#     _q = q / norm(q, ord=1)
    _p = p/p.sum()
    _q = q/q.sum()
    _m = (_p + _q) / 2
    if print_dist:
        print('p\n{}'.format(_p))
        print('q\n{}'.format(_q))
        print('m\n{}'.format(_m))
    return (scipy.stats.entropy(_p, _m,base=2) + scipy.stats.entropy(_q, _m,base=2)) / 2

# split into tokens and count individual tokens (dict)
def tokenize(_str,stopword_filter=False):
    # return _str.split(' ')
    # stopwords = ['and', 'for', 'if', 'the', 'then', 'be', 'is', 'are', 'will', 'in', 'it', 'to', 'that']
    tokens = collections.defaultdict(lambda: 0.)
    if stopword_filter:
        stop_words = set(stopwords.words('english'))
        for m in _str:
            # m = m.group(1).lower()
            # if len(m) < 2: continue
            if len(m) < 2 or m in stop_words:
                pass
            else:
                tokens[m] += 1
    else:
        for m in _str:
            # m = m.group(1).lower()
            # if len(m) < 2: continue
            # if len(m) < 2:
            #     pass
            # else:
            tokens[m] += 1
    return tokens

# get tokens not contained in each of the distributions, add with count 0
def add_missing_tokens(dict1,dict2):
    missing_from_dict1 = set(dict2.keys()).difference(set(dict1.keys())) # elements in dict2, but not dict1
#     print(missing_from_dict1)
    for v in missing_from_dict1:
        dict1[v]=0
    missing_from_dict2 = set(dict1.keys()).difference(set(dict2.keys())) # elements in dict1, but not dict2
#     print(missing_from_dict2)
    for v in missing_from_dict2:
        dict2[v]=0
    return dict1,dict2

# order dict by keys alphabetically and turn into list
def to_list(d1_complete,d2_complete):
    keys = sorted(list(d1_complete.keys()))
    # print(keys)
    l1 = [d1_complete[k] for k in keys]
    l2 = [d2_complete[k] for k in keys]
    return l1,l2

# put everything together in one function
def js_divergence(d1, d2,print_m=False):
    '''
    Calculates JS divergence between two documents (str)
    '''
    d1_complete,d2_complete = add_missing_tokens(tokenize(d1),tokenize(d2))
    if print_m:
        print(d1_complete)
        print(d2_complete)
    l1,l2 = to_list(d1_complete,d2_complete)
    return js(np.array(l1), np.array(l2),print_m)

def calculate_js_div(R1,R2):
    '''
    Loads dataset defined by opt, calculates Jensen-Shannon divergence for each sentence pair and returns nested list with Jaccard similarities in each subset.
    :param opt: option dictionary to load dataset
    :return : nested list with overlap ratios
    '''
    subset_overlap = []
    for n in range(len(R1)):
        sim_per_pair = []
        for i in range(len(R1[n])):
            s1 = R1[n][i]
            s2 = R2[n][i]
            try:
                sim = js_divergence(s1,s2) # change here
            except TypeError:
                print(s1)
                print(s2)
                print('---')
            sim_per_pair.append(sim)
        subset_overlap.append(sim_per_pair)
    return subset_overlap

if __name__=='__main__':

    # js(np.array([0.1, 0.9]), np.array([1.0, 0.0]), True)
    #
    # d1 = """Many research publications want you to use BibTeX, which better
    # organizes the whole process. Suppose for concreteness your source
    # file is x.tex. Basically, you create a file x.bib containing the
    # bibliography, and run bibtex on that file."""
    # d2 = """In this case you must supply both a \left and a \right because the
    # delimiter height are made to match whatever is contained between the
    # two commands. But, the \left doesn't have to be an actual 'left
    # delimiter', that is you can use '\left)' if there were some reason
    # to do it."""
    # d1_complete, d2_complete = add_missing_tokens(tokenize(d1), tokenize(d2))
    # print(d1_complete)
    # l1, l2 = to_list(d1_complete, d2_complete)
    # js(np.array(l1), np.array(l2), True)
    #
    # js_divergence(d1,d2)

    opt = {'dataset': 'MSRP', 'datapath': 'data/',
           'tasks': ['B'], 'n_gram_embd': False,
           'subsets': ['train', 'dev', 'test'], 'simple_padding': True, 'padding': True,
           'model': 'basic_cnn', 'load_ids': False, 'cache': True}
    js = calculate_js_div(opt)