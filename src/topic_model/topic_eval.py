from src.topic_model.topic_trainer import load_sentences_for_topic_model,lda_preprocess,load_topic_model
from src.topic_model.lda import evaluate_lda_topic_model
import os
from src.topic_model.topic_loader import get_topic_root_folder
import argparse

def evaluate_topic_model(topic_model, corpus, processed_texts, id2word, opt):
    if opt['topic_type'] in ['LDA','ldamallet']:
        results = evaluate_lda_topic_model(topic_model, corpus, processed_texts, id2word)
    elif opt['topic_type'] =='gsdmm':
        raise NotImplementedError()
    write_topic_model_log(opt, results)
    return results

def write_topic_model_log(opt,results):
    log_path = os.path.join(get_topic_root_folder(opt), 'eval_log.txt')
    if not os.path.exists(log_path):
        with open(log_path, 'a') as outfile:
            outfile.writelines('{}\t{}\t{}\t{}\n'.format('num_topics', 'topic_alpha','coherence','perplexity')) # todo: add metric for gsdmm evaluation?
    with open(log_path, 'a') as outfile:
        outfile.writelines('{}\t{}\t{}\t{}\n'.format(opt['num_topics'],opt.get('topic_alpha',50),results.get('coherence','N/A'),results.get('perplexity','N/A')))
        print('Wrote eval log: {}'.format(log_path))

if __name__ == '__main__':

    # get command line arguments
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--stem",type="bool",nargs="?",const=True,default=False)
    parser.add_argument('-dataset', action="store", dest="dataset", type=str, default='Quora')
    parser.add_argument('-topic_type', action="store", dest="topic_type", type=str, default='ldamallet')
    FLAGS, unparsed = parser.parse_known_args()

    # Import Dataset
    if FLAGS.dataset in ['Quora','MSRP']:
        subsets = ['train']  # , 'dev', 'test'
        tasks = ['B']
    elif FLAGS.dataset == 'Semeval':
        subsets = ['train_large'] # , 'test2016', 'test2017'
        tasks = ['A', 'B', 'C']
    opt = {'dataset': FLAGS.dataset, 'datapath': 'data/', 'topic_type': FLAGS.topic_type,
           'tasks': tasks, 'n_gram_embd': False, 'numerical': False,
           'subsets': subsets, 'cache': True, 'stem': FLAGS.stem}

    # load input data
    train_data = load_sentences_for_topic_model(opt)
    # preprocess input for LDA
    stem = opt.get('stem', False)
    lemmatize = opt.get('lemmatize', False)
    corpus, id2word, processed_texts = lda_preprocess(train_data, id2word=None, delete_stopwords=True, print_steps=True)
    for alpha in [0.1]: # ,10,1,0.1
        for topics in [t*10 for t in range(1,11)]:
            # load topic model, evaluate and save in log
            opt['num_topics'] = topics
            opt['topic_alpha'] = alpha
            topic_model = load_topic_model(opt)
            results = evaluate_topic_model(topic_model, corpus, processed_texts, topic_model.id2word, opt)
