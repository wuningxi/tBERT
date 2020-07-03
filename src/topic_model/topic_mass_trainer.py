from src.topic_model.topic_trainer import lda_preprocess,load_sentences_for_topic_model,train_topic_model
from src.topic_model.topic_predictor import infer_and_write_word_topics,infer_and_write_document_topics
from src.topic_model.topic_loader import get_topic_root_folder,get_alpha_str
import os
from os import path

import argparse

# trains a number of topic models with different number of topics for specific dataset
# usage example: python src/topic_model/topic_mass_trainer.py -dataset Quora -min 10 -max 100 -alpha 50


def check_existing_topic_models(opt):
    topic_folder = get_topic_root_folder(opt)
    if path.exists(topic_folder):
        print('Looking for existing topic models for {} in {}'.format(opt['dataset'],topic_folder))
        prefix = opt['dataset']+get_alpha_str(opt)+'_'
        existing = [int(f.split('_')[-1]) for f in os.listdir(topic_folder) if f.startswith(prefix)]
        print('Found {}.'.format(existing))
        return existing
    else:
        return []

# read logs
def train_many_topic_models(opt,minimum_topics,maximum_topics,existing_models):
    # load input data
    data = load_sentences_for_topic_model(opt)
    # preprocess input for LDA
    corpus, id2word, processed_texts = lda_preprocess(data, id2word=None, delete_stopwords=True, print_steps=False)

    for n in range(minimum_topics, maximum_topics, 10):
        if n not in existing_models:
            print('===')
            print('Training topic model with {} topics '.format(n))
            # train model
            opt['num_topics'] = n
            topic_model = train_topic_model(corpus, id2word, processed_texts, opt)
            # predict
            for t in tasks:
                opt['tasks'] = [t]
                opt['subsets'] = subsets
                infer_and_write_document_topics(opt, topic_model, id2word)  # different for each subset and task
            infer_and_write_word_topics(opt, topic_model, id2word)  # same for each subset and task
        print('===')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--debug",
        type="bool",
        nargs="?",
        const=True,
        default=False,
        help="Use to run with tiny portion of actual data. ")
    parser.add_argument('-max', action="store", dest="max", type=int, default=None)
    parser.add_argument('-min', action="store", dest="min", type=int, default=None)
    parser.add_argument('-alpha', action="store", dest="alpha", type=float, default=None)
    parser.add_argument('-dataset', action="store", dest="dataset", type=str, default='Quora')
    parser.add_argument('-topic_type', action="store", dest="topic_type", type=str, default='ldamallet')

    FLAGS, unparsed = parser.parse_known_args()

    print('Dataset: {}'.format(FLAGS.dataset))
    print('Minimum: {}'.format(FLAGS.min))
    print('Maximum: {}'.format(FLAGS.max))
    if (FLAGS.alpha).is_integer():
        FLAGS.alpha = int(FLAGS.alpha)
    print('Alpha: {}'.format(FLAGS.alpha))

    datasets = [FLAGS.dataset]

    for d in datasets:
        print('Start topic model training for {}'.format(d))
        # construct opt
        if d == 'Semeval':
            tasks = ['A', 'B', 'C']
            subsets = ['train_large', 'test2016', 'test2017']
        else:
            tasks = ['B']
            subsets = ['train', 'dev', 'test']

        opt = {'dataset': d, 'datapath': 'data/', 'num_topics': None, 'topic_type': FLAGS.topic_type,
               'subsets': [subsets[0]], 'cache': True}
        if not FLAGS.alpha is None:
            opt['topic_alpha'] = FLAGS.alpha
        if FLAGS.debug:
            opt['max_m']=1000
            print('--- DEBUG MODE ---')

        existing = check_existing_topic_models(opt)
        minimum_topics = FLAGS.min
        maximum_topics = FLAGS.max+10

        if len([i for i in range(minimum_topics, maximum_topics, 10) if not i in existing])>0:
            print('Preparing to train the following topic models:')
            print([i for i in range(minimum_topics, maximum_topics, 10) if not i in existing])
            train_many_topic_models(opt,minimum_topics,maximum_topics,existing)
        else:
            print('All topic models already exist for specified number of topics.')