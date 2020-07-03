import src.loaders.build_data as build_data
import os
from src.loaders.Semeval.helper import Loader
import csv
import random

files = ['p_train','p_test']


def reformat_original(outpath, inpath):
    random.seed(1)
    print('reformatting:' + inpath)
    # randomly split in train and test set
    with open(os.path.join(outpath, 'quora_train_B.txt'), 'w', encoding='utf-8') as output_file:
        with open(inpath, newline='', encoding='utf-8') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            next(csv_reader) # skip header
            for pair_id, id1, id2, doc1, doc2, labelB in csv_reader:
                # print(pair_id + '\t' + id1 + '\t' + id2 + '\t' + doc1 + '\t' + doc2 + '\t' + labelB + '\n')
                # break
                output_file.write(id1 + '\t' + id2 + '\t' + doc1 + '\t' + doc2 + '\t' + labelB + '\n')
    # # randomly split in train and test set
    # with open(os.path.join(outpath, 'train' + '_B.txt'), 'w', encoding='utf-8') as training_file:
    #     with open(os.path.join(outpath, 'test' + '_B.txt'), 'w', encoding='utf-8') as testing_file:
    #         with open(inpath, newline='', encoding='utf-8') as f:
    #             data = f.read().split('\n')
    #             data = data[1:] # skip header
    #             random.shuffle(data) # shuffle
    #             # csv_reader = csv.reader(data, delimiter='\t')
    #             train_examples = round(len(data) * 0.99)
    #             i = 0
    #             for line in data:
    #                 try:
    #                     pair_id, id1, id2, doc1, doc2, labelB = line.split('\t')
    #                     # print(pair_id + '\t' + id1 + '\t' + id2 + '\t' + doc1 + '\t' + doc2 + '\t' + labelB + '\n')
    #                     # break
    #                     if i < train_examples:
    #                         training_file.write(id1 + '\t' + id2 + '\t' + doc1 + '\t' + doc2 + '\t' + labelB + '\n')
    #                     else:
    #                         testing_file.write(id1 + '\t' + id2 + '\t' + doc1 + '\t' + doc2 + '\t' + labelB + '\n')
    #                 except ValueError:
    #                     print(line)
    #                 i+=1

def reformat_split(outpath, dtype, inpath):
    print('reformatting:' + inpath)
    # reformat Wang's split
    with open(os.path.join(outpath, dtype+'_B.txt'), 'w', encoding='utf-8') as output_file:
        with open(inpath, newline='', encoding='utf-8') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            next(csv_reader) # skip header
            for pair_id, doc1, doc2, labelB in csv_reader:
                id1 = pair_id+'-1'
                id2 = pair_id+'-2'
                output_file.write(id1 + '\t' + id2 + '\t' + doc1 + '\t' + doc2 + '\t' + labelB + '\n')


def build(opt):
    dpath = os.path.join(opt['datapath'], opt['dataset'])
    embpath = os.path.join(opt['datapath'], 'embeddings')
    logpath = os.path.join(opt['datapath'], 'logs')
    modelpath = os.path.join(opt['datapath'], 'models')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)
        build_data.make_dir(embpath)
        build_data.make_dir(logpath)
        build_data.make_dir(modelpath)

        # Download the data.
        # Use data from Kaggle
        # raise NotImplemented('Quora data not implemented yet')

        # '/Users/nicole/code/CQA/data/Quora/quora_duplicate_questions.tsv'
        # fnames = ['quora_duplicate_questions.tsv']

        # urls = ['https://zhiguowang.github.io' + fnames[0]]

        dpext = os.path.join(dpath, 'data')
        # build_data.make_dir(dpext)

        # for fname, url in zip(fnames,urls):
        #     build_data.download(url, dpext, fname)
        #     build_data.untar(dpext, fname) # should be able to handle zip

        reformat_split(dpath, files[0], os.path.join(dpext, 'train.tsv'))
        reformat_split(dpath, files[1], os.path.join(dpext, 'dev_and_test.tsv'))
        # reformat_split(dpath, files[2], os.path.join(dpext, 'test.tsv'))

        # reformat(dpath, files[1], os.path.join(dpext, 'test.csv'))

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)