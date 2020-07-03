# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# Download and build the data if it does not exist.

import src.loaders.build_data as build_data
import os
from src.loaders.Semeval.helper import Loader


files = ['train','train2','dev','test2016','test2017']

def reformat(outpath, dtype, inpath,concat=True):
    print('reformatting:' + dtype)

    with open(os.path.join(outpath, dtype + '_A.txt'), 'w',encoding='utf-8') as Aout:
        with open(os.path.join(outpath, dtype + '_B.txt'), 'w',encoding='utf-8') as Bout:
            with open(os.path.join(outpath, dtype + '_C.txt'), 'w',encoding='utf-8') as Cout:
                questions,taska_exclude = Loader.loadXMLQuestions([inpath])
                # questions = OrderedDict(sorted(questions.items()))

                for k in sorted(questions.keys()):
                    # print(k)
                    # original question id, subject and question
                    id1 = questions[k]['id']
                    if concat:
                        doc1 = questions[k]['subject'] + ' ' + questions[k]['question']
                    else:
                        doc1 = questions[k]['question']

                    for r in questions[k]['related'].keys():
                        # print(r)
                        # related question id, subject and question
                        id2 = questions[k]['related'][r]['id']
                        if concat:
                            doc2 = questions[k]['related'][r]['subject'] + ' ' + questions[k]['related'][r]['question']
                        else:
                            doc2 = questions[k]['related'][r]['question']
                        labelB = questions[k]['related'][r]['B-label']
                        # encode labels
                        if labelB in ['Relevant','PerfectMatch']:
                            labelB = '1'
                        elif labelB in ['Irrelevant']:
                            labelB = '0'
                        else:
                            raise ValueError('Annotation {} for example {} not defined!'.format((labelB,id2)))
                        Bout.write(id1 + '\t' + id2 + '\t' + doc1 + '\t' + doc2 + '\t' + labelB + '\n')

                        for c in questions[k]['related'][r]['comments'].keys():
                            # print(c)
                            # comment id and comment
                            id3 = questions[k]['related'][r]['comments'][c]['id']
                            doc3 = questions[k]['related'][r]['comments'][c]['comment']
                            labelA = questions[k]['related'][r]['comments'][c]['A-label']
                            labelC = questions[k]['related'][r]['comments'][c]['C-label']
                            # encode labels
                            if labelA in ['Good']:
                                labelA = '1'
                            elif labelA in ['Bad','PotentiallyUseful']:
                                labelA = '0'
                            else:
                                raise ValueError('Annotation {} for example {} not defined!'.format((labelA,id3)))
                            if labelC in ['Good']:
                                labelC = '1'
                            elif labelC in ['Bad','PotentiallyUseful']:
                                labelC = '0'
                            else:
                                raise ValueError('Annotation {} for example {} not defined!'.format((labelC,id3)))
                            if r not in taska_exclude:
                                Aout.write(id2 + '\t' + id3 + '\t' + doc2 + '\t' + doc3 + '\t' + labelA + '\n')
                            Cout.write(id1 + '\t' + id3 + '\t' + doc1 + '\t' + doc3 + '\t' + labelC + '\n')

    # output format:
    #
    # dict(question_id => dict(
    #   question
    #   id
    #   subject
    #   comments = {}
    #   related = dict(related_id => dict(
    #     question
    #     id
    #     subject
    #     relevance
    #     comments = dict(comment_id => dict(
    #       comment
    #       date
    #       id
    #       username
    #     )
    #   )
    # )


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
        fnames = ['semeval2016-task3-cqa-ql-traindev-v3.2.zip','semeval2016_task3_test.zip','semeval2017_task3_test.zip']
        urls = ['http://alt.qcri.org/semeval2016/task3/data/uploads/' + fnames[0],
                'http://alt.qcri.org/semeval2016/task3/data/uploads/' + fnames[1],
               'http://alt.qcri.org/semeval2017/task3/data/uploads/' + fnames[2]]

        dpext = os.path.join(dpath, 'Semeval2017')
        build_data.make_dir(dpext)

        for fname, url in zip(fnames,urls):
            build_data.download(url, dpext, fname)
            build_data.untar(dpext, fname) # should be able to handle zip

        reformat(dpath, files[0],
                         os.path.join(dpext, 'v3.2/train/SemEval2016-Task3-CQA-QL-train-part1.xml'))
        reformat(dpath, files[1],
                         os.path.join(dpext, 'v3.2/train/SemEval2016-Task3-CQA-QL-train-part2.xml'))
        reformat(dpath, files[2],
                         os.path.join(dpext, 'v3.2/dev/SemEval2016-Task3-CQA-QL-dev.xml'))
        reformat(dpath, files[3],
                         os.path.join(dpext, 'SemEval2016_task3_test/English/SemEval2016-Task3-CQA-QL-test.xml'))
        reformat(dpath, files[4],
                         os.path.join(dpext, 'SemEval2017_task3_test/English/SemEval2017-task3-English-test.xml'))

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
