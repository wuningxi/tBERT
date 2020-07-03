# adapted from https://github.com/whiskeyromeo/CommunityQuestionAnswering

import xml.etree.ElementTree as ElementTree
import sys
from collections import OrderedDict


# preexisting
def getargvalue(name, required):
    output = False
    for arg in sys.argv:
        if arg[2:len(name)+2] == name:
            output = arg[3+len(name):]
    if required and not output:
        raise Exception("Required argument " + name + " not found in sys.argv")
    return output

def argvalueexists(name):
    output = False
    for arg in sys.argv:
        if arg[2:len(name)+2] == name:
            output = True
    return output


class Loader:

    @staticmethod
    def getfilenames():
        if not argvalueexists("questionfiles"):
            output = Loader.defaultfilenames()
        else:
            output = getargvalue("questionfiles", True).split(",")
        return output


    @staticmethod
    def defaultfilenames():
        filePaths = [
            # train 2016
            'data/semeval2017/v3.2/train/SemEval2016-Task3-CQA-QL-train-part1.xml',
            'data/semeval2017/v3.2/train/SemEval2016-Task3-CQA-QL-train-part2.xml',
            # train 2015
            # 'data/semeval2017/v3.2/train-more-for-subtaskA-from-2015/SemEval2015-Task3-CQA-QL-train-reformatted-excluding-2016-questions-cleansed.xml',
            # dev 2015
            # 'data/semeval2017/v3.2/train-more-for-subtaskA-from-2015/SemEval2015-Task3-CQA-QL-dev-reformatted-excluding-2016-questions-cleansed.xml',
            # dev 2016
            'data/semeval2017/v3.2/dev/SemEval2016-Task3-CQA-QL-dev.xml',
            # test 2015
            # 'data/semeval2017/v3.2/train-more-for-subtaskA-from-2015/SemEval2015-Task3-CQA-QL-test-reformatted-excluding-2016-questions-cleansed.xml',
            # test 2016
            'data/semeval2017/v3.2/test/SemEval2016-Task3-CQA-QL-test.xml',
            # test 2017
            'data/semeval2017/v3.2/test/SemEval2017-task3-English-test.xml'
        ]
        return filePaths


    @staticmethod
    def loadXMLQuestions(filenames):
        output = {}
        for filePath in filenames:
            print("\nParsing %s" % filePath)
            fileoutput,excludelist = Loader.parseTask3TrainingData(filePath)
            print("  Got %s primary questions" % len(fileoutput))
            if not len(fileoutput):
                raise Exception("Failed to load any entries from " + filePath)
            isTraining = "train" in filePath
            for q in fileoutput:
                fileoutput[q]['isTraining'] = isTraining
                for r in fileoutput[q]['related']:
                    fileoutput[q]['related'][r]['isTraining'] = isTraining
            output.update(fileoutput)
        print("\nTotal of %s entries" % len(output))
        return output,excludelist

    @staticmethod
    def parseTask3TrainingData(filepath):
        tree = ElementTree.parse(filepath)
        root = tree.getroot()
        OrgQuestions = OrderedDict()
        exclude_task_a = []
        for OrgQuestion in root.iter('OrgQuestion'):
            OrgQuestionOutput = {}
            OrgQuestionOutput['id'] = OrgQuestion.attrib['ORGQ_ID']
            OrgQuestionOutput['subject'] = OrgQuestion.find('OrgQSubject').text
            OrgQuestionOutput['question'] = OrgQuestion.find('OrgQBody').text
            OrgQuestionOutput['comments'] = {}
            OrgQuestionOutput['related'] = OrderedDict()
            OrgQuestionOutput['featureVector'] = []
            hasDuplicate = OrgQuestion.find('Thread').get('SubtaskA_Skip_Because_Same_As_RelQuestion_ID',None)
            if hasDuplicate is not None:
                exclude_task_a.append(OrgQuestion.find('Thread').attrib['THREAD_SEQUENCE'])
            # SubtaskA_Skip_Because_Same_As_RelQuestion_ID
            if OrgQuestionOutput['id'] not in OrgQuestions:
                OrgQuestions[OrgQuestionOutput['id']] = OrgQuestionOutput
            for RelQuestion in OrgQuestion.iter('RelQuestion'):
                RelQuestionOutput = {}
                RelQuestionOutput['id'] = RelQuestion.attrib['RELQ_ID']
                RelQuestionOutput['subject'] = RelQuestion.find('RelQSubject').text
                RelQuestionOutput['question'] = RelQuestion.find('RelQBody').text
                RelQuestionOutput['B-label'] = RelQuestion.attrib['RELQ_RELEVANCE2ORGQ']
                RelQuestionOutput['givenRank'] = RelQuestion.attrib['RELQ_RANKING_ORDER']
                RelQuestionOutput['comments'] = OrderedDict()
                RelQuestionOutput['featureVector'] = []
                for RelComment in OrgQuestion.iter('RelComment'):
                    RelCommentOutput = {}
                    RelCommentOutput['id'] = RelComment.attrib['RELC_ID']
                    RelCommentOutput['date'] = RelComment.attrib['RELC_DATE']
                    RelCommentOutput['username'] = RelComment.attrib['RELC_USERNAME']
                    RelCommentOutput['comment'] = RelComment.find('RelCText').text
                    RelCommentOutput['C-label'] = RelComment.attrib['RELC_RELEVANCE2ORGQ']
                    RelCommentOutput['A-label'] = RelComment.attrib['RELC_RELEVANCE2RELQ']
                    RelQuestionOutput['comments'][RelCommentOutput['id']] = RelCommentOutput
                # if RelQuestionOutput['question'] != None:
                if RelQuestionOutput['question'] == None:
                    RelQuestionOutput['question'] = ""
                OrgQuestions[OrgQuestionOutput['id']]['related'][RelQuestionOutput['id']] = RelQuestionOutput
                # else:
                # print("Warning: skipping empty question " + RelQuestionOutput['id'])
        return OrgQuestions, exclude_task_a

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


