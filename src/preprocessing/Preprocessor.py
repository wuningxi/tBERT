# adapted from https://github.com/whiskeyromeo/CommunityQuestionAnswering

import nltk
import re
import numpy as np
import os
from pathlib import Path
from src.models.helpers.bert import convert_sentence_pairs_to_features,create_tokenizer,get_bert_version
# from src.models.helpers.bert import convert_sentence_pairs_to_features,create_tokenizer

def get_homedir():
    '''
    Returns user homedir across different platforms
    :return:
    '''
    return str(Path.home())

def reduce_embd_id_len(E1,tasks, cutoff=100):
    '''
    Reduces numpy array dimensions for word embedding ids to save computational resources, e.g. from (2930, 200) to (2930, 100)
    :param E1: document represented as numpy array with word ids of shape (m,sent_len)
    :param cutoff: sentence length after cutoff
    :return: shortened E1
    '''
    if len(tasks) > 1:
        raise NotImplementedError('Not implemented minimum length with multiple tasks yet.')
    # cut length of questions to 100 tokens, leave answers as is
    # only select questions
    E1_short = []
    for sub in E1:
        # reduce length (drop last 100 elements in array)
        d = np.delete(sub, np.s_[cutoff:], 1)
        E1_short.append(d)
    assert E1_short[-1].shape == (E1[-1].shape[0], 100)
    return E1_short

# same as convert_to_one_hot().T
def get_onehot_encoding(labels):
    classes = 2
    test = labels.reshape(labels.size, )
    onehotL = np.zeros((test.size, classes))
    onehotL[np.arange(test.size), test] = 1
    onehotL[np.arange(test.size), test] = 1
    return onehotL

class Preprocessor:
    vocab_processor = None

    @staticmethod
    def basic_pipeline(sentences):
        # process text
        print("Preprocessor: replace urls and images")
        sentences = Preprocessor.replaceImagesURLs(sentences)
        print("Preprocessor: to lower case")
        sentences = Preprocessor.toLowerCase(sentences)
        print("Preprocessor: split sentence into words")
        sentences = Preprocessor.tokenize_tweet(sentences)
        print("Preprocessor: remove quotes")
        sentences = Preprocessor.removeQuotes(sentences)
        return sentences

    @staticmethod
    def replaceImagesURLs(sentences):
        out = []
        # URL_tokens = ['<url>','<URL>','URLTOK']  # 'URLTOK' or '<URL>'
        # IMG_tokens = ['<pic>','IMG']
        URL_token = '<URL>'
        IMG_token = '<IMG>'

        for s in sentences:
            s = re.sub(r'(http://)?www.*?(\s|$)', URL_token+'\\2', s) # URL containing www
            s = re.sub(r'http://.*?(\s|$)', URL_token+'\\1', s) # URL starting with http
            s = re.sub(r'\w+?@.+?\\.com.*',URL_token,s) #email
            s = re.sub(r'\[img.*?\]',IMG_token,s) # image
            s = re.sub(r'< ?img.*?>', IMG_token, s)
            out.append(s)
        return out

    @staticmethod
    def removeQuotes(sentences):
        '''
        Remove punctuation from list of strings
        :param sentences: list with tokenised sentences
        :return: list
        '''
        out = []
        for s in sentences:
            out.append([w for w in s if not re.match(r"['`\"]+",w)])
            # # Twitter embeddings retain punctuation and use the following special tokens:
            # # <unknown>, <url>, <number>, <allcaps>, <pic>
            # # s = re.sub(r'[^\w\s]', ' ', s)
            # s = re.sub(r'[^a-zA-Z0-9_<>?.,]', ' ', s)
            # s = re.sub(r'[\s+]', ' ', s)
            # s = re.sub(r' +', ' ', s)  # prevent too much whitespace
            # s = s.lstrip().rstrip()
            # out.append(s)
        return out

    @staticmethod
    def stopwordsList():
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.append('...')
        stopwords.append('___')
        stopwords.append('<url>')
        stopwords.append('<img>')
        stopwords.append('<URL>')
        stopwords.append('<IMG>')
        stopwords.append("can't")
        stopwords.append("i've")
        stopwords.append("i'll")
        stopwords.append("i'm")
        stopwords.append("that's")
        stopwords.append("n't")
        stopwords.append('rrb')
        stopwords.append('lrb')
        return stopwords

    @staticmethod
    def removeStopwords(question):
        stopwords = Preprocessor.stopwordsList()
        return [i for i in question if i not in stopwords]

    @staticmethod
    def removeShortLongWords(sentence):
        return [w for w in sentence if len(w)>2 and len(w)<200]

    @staticmethod
    def tokenize_simple(iterator):
        return [sentence.split(' ') for sentence in iterator]

    @staticmethod
    def tokenize_nltk(iterator):
        return [nltk.word_tokenize(sentence) for sentence in iterator]

    @staticmethod
    def tokenize_tweet(iterator,strip=True):
        # tknzr = nltk.tokenize.TweetTokenizer(strip_handles=True, reduce_len=True)
        tknzr = nltk.tokenize.TweetTokenizer(strip_handles=True, reduce_len=True)
        result = [tknzr.tokenize(sentence) for sentence in iterator]
        if strip:
            result = [[w.replace(" ", "") for w in s] for s in result]
        return result

    @staticmethod
    def toLowerCase(sentences):
        out = []
        special_tokens = ['UNK','<IMG>','<URL>']
        for s in Preprocessor.tokenize_tweet(sentences):
            sent =[]
            # split sentences in tokens and lowercase except for special tokens
            for w in s:
                if w in special_tokens:
                    sent.append(w)
                else:
                    sent.append(w.lower())
            out.append(' '.join(sent))
        return out

    @staticmethod
    def max_document_length(sentences,tokenizer):
        sentences = tokenizer(sentences)
        return max([len(x) for x in sentences]) # tokenised length of sentence!

    @staticmethod
    def pad_sentences(sentences, max_length,pad_token='<PAD>',tokenized=False):
        '''
        Manually pad sentences with pad_token (to avoid the same representation for <unk> and <pad>)
        :param sentences: 
        :param tokenizer: 
        :param max_length: 
        :param pad_token: 
        :return: 
        '''
        if tokenized:
            tokenized = sentences
            return [(s + [pad_token] * (max_length - len(s))) for s in tokenized]
        else:
            tokenized = Preprocessor.tokenize_tweet(sentences)
            return [' '.join(s + [pad_token] * (max_length - len(s))) for s in tokenized]

    @staticmethod
    def reduce_sentence_len(r_tok,max_len):
        '''
        Reduce length of tokenised sentence
        :param r_tok: nested list consisting of tokenised sentences e.g. [['w1','w2'],['w3']]
        :param max_len: maximum length of sentence
        :return: nested list consisting of tokenised sentences, none longer than max_len
        '''
        return [s if len(s) <= max_len else s[:max_len] for s in r_tok]

    @staticmethod
    def map_topics_to_id(r_tok,word2id_dict,s_max_len,opt):
        r_red = Preprocessor.reduce_sentence_len(r_tok, s_max_len)
        r_pad = Preprocessor.pad_sentences(r_red, s_max_len, pad_token='UNK', tokenized=True)
        mapped_sentences = []
        for s in r_pad:
            ids = [word2id_dict[lemma] if lemma in word2id_dict.keys() else 0 for lemma in s] # todo:fix 0 for UNK
            assert len(ids)==s_max_len, 'id len for {} should be {}, but is {}'.format(s,s_max_len,len(ids))
            mapped_sentences.append(np.array(ids))
        return np.array(mapped_sentences)

    @staticmethod
    def map_files_to_bert_ids(T1, T2, max_length, calculate_mapping_rate=False, bert_cased=False, bert_large=False):
        '''
        Split raw text into tokens and map to embedding ids for all subsets
        :param T1: nested list with tokenized sentences in each subset e.g. [R1_train,R1_dev,R1_test]
        :param T1: nested list with tokenized sentences in each subset e.g. [R2_train,R2_dev,R2_test]
        :param max_length: number of tokens in longest sentence, int
        :param pretrained_embedding: use mapping from existing embeddings?, boolean
        :param padding_tokens: padding tokens to use, should be ['<PAD>'] or ['<PAD_L>','<PAD_R>']
        :return: {'E1':E1,'E2':E2, 'mapping_rates':mapping_rates or None}
        '''
        mapping_rates = [] #todo: fix mapping rates

        # set unused to None rather than []
        E1 = []
        E1_mask = []
        E1_seg = []
        # use new_bert preprocessing code to encode sentence pairs
        for S1,S2 in zip(T1,T2): # look through subsets
            BERT_version = get_bert_version(bert_cased, bert_large)
            if bert_cased:
                lower=False
            else:
                lower=True
            tokenizer = create_tokenizer('{}/tf-hub-cache/{}/vocab.txt'.format(get_homedir(),BERT_version),
                                         do_lower_case=lower)
            Preprocessor.word2id = tokenizer.vocab  # dict(zip(vocabulary, range(len(vocabulary))))
            Preprocessor.id2word = {v: k for k, v in Preprocessor.word2id.items()}
            # S1 = [' '.join(s) for s in S1]  # don't use tokenized version
            # S2 = [' '.join(s) for s in S2]  # don't use tokenized version
            input_ids_vals, input_mask_vals, segment_ids_vals = convert_sentence_pairs_to_features(S1,S2, tokenizer,max_seq_len=max_length) # double length due to 2 sentences
            assert input_ids_vals.shape == input_mask_vals.shape == segment_ids_vals.shape
            E1.append(input_ids_vals)
            E1_mask.append(input_mask_vals)
            E1_seg.append(segment_ids_vals)

        if not calculate_mapping_rate:
            mapping_rates = None
        return {'E1':E1,'E1_mask':E1_mask,'E1_seg':E1_seg,'E2':None, 'mapping_rates':mapping_rates, 'word2id':Preprocessor.word2id,'id2word':Preprocessor.id2word}