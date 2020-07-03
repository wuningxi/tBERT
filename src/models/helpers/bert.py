import tensorflow as tf
from src.preprocessing import bert_tokenization
import numpy as np

def get_bert_version(cased,large):
    if large:
        size = '24_H-1024_A-16'
    else:
        size = '12_H-768_A-12'
    if cased:
        BERT_version = 'cased_L-{}'.format(size)
    else:
        BERT_version = 'uncased_L-{}'.format(size)
    return BERT_version

def create_tokenizer(vocab_file, do_lower_case=False):
    return bert_tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def convert_sentence_pairs_to_features(S1, S2, tokenizer, max_seq_len):
    ids = [i for i in range(len(S1))]
    L = [None for i in range(len(S1))]
    examples = generate_examples(ids, S1, S2, L)
    features = convert_examples_to_features(examples, [None,0, 1], max_seq_len, tokenizer)
    input_ids = []
    input_mask = []
    segment_ids = []
    for feature in features:
        input_ids.append(feature.input_ids)
        input_mask.append(feature.input_mask)
        segment_ids.append(feature.segment_ids)
    return np.array(input_ids), np.array(input_mask), np.array(segment_ids)


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label

def generate_examples(ids,S1,S2,L):
    '''
    Create example objects from ids, sentence1s, sentence2s and labels
    :param ids: list of ids
    :param S1: list of sentences
    :param S2: list of sentences
    :param L: list of labels
    :return: list of example objects
    '''
    # toy example
    # ids = [0,1]
    # S1 = ['this is a test', 'absd']
    # S2 = ['this is another test', 'absd dsd']
    # L = [1,0]
    # generate_examples(ids,S1,S2,L)
    examples = []
    for i,s1,s2,l in zip(ids,S1,S2,L):
        example = InputExample(i, s1, s2, l)
        examples.append(example)
    return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 100000 == 0:
      tf.logging.info("Converting example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)

    features.append(feature)
  return features

def convert_single_example(ex_index, example, label_list, max_seq_length,tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [bert_tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


if __name__ == '__main__':
    # single toy example
    example = InputExample(231, 'This is a test', 'Here is another sentence', 0)
    BERT_version = 'cased_L-12_H-768_A-12'
    tokenizer = create_tokenizer('/Users/nicole/tf-hub-cache/{}/vocab.txt'.format(BERT_version), do_lower_case=True)
    max_seq_length = 20
    feature = convert_single_example(0,example, [0,1], max_seq_length, tokenizer)
    feature.label_id

    # multiple examples
    ids = [0,1]
    S1 = ['this is a test', 'absd']
    S2 = ['this is another test', 'absd dsd']
    L = [1,0]
    examples = generate_examples(ids,S1,S2,L)
    features = convert_examples_to_features(examples, [0,1], max_seq_length, tokenizer)