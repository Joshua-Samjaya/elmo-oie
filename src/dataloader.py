import torch
import torch.nn.utils.rnn as rnn
import numpy as np
import pandas
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from parsers.spacy_wrapper import spacy_whitespace_parser as spacy_ws
from common.symbols import SPACY_POS_TAGS
import json

import transformers
from transformers import BertForTokenClassification, BertConfig, BertTokenizer

class OpenIE_CONLL_Dataset(Dataset):
    def __init__(self, file_path, emb, sep='\t', sent_maxlen=300, label_map=None):
        '''
        data is a list of triples (according to data keys)
        label is a list of int
        '''
        self.file_path = file_path
        self.sep = sep
        self.emb = emb
        self.sent_maxlen = sent_maxlen
        self.label_map = label_map
        if label_map is None:
            self.label_map = LabelEncoder()
        self.classes = set()
        self.data = []
        self.labels = []
        self.data_keys = ["word_inputs", "predicate_inputs", "postags_inputs"]
        self.build()

    def __getitem__(self, i):
        x = []
        for key in self.data_keys:
            datum = self.data[key][i]
            x.append(datum)
        return x, self.labels[i]
    
    def __len__(self):
        return len(self.labels)

    def collate(self, data):
        x = [[],[],[]]
        y = []
        for i in data:
            for j in range(len(i[0])):
                x[j].append(torch.LongTensor(i[0][j]))
            y.append(torch.LongTensor(i[1]))
        return x, y

    def build(self):
        """
        Load a supervised OIE dataset from file
        """
        df = pandas.read_csv(self.file_path,
                             sep = self.sep,
                             header = 0,
                             keep_default_na = False)

        self.label_map.fit(df.label.values)

        # Split according to sentences and encode
        sents = self.get_sents_from_df(df)
        self.data = self.encode_inputs(sents)
        self.labels = self.encode_outputs(sents)
    
    def get_sents_from_df(self, df):
        """
        Split a data frame by rows accroding to the sentences
        """
        return [df[df.run_id == run_id]
                for run_id
                in sorted(set(df.run_id.values))]

    def encode_inputs(self, sents):
        """
        Given a dataframe which is already split to sentences,
        encode inputs for rnn classification.
        Should return a dictionary of sequences of sample of length maxlen.
        """
        word_inputs = []
        pred_inputs = []
        pos_inputs = []

        # Preproc to get all preds per run_id
        # Sanity check - make sure that all sents agree on run_id
        assert(all([len(set(sent.run_id.values)) == 1
                    for sent in sents]))
        run_id_to_pred = dict([(int(sent.run_id.values[0]),
                                self.get_head_pred_word(sent))
                               for sent in sents])

        # Construct a mapping from running word index to pos
        word_id_to_pos = {}
        for sent in sents:
            indices = sent.index.values
            words = sent.word.values

            for index, word in zip(indices,
                                   spacy_ws(" ".join(words))):
                word_id_to_pos[index] = word.tag_

        fixed_size_sents = sents # removed

        for sent in fixed_size_sents:

            assert(len(set(sent.run_id.values)) == 1)

            word_indices = sent.index.values
            sent_words = sent.word.values

            sent_str = " ".join(sent_words)

            pos_tags_encodings = [(SPACY_POS_TAGS.index(word_id_to_pos[word_ind]) \
                                   if word_id_to_pos[word_ind] in SPACY_POS_TAGS \
                                   else 0)
                                  for word_ind
                                  in word_indices]
            for hh in pos_tags_encodings:
                if hh > 55:
                    print(pos_tags_encodings)

            word_encodings = [self.emb.get_word_index(w)
                              for w in sent_words]

            # Same pred word encodings for all words in the sentence
            pred_word = run_id_to_pred[int(sent.run_id.values[0])]
            pred_word_encodings = [self.emb.get_word_index(pred_word)
                                    for _ in sent_words]

            word_inputs.append(word_encodings)
            pred_inputs.append(pred_word_encodings)
            pos_inputs.append(pos_tags_encodings)

        # Pad / truncate to desired maximum length
        # NOTE: removed pad in reimplementation
        ret = {}

        for name, sequence in zip(["word_inputs", "predicate_inputs", "postags_inputs"],
                                  [word_inputs, pred_inputs, pos_inputs]):
            ret[name] = []
            for samples in truncate_sequences(sequence,
                                         maxlen = self.sent_maxlen):
                ret[name].append(samples)

        return {k: np.array(v) for k, v in ret.items()}


    def encode_outputs(self, sents):
        """
        Given a dataframe split to sentences, encode outputs for rnn classification.
        Should return a list sequence of sample of length maxlen.
        """
        output_encodings = []
        # Encode outputs
        for sent in sents:
            output_encodings.append(list(self.transform_labels(sent.label.values)))

        return truncate_sequences(output_encodings, maxlen=self.sent_maxlen)

    def transform_labels(self, labels):
        """
        Encode a list of textual labels
        """
        # Fallback:
        return self.label_map.transform(labels)

    def num_of_classes(self):
        if self.label_map is not None:
            return len(self.label_map.classes_)
        else:
            print("encoder not instantiated for num of classes")
            return 0

    def get_head_pred_word(self, full_sent):
        """
        Get the head predicate word from a full sentence conll.
        """
        assert(len(set(full_sent.head_pred_id.values)) == 1) # Sanity check
        pred_ind = full_sent.head_pred_id.values[0]

        return full_sent.word.values[pred_ind] \
            if pred_ind != -1 \
               else full_sent.pred.values[0].split(" ")[0]


class OIE_BERT_Dataset(Dataset):
    def __init__(self, file_path, sep='\t', sent_maxlen=300, label_map=None, bert_model='bert-base-uncased'):
        '''
        data is a list of triples (according to data keys)
        label is a list of int
        '''
        self.file_path = file_path
        self.sep = sep
        self.sent_maxlen = sent_maxlen
        self.label_map = label_map
        self.bert_model = bert_model
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)
        if label_map is None:
            self.label_map = LabelEncoder()
        self.classes = set()
        self.data = []
        self.labels = []
        self.data_keys = ["word_inputs", "predicate_inputs", "postags_inputs"]
        self.build()

    def __getitem__(self, i):
        x = {}
        for key in self.data.keys():
            x[key] = self.data[key][i]
        return x, self.labels[i]
    
    def __len__(self):
        return len(self.labels)

    def collate(self, data):
        x = {}
        y = []
        batch_max_len = 0
        for i in data:
            for key in self.data.keys():
                x[key] = x.get(key, [])
                if key == 'word_inputs':
                    x[key].append(i[0][key])
                    batch_max_len = max(batch_max_len, len(i[0][key]))
                else:
                    x[key].append(torch.LongTensor(i[0][key]))
            y.append(torch.LongTensor(i[1]))
        x['predicate_inputs'] = torch.LongTensor(x['predicate_inputs'])
        bert_in = self.tokenizer.batch_encode_plus(x['word_inputs'], 
                            return_tensors='pt', pad_to_max_length=True, 
                            max_length=batch_max_len, return_lengths=True,
                            add_special_tokens = False)
        x['bert_inputs'] = bert_in
        return x, y

    def build(self):
        """
        Load a supervised OIE dataset from file
        """
        df = pandas.read_csv(self.file_path,
                             sep = self.sep,
                             header = 0,
                             keep_default_na = False)

        self.label_map.fit(df.label.values)

        # Split according to sentences and encode
        sents = self.get_sents_from_df(df)
        data, labels = self.encode_data(sents)
        self.data = data
        self.labels = labels
    
    def get_sents_from_df(self, df):
        """
        Split a data frame by rows accroding to the sentences
        """
        return [df[df.run_id == run_id]
                for run_id in sorted(set(df.run_id.values))]

    def encode_data(self, sents):
        """
        Given a dataframe which is already split to sentences,
        Should return a tuple of (sequences of sample of length maxlen, sequencecs of labels).
        """
        word_inputs = []
        pred_inputs = []
        pos_inputs = []

        output_encodings = []

        # Preproc to get all preds per run_id
        # Sanity check - make sure that all sents agree on run_id
        assert(all([len(set(sent.run_id.values)) == 1
                    for sent in sents]))
        run_id_to_pred = dict([(int(sent.run_id.values[0]),
                                self.get_head_pred_id(sent))
                               for sent in sents])

        # Construct a mapping from running word index to pos
        word_id_to_pos = {}
        for sent in sents:
            indices = sent.index.values
            words = sent.word.values

            for index, word in zip(indices, spacy_ws(" ".join(words))):
                word_id_to_pos[index] = word.tag_

        for sent in sents:

            assert(len(set(sent.run_id.values)) == 1)

            word_indices = sent.index.values
            sent_words = sent.word.values

            pos_tags_encodings = [(SPACY_POS_TAGS.index(word_id_to_pos[word_ind]) \
                                   if word_id_to_pos[word_ind] in SPACY_POS_TAGS \
                                   else 0)
                                   for word_ind in word_indices]

            # Same pred word encodings for all words in the sentence
            word_encodings = sent_words.tolist()
            pred_id = run_id_to_pred[int(sent.run_id.values[0])]
            pred_word_encodings = [pred_id]

            if pred_id != -1:
                word_inputs.append(word_encodings)
                pred_inputs.append(pred_word_encodings)
                pos_inputs.append(pos_tags_encodings)
                output_encodings.append(list(self.transform_labels(sent.label.values)))

        x = {}
        for name, sequence in zip(self.data_keys,
                                  [word_inputs, pred_inputs, pos_inputs]):
            x[name] = []
            for samples in truncate_sequences(sequence, maxlen = self.sent_maxlen):
                x[name].append(samples)

        y = truncate_sequences(output_encodings, maxlen=self.sent_maxlen)
        return x, y

    def transform_labels(self, labels):
        """
        Encode a list of textual labels
        """
        # Fallback:
        return self.label_map.transform(labels)

    def num_of_classes(self):
        if self.label_map is not None:
            return len(self.label_map.classes_)
        else:
            print("encoder not instantiated for num of classes")
            return 0

    def get_head_pred_word(self, full_sent):
        """
        Get the head predicate word from a full sentence conll.
        """
        assert(len(set(full_sent.head_pred_id.values)) == 1) # Sanity check
        pred_ind = full_sent.head_pred_id.values[0]

        return full_sent.word.values[pred_ind] \
            if pred_ind != -1 \
               else full_sent.pred.values[0].split(" ")[0]

    def get_head_pred_id(self, full_sent):
        # only get the id
        assert(len(set(full_sent.head_pred_id.values)) == 1) # Sanity check
        pred_ind = full_sent.head_pred_id.values[0]
        if pred_ind == -1:
            pred_word = full_sent.pred.values[0].split(" ")[0]
            words = full_sent.word.values.tolist()
            if pred_word in words:
                pred_ind = words.index(pred_word) # might not capture the second or later occurrence
            else:
                pred_ind = -1 # will be filtered out
        return pred_ind


def truncate_sequences(sequences, maxlen=None):
    ret = []
    if maxlen is not None:
        for seq in sequences:
            truc_seq = seq[:maxlen]
            ret.append(truc_seq)
    return ret