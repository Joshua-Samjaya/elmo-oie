import numpy as np 
import os, sys 
import re 

from common.symbols import SPACY_POS_TAGS
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from load_pretrained_word_embeddings import Glove
from sklearn.preprocessing import LabelEncoder
from parsers.spacy_wrapper import spacy_whitespace_parser as spacy_ws
import json
import logging

# for torch
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule
from dataloader import OpenIE_CONLL_Dataset, OIE_BERT_Dataset

# for bert
import transformers
from transformers import BertModel, BertConfig, BertTokenizer


class RNN_Model(LightningModule):
    def __init__(self, train_file, dev_file, test_file, emb_filename=None, sent_maxlen=300,
                 batch_size = 32, seed = 42, sep='\t',
                 hidden_units = pow(2, 7),trainable_emb = True,
                 emb_dropout = 0.1, num_of_latent_layers = 2,
                 epochs = 10, pred_dropout = 0.1, model_dir = "./models/",
                 classes = None, pos_tag_embedding_size = 5, num_classes=15, 
                 num_workers = 0, lr=0.001):
        # NOTE: So far, num classes must be provided at the beginning
        super(RNN_Model, self).__init__()
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.model_dir = model_dir
        self.sent_maxlen = sent_maxlen
        self.batch_size = batch_size
        self.seed = seed
        self.sep = sep
        self.hidden_units = hidden_units
        self.emb_filename = emb_filename
        self.emb = Glove(emb_filename)
        self.embedding_size = self.emb.dim
        self.trainable_emb = trainable_emb
        self.emb_dropout = emb_dropout
        self.num_of_latent_layers = num_of_latent_layers
        self.epochs = epochs
        self.pred_dropout = pred_dropout
        self.classes = classes
        self.label_map = None
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.lr = lr
        if self.classes is not None:
            self.label_map = LabelEncoder()
            self.label_map.fit(self.classes)
        self.pos_tag_embedding_size = pos_tag_embedding_size

        np.random.seed(self.seed)
        # build_model
        self.build_model()
    
    def build_model(self):
        self.word_embedding = self.embed_word()
        self.pos_embedding = self.embed_pos()
        self.lstm = nn.LSTM(self.embedding_size*2+self.pos_tag_embedding_size, self.hidden_units, 
                                num_layers=self.num_of_latent_layers, 
                                bidirectional=True)

        ## Dropout
        self.dropout = nn.Dropout(self.pred_dropout)
        self.linears = nn.ModuleList([nn.Linear(self.hidden_units*2, self.hidden_units)])
        self.linears.extend([nn.Linear(self.hidden_units, self.hidden_units) for i in range(1)])
        linear_pred = nn.Linear(self.hidden_units, self.num_classes)
        self.linears.append(linear_pred)

    def forward(self, x):
        lengths = [len(i) for i in x[0]]
        batch_size = len(lengths)
        x = [rnn.pad_sequence(i) for i in x]
        sents, predicates, tags = x[0], x[1], x[2]

        embed_sent = self.word_embedding(sents)
        embed_predicate = self.word_embedding(predicates)
        embed_pos = self.pos_embedding(tags)
        embed = torch.cat([embed_sent, embed_predicate, embed_pos], dim=-1)
        
        out = rnn.pack_padded_sequence(embed, lengths, enforce_sorted=False)
        out, _ = self.lstm(out)
        out, _ = rnn.pad_packed_sequence(out)
        out = out.view(-1, batch_size, 2*self.hidden_units)
        for i in self.linears:
            out = i(out)
            out = F.relu(out)
            
        out = rnn.pack_padded_sequence(out, lengths, enforce_sorted=False)
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        self.train_dataset = OpenIE_CONLL_Dataset(self.train_file, self.emb, sep=self.sep, 
                    label_map=self.label_map, sent_maxlen=self.sent_maxlen)
        print("Num train instances:", len(self.train_dataset))
        self.label_map = self.train_dataset.label_map
        self.classes = self.label_map.classes_
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                    collate_fn=self.train_dataset.collate, num_workers=self.num_workers)
        return loader

    def val_dataloader(self):
        self.dev_dataset = OpenIE_CONLL_Dataset(self.dev_file, self.emb, sep=self.sep, 
                    label_map=self.label_map, sent_maxlen=self.sent_maxlen)
        print("Num dev instances:", len(self.dev_dataset))
        loader = DataLoader(self.dev_dataset, batch_size=self.batch_size, shuffle=False,
                    collate_fn=self.dev_dataset.collate, num_workers=self.num_workers)
        return loader

    def test_dataloader(self):
        dataset = OpenIE_CONLL_Dataset(self.test_file, self.emb, sep=self.sep, 
                    label_map=self.label_map, sent_maxlen=self.sent_maxlen)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                    collate_fn=dataset.collate, num_workers=self.num_workers)
        return loader

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = rnn.pack_sequence(y, enforce_sorted=False)
        loss = F.cross_entropy(y_hat.data, y.data)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = rnn.pack_sequence(y, enforce_sorted=False)
        loss = F.cross_entropy(y_hat.data, y.data)
        
        _, y_hat = torch.max(y_hat.data, dim=-1) # 1 is for the class
        acc = accuracy_score(y.data.cpu(), y_hat.cpu())
        acc = torch.tensor(acc, dtype=torch.float)
        return {'val_loss': loss, 'val_acc':acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc':avg_acc}
        return {'val_loss': avg_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = rnn.pack_sequence(y, enforce_sorted=False)
        
        _, y_hat = torch.max(y_hat.data, dim=-1) # 1 is for the class
        acc = accuracy_score(y.data.cpu(), y_hat.cpu())
        acc = torch.tensor(acc, dtype=torch.float)
        return {'test_acc': acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_acc'] for x in outputs]).mean()
        tensorboard_logs = {'test_acc': avg_loss}
        return {'avg_test_acc': avg_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def compute_accuracy_packed(self, y_hat, y):
        pass

    def predict_sentence(self, sent):
        """
        Return a predicted label for each word in an arbitrary length sentence
        sent - a list of string tokens
        """
        ret = []
        sent_str = " ".join(sent)

        # Extract predicates by looking at verbal POS

        preds = [(word.i, str(word))
                 for word
                 in spacy_ws(sent_str)
                 if word.tag_.startswith("V")]

        # Calculate num of samples (round up to the nearst multiple of sent_maxlen)
        num_of_samples = np.ceil(float(len(sent)) / self.sent_maxlen) * self.sent_maxlen

        # Run RNN for each predicate on this sentence
        for ind, pred in preds:
            cur_sample = self.create_sample(sent, ind)
            X = self.encode_inputs([cur_sample])
            ret.append(((ind, pred),
                        [(self.consolidate_label(label), float(prob))
                         for (label, prob) in
                         self.transform_output_probs(self.model.predict(X),           # "flatten" and truncate
                                                     get_prob = True).reshape(num_of_samples,
                                                                              2)[:len(sent)]]))
        return ret

    def get_head_pred_word(self, full_sent):
        """
        Get the head predicate word from a full sentence conll.
        """
        assert(len(set(full_sent.head_pred_id.values)) == 1) # Sanity check
        pred_ind = full_sent.head_pred_id.values[0]

        return full_sent.word.values[pred_ind] \
            if pred_ind != -1 \
               else full_sent.pred.values[0].split(" ")[0]

    def embed_word(self):
        #TODO: dropout and maxlen
        """
        Embed word sequences using self's embedding class
        """
        return self.emb.get_torch_embedding(freeze = not self.trainable_emb)

    def embed_pos(self):
        """
        Embed Part of Speech using this instance params
        """
        return nn.Embedding(len(SPACY_POS_TAGS), self.pos_tag_embedding_size)


class BERT_OIE(LightningModule):
    def __init__(self, train_file, dev_file, test_file, sent_maxlen=300,
                 batch_size = 32, seed = 42, sep='\t', bert_model='bert-base-uncased',
                 hidden_units = pow(2, 7),trainable_emb = True,
                 emb_dropout = 0.1, num_of_latent_layers = 2,
                 epochs = 10, pred_dropout = 0.1, model_dir = "./models/",
                 classes = None, pos_tag_embedding_size = 5, num_classes=15, 
                 num_workers = 0, lr=0.001, **kwargs):
        # NOTE: num classes must be provided at the beginning
        super(BERT_OIE, self).__init__()
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.model_dir = model_dir
        self.sent_maxlen = sent_maxlen
        self.batch_size = batch_size
        self.seed = seed
        self.sep = sep
        
        self.bert_model = bert_model
        self.bert = BertModel.from_pretrained(bert_model)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.hidden_units = self.bert.config.hidden_size
        self.embedding_size = self.bert.config.hidden_size
        
        self.trainable_emb = trainable_emb
        self.emb_dropout = emb_dropout
        self.num_of_latent_layers = num_of_latent_layers
        self.epochs = epochs
        self.pred_dropout = pred_dropout
        self.classes = classes
        self.label_map = None
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.lr = lr
        if self.classes is not None:
            self.label_map = LabelEncoder()
            self.label_map.fit(self.classes)
        self.pos_tag_embedding_size = pos_tag_embedding_size

        np.random.seed(self.seed)
        # build_model
        self.build_model()
    
    def build_model(self):
        self.pos_embedding = self.embed_pos()

        ## Dropout
        self.dropout = nn.Dropout(self.pred_dropout)
        self.linears = nn.ModuleList([nn.Linear(self.hidden_units*2+self.pos_tag_embedding_size, self.hidden_units)])
        # self.linears.extend([nn.Linear(self.hidden_units, self.hidden_units) for i in range(1)])
        linear_pred = nn.Linear(self.hidden_units, self.num_classes)
        self.linears.append(linear_pred)

    def forward(self, x):
        sents = x[self.dev_dataset.data_keys[0]]
        tags = x[self.dev_dataset.data_keys[2]]

        lengths = [len(i) for i in sents]
        bert_in = x['bert_inputs']
        input_ids = bert_in['input_ids']
        attention_mask = bert_in['attention_mask']
        bert_out = self.bert(input_ids, attention_mask)
        hidden = bert_out[0]

        pred_ids = x[self.dev_dataset.data_keys[1]].view(-1, 1, 1)
        pred_ids = pred_ids.repeat(1, 1, hidden.size(2))
        pred_rep = torch.gather(hidden, dim=1, index=pred_ids)
        pred_rep = pred_rep.repeat(1, hidden.size(1), 1)

        tags = rnn.pad_sequence(tags, batch_first=True)
        embed_pos = self.pos_embedding(tags)
        embed = torch.cat([hidden, pred_rep, embed_pos], dim=-1)
        out = embed
        for i in self.linears:
            out = i(out)
            out = F.relu(out)
        out = rnn.pack_padded_sequence(out, lengths, batch_first=True, enforce_sorted=False)
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        self.train_dataset = OIE_BERT_Dataset(self.train_file, sep=self.sep, 
                    label_map=self.label_map, sent_maxlen=self.sent_maxlen, bert_model=self.bert_model)
        print("Num train instances:", len(self.train_dataset))
        self.label_map = self.train_dataset.label_map
        self.classes = self.label_map.classes_
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                    collate_fn=self.train_dataset.collate, num_workers=self.num_workers)
        return loader

    def val_dataloader(self):
        self.dev_dataset = OIE_BERT_Dataset(self.dev_file, sep=self.sep, 
                    label_map=self.label_map, sent_maxlen=self.sent_maxlen, bert_model=self.bert_model)
        print("Num dev instances:", len(self.dev_dataset))
        loader = DataLoader(self.dev_dataset, batch_size=self.batch_size, shuffle=False,
                    collate_fn=self.dev_dataset.collate, num_workers=self.num_workers)
        return loader

    def test_dataloader(self):
        dataset = OIE_BERT_Dataset(self.test_file, sep=self.sep, 
                    label_map=self.label_map, sent_maxlen=self.sent_maxlen, bert_model=self.bert_model)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                    collate_fn=dataset.collate, num_workers=self.num_workers)
        return loader

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = rnn.pack_sequence(y, enforce_sorted=False)
        loss = F.cross_entropy(y_hat.data, y.data)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = rnn.pack_sequence(y, enforce_sorted=False)
        loss = F.cross_entropy(y_hat.data, y.data)
        
        _, y_hat = torch.max(y_hat.data, dim=-1) # 1 is for the class
        acc = accuracy_score(y.data.cpu(), y_hat.cpu())
        acc = torch.tensor(acc, dtype=torch.float)
        return {'val_loss': loss, 'val_acc':acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc':avg_acc}
        return {'val_loss': avg_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = rnn.pack_sequence(y, enforce_sorted=False)
        
        _, y_hat = torch.max(y_hat.data, dim=-1) # 1 is for the class
        acc = accuracy_score(y.data.cpu(), y_hat.cpu())
        acc = torch.tensor(acc, dtype=torch.float)
        return {'test_acc': acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_acc'] for x in outputs]).mean()
        tensorboard_logs = {'test_acc': avg_loss}
        return {'avg_test_acc': avg_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def compute_accuracy_packed(self, y_hat, y):
        pass

    def predict_sentence(self, sent):
        """
        Return a predicted label for each word in an arbitrary length sentence
        sent - a list of string tokens
        """
        ret = []
        sent_str = " ".join(sent)

        # Extract predicates by looking at verbal POS

        preds = [(word.i, str(word))
                 for word
                 in spacy_ws(sent_str)
                 if word.tag_.startswith("V")]

        # Calculate num of samples (round up to the nearst multiple of sent_maxlen)
        num_of_samples = np.ceil(float(len(sent)) / self.sent_maxlen) * self.sent_maxlen

        # Run RNN for each predicate on this sentence
        for ind, pred in preds:
            cur_sample = self.create_sample(sent, ind)
            X = self.encode_inputs([cur_sample])
            ret.append(((ind, pred),
                        [(self.consolidate_label(label), float(prob))
                         for (label, prob) in
                         self.transform_output_probs(self.model.predict(X),           # "flatten" and truncate
                                                     get_prob = True).reshape(num_of_samples,
                                                                              2)[:len(sent)]]))
        return ret

    def get_head_pred_word(self, full_sent):
        """
        Get the head predicate word from a full sentence conll.
        """
        assert(len(set(full_sent.head_pred_id.values)) == 1) # Sanity check
        pred_ind = full_sent.head_pred_id.values[0]

        return full_sent.word.values[pred_ind] \
            if pred_ind != -1 \
               else full_sent.pred.values[0].split(" ")[0]

    def embed_word(self):
        #TODO: dropout and maxlen
        """
        Embed word sequences using self's embedding class
        """
        return self.emb.get_torch_embedding(freeze = not self.trainable_emb)

    def embed_pos(self):
        """
        Embed Part of Speech using this instance params
        """
        return nn.Embedding(len(SPACY_POS_TAGS), self.pos_tag_embedding_size)