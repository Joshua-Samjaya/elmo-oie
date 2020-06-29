import os, sys
sys.path.append("./common")
import logging
logging.basicConfig(level = logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

import pytorch_lightning as pl 
from model import RNN_Model, BERT_OIE
import json
from pprint import pformat
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--train', type=str, default=None,
                        help='train file')
    parser.add_argument('--dev', type=str, default=None,
                        help='dev file')
    parser.add_argument('--test', type=str, default=None,
                        help='test file')
    parser.add_argument('--load_hyperparams', type=str, default=None, 
                        help='load hyperparameters')
    parser.add_argument('--path', type=str, default='models/model.pt', metavar='M',
                        help='path for model')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='path for pretrained checkpoint')
    parser.add_argument('--loss', default="BCE", help='Loss function')
    parser.add_argument('--dataset', type=str, default=None,
                        help='dataset name (listed in config)')
    parser.add_argument('--no_tensorboard', action='store_true', default=False, 
                        help='disable tensorboard logging')
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--early-stop', action='store_true', default=False, 
                        help='enable early stopping')
    parser.add_argument('--use-bert', action='store_true', default=False, 
                        help='enable bert')
    args = parser.parse_args()

    train_fn = args.train
    dev_fn = args.dev
    test_fn = args.test

    if args.load_hyperparams is not None:
        json_fn = args.load_hyperparams
        logging.info("Loading model from: {}".format(json_fn))
        rnn_params = json.load(open(json_fn))["rnn"]
        rnn_params["classes"] = None  # Just to make sure the model computes the correct labels

    else:
        # Use some default params
        rnn_params = {"sent_maxlen":  20,
                        "hidden_units": pow(2, 10),
                        "num_of_latent_layers": 2,
                        "epochs": 10,
                        "trainable_emb": True,
                        "batch_size": 50,
                        "emb_filename": "../pretrained_word_embeddings/glove.6B.50d.txt",
        }


    logging.debug("hyperparams:\n{}".format(pformat(rnn_params)))
    model_dir = "../models/{}/".format(time.strftime("%d_%m_%Y_%H_%M"))
    logging.debug("Saving models to: {}".format(model_dir))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_class = RNN_Model
    if args.use_bert:
        model_class = BERT_OIE
    if args.checkpoint is not None:
        model = model_class(train_fn, dev_fn, test_fn,
                    model_dir=model_dir, **rnn_params)
        trainer = pl.Trainer(gpus=args.gpus, resume_from_checkpoint=args.checkpoint)
        trainer.test(model)
    else:
        model = model_class(train_fn, dev_fn, test_fn,
                    model_dir=model_dir, **rnn_params)
        trainer = pl.Trainer(gpus=args.gpus, max_epochs=rnn_params['epochs'], 
                                early_stop_callback=args.early_stop)
        trainer.fit(model)
        trainer.test(model)


if __name__ == "__main__":
    main()