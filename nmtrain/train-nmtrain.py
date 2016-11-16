#!/usr/bin/env python3

import argparse

import nmtrain.environment
import nmtrain.enumeration
import nmtrain.classifiers
import nmtrain.trainers

""" Arguments """
parser = argparse.ArgumentParser("NMT model trainer")
# Required
parser.add_argument("--src", type=str, required=True)
parser.add_argument("--trg", type=str, required=True)
parser.add_argument("--model_out", type=str, required=True)
# Parameters
parser.add_argument("--hidden", type=int, default=128, help="Size of hidden layer.")
parser.add_argument("--embed", type=int, default=128, help="Size of embedding vector.")
parser.add_argument("--batch", type=int, default=64, help="Number of (src) sentences in batch.")
parser.add_argument("--epoch", type=int, default=10, help="Number of max epoch to train the model.")
parser.add_argument("--depth", type=int, default=1, help="Depth of the network.")
parser.add_argument("--unk_cut", type=int, default=1, help="Threshold for words in corpora to be treated as unknown.")
parser.add_argument("--dropout", type=float, default=0.2, help="Dropout ratio for LSTM.")
parser.add_argument("--optimizer", type=str, default="adam:alpha=0.001,beta1=0.9,beta2=0.999,eps=1e-8", help="Optimizer used for BPTT")
# Configuration
parser.add_argument("--gpu", type=int, default=-1, help="Specify GPU to be used, negative for using CPU.")
parser.add_argument("--init_model", type=str, help="Init the model with the pretrained model.")
parser.add_argument("--model_architecture",type=str,choices=["encdec","attn"], default="attn", help="Type of model being trained.")
parser.add_argument("--seed", type=int, default=0, help="Seed for RNG. 0 for totally random seed.")
parser.add_argument("--verbosity", type=int, default=0, help="Verbosity level.")
parser.add_argument("--bptt_len", type=int, default=0, help="Length of iteration until bptt is trigerred. <= 0 for Infinite")
# Development set
parser.add_argument("--src_dev", type=str, help="Development data src")
parser.add_argument("--trg_dev", type=str, help="Development data trg")
# Attentional
parser.add_argument("--attention_type", type=str, choices=["dot", "general", "concat"], default="dot", help="How to calculate attention layer")
# DictAttn
parser.add_argument("--dict",type=str, help="Tab separated trg give src dictionary")
parser.add_argument("--dict_method", type=str, help="Method to be used for dictionary", choices=["bias", "linear"], default="bias")
args = parser.parse_args()

# Initiation
nmtrain.argument_checker.train_sanity_check(args)
nmtrain.environment.init(args, nmtrain.enumeration.RunMode.TRAIN)

# Load up data
trainer = nmtrain.trainers.MaximumLikelihoodTrainer(args)
trainer.train(nmtrain.classifiers.RNN_NMT())

