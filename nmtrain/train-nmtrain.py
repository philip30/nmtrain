#!/usr/bin/env python3

import argparse
import os

import nmtrain.log as log
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
parser.add_argument("--src_max_vocab", type=int, default=50000, help="Maximum src vocabulary size in the model")
parser.add_argument("--trg_max_vocab", type=int, default=50000, help="Maximum trg vocabulary size in the model")
parser.add_argument("--early_stop", type=int, default=100, help="How many iterations should the model patiently keeps training before it stop due to low dev ppl")
# Configuration
parser.add_argument("--gpu", type=int, default=-1, help="Specify GPU to be used, negative for using CPU.")
parser.add_argument("--init_model", type=str, help="Init the model with the pretrained model.")
parser.add_argument("--model_architecture",type=str,choices=["encdec","attn"], default="attn", help="Type of model being trained.")
parser.add_argument("--seed", type=int, default=0, help="Seed for RNG. 0 for totally random seed.")
parser.add_argument("--verbosity", type=int, default=0, help="Verbosity level.")
parser.add_argument("--bptt_len", type=int, default=0, help="Length of iteration until bptt is trigerred. <= 0 for Infinite")
# Gradient
parser.add_argument("--gradient_clipping", type=float, default=5.0, help="Threshold for gradient clipping")
parser.add_argument("--gradient_noise_eta", type=float, default=0.3, dest="gradient_noise", help="Gradient noise eta inside noise N(0, eta/(1+t^gamma))")
# Development set
parser.add_argument("--src_dev", type=str, help="Development data source")
parser.add_argument("--trg_dev", type=str, help="Development data target")
parser.add_argument("--src_test", type=str, help="Testing source data, for per epoch testing")
parser.add_argument("--trg_test", type=str, help="Testing target data, for per epoch testing")
# Attentional Setting
parser.add_argument("--no_input_feeding", dest="input_feeding", action="store_false")
parser.add_argument("--attention_type", default="mlp", type=str, choices=["dot", "general", "mlp"])
parser.set_defaults(input_feeding=True)
args = parser.parse_args()

def main(args):
  # Initiation
  sanity_check(args)
  nmtrain.environment.init(args, nmtrain.enumeration.RunMode.TRAIN)

  # Load up data
  trainer = nmtrain.trainers.MaximumLikelihoodTrainer(args)
  trainer.train(nmtrain.classifiers.RNN_NMT())

def sanity_check(args):
  if (args.src_dev and not args.trg_dev) or (not args.src_dev and args.trg_dev):
    log.fatal("Need to specify both src_dev and trg_dev")
  if (args.src_test and not args.trg_test) or (not args.src_test and args.trg_test):
    log.fatal("Need to specify both src_test and trg_test")
  if any([getattr(args, attr) <= 0 for attr in ["batch", "epoch", "depth"]]):
    log.fatal("Batch, Epoch, Depth should be > 0")
  if any([getattr(args, attr) < 0 for attr in ["unk_cut"]]):
    log.fatal("Unknown Cut should be >= 0")
  if args.dropout < 0 or args.dropout > 1:
    log.fatal("Dropout should be 0 <= dropout <= 1")

if __name__ == "__main__":
  main(args)
