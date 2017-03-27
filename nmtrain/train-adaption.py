#!/usr/bin/env python3

import argparse

import nmtrain.log as log
import nmtrain.enumeration
import nmtrain.classifiers
import nmtrain.trainers
import nmtrain.arguments as builder

""" Arguments """
parser = argparse.ArgumentParser("NMT model trainer")
# Required
parser.add_argument("--init_model", type=str, required=True)
parser.add_argument("--orig_src", type=str, required=True)
parser.add_argument("--orig_trg", type=str, required=True)
parser.add_argument("--dest_src", type=str, default=None)
parser.add_argument("--dest_trg", type=str, default=None)
builder.add_max_sent_length(parser)
builder.add_seed(parser)
builder.add_sort_method(parser)
builder.add_gpu(parser)
builder.add_batch(parser)
builder.add_generation_limit(parser)
builder.add_memory_optimization(parser)
args = parser.parse_args()

def main(args):
  # Initiation
  sanity_check(args)
  nmtrain.environment.init(args, nmtrain.enumeration.RunMode.TRAIN)

  # Load up data
  trainer = nmtrain.trainers.SequenceGANTrainer(args)
  trainer.train(nmtrain.classifiers.RNN_NMT())

def sanity_check(args):
  pass

if __name__ == "__main__":
  main(args)
