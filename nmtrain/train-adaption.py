#!/usr/bin/env python3

import argparse
import nmtrain
import nmtrain.classifiers

""" Arguments """
parser = argparse.ArgumentParser("Adaptation trainer")
parser.add_argument("-c", "--config", required=True, type=str, help="The training configuration file")
args = parser.parse_args()

def main(args):
  # Initiation
  config = nmtrain.util.open_proto_str(args.config, nmtrain.adaptation_config_pb.TrainAdaptationConfig())
  sanity_check(config)
  nmtrain.environment.init(config)
  # Training
  trainer = nmtrain.trainers.SequenceGANTrainer(config)
  trainer(nmtrain.classifiers.RNN_NMT())

def sanity_check(args):
  pass

if __name__ == "__main__":
  main(args)
