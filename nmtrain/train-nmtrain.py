#!/usr/bin/env python3

import argparse
import os
import nmtrain

import nmtrain.log as log
import nmtrain.classifiers
import nmtrain.trainers

""" Arguments """
parser = argparse.ArgumentParser("NMT model trainer")
parser.add_argument("-c", "--config", required=True, type=str, help="The training configuration file")
args = parser.parse_args()

def main(args):
  # Initiation
  config = nmtrain.util.open_proto_str(args.config, nmtrain.train_config_pb.TrainingConfig())
  sanity_check(config)
  nmtrain.environment.init(config)

  # Load up data
  trainer = nmtrain.trainers.MaximumLikelihoodTrainer(config)
  trainer(nmtrain.classifiers.RNN_NMT())

def sanity_check(config):
  # Corpus check
  corpus = config.corpus
  ## Train Data
  if not corpus.train_data.source or not corpus.train_data.target:
    log.fatal("Need to specify trainining data source and target!")
  else:
    log.fatal_if(not os.path.exists(corpus.train_data.source), "Could not find:", corpus.train_data.source)
    log.fatal_if(not os.path.exists(corpus.train_data.target), "Could not find:", corpus.train_data.target)
  ## Dev Data
  if not corpus.dev_data.source and not corpus.dev_data.target or \
      not corpus.dev_data.target and not corpus.dev_data.source:
        log.fatal("Need to specify both source and target (or not at all) for dev data")
  elif corpus.dev_data.source and corpus.dev_data.target:
    log.fatal_if(not os.path.exists(corpus.dev_data.source), "Could not find:", corpus.dev_data.source)
    log.fatal_if(not os.path.exists(corpus.dev_data.target), "Could not find:", corpus.dev_data.target)
  ## Test Data
  if not corpus.test_data.source and not corpus.test_data.target or \
      not corpus.test_data.target and not corpus.test_data.source:
        log.fatal("Need to specify both source and target (or not at all) for test data")
  elif corpus.test_data.source and corpus.test_data.target:
    log.fatal_if(not os.path.exists(corpus.test_data.source), "Could not find:", corpus.test_data.source)
    log.fatal_if(not os.path.exists(corpus.test_data.target), "Could not find:", corpus.test_data.target)

  # Learning Config
  learning_config = config.learning_config
  def check_dropout(dropout):
    if dropout < 0 or dropout > 1:
      log.fatal("All Dropout should be 0 <= dropout <= 1")

  check_dropout(learning_config.dropout.stack_lstm)
  check_dropout(learning_config.dropout.encode_embed)
  check_dropout(learning_config.dropout.decode_embed)
  check_dropout(learning_config.dropout.encode)
  check_dropout(learning_config.dropout.decode)

  data_config = config.data_config
  if data_config.unknown_training.method == "sentence_dropout":
    if data_config.src_max_vocab == -1 and data_config.trg_max_vocab == -1 and data_config.unk_cut == 0:
      nmtrain.log.info("Sentence Dropout training. Setting cut to 1 because no unknown option is specified.")
      data_config.unk_cut = 1

  # Output config
  output_config = config.output_config
  log.fatal_if(len(output_config.train.model_out) == 0, "Please specify the output_config.train.model_out")

if __name__ == "__main__":
  main(args)
