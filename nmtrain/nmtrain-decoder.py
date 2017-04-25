#!/usr/bin/env python3

import argparse
import numpy
import sys

import nmtrain
import nmtrain.classifiers
import nmtrain.log as log

""" Arguments """
parser = argparse.ArgumentParser("NMT decoder")
parser.add_argument("-c", "--config", required=True, type=str, help="The decoding configuration file")
args = parser.parse_args()

def main(args):
  # Initiation
  config = nmtrain.util.open_proto_str(args.config, nmtrain.test_config_pb.TestConfig())
  sanity_check(config)
  nmtrain.environment.init(config)

  # Print Args
  nmtrain.log.info(str(config))

  # Manager of batches and data
  data_manager  = nmtrain.data.DataManager()
  # The model, chainer model inside
  model         = load_models(config.init_model)
  # Outputer
  outputer = nmtrain.outputers.Outputer(model.src_vocab, model.trg_vocab)
  outputer.register_outputer("test", config.output)
  # Watcher
  watcher = nmtrain.structs.watchers.Watcher(model.nmtrain_state)
  # Classifier, that run the data and the model
  classifier    = nmtrain.classifiers.RNN_NMT()
  # Load data
  log.info("Loading Data")
  data_manager.load_test(config.test_data, model)
  log.info("Loading Finished.")
  # Begin Testing
  tester = nmtrain.testers.tester.Tester(watcher, classifier, model, outputer, config)
  # If model is a single model
  if model.__class__.__name__ == "NmtrainModel":
    model = model.chainer_model
  model.set_train(False)
  outputer.test.begin_collection()
  tester(model = model,
         data  = data_manager.test_data,
         mode  = nmtrain.testers.TEST,
         outputer = outputer.test)
  outputer.test.end_collection()

def sanity_check(args):
  pass

def load_single_model(config):
  model = nmtrain.NmtrainModel(config)
  model.finalize_model()
  model.describe()
  return model

def load_models(config):
  models = config.model

  if len(models) == 1:
    model_config = nmtrain.test_config_pb.ModelConfig()
    model_config.init_model = models[0]
    nmtrain_models = load_single_model(model_config)
  else:
    # Multi Models
    nmtrain_models = []
    for model_dir in models:
      model_config = nmtrain.test_config_pb.ModelConfig()
      model_config.init_model = models[0]
      nmtrain_models.append(load_single_model(model_config))
    # How to ensemble them
    ensemble_method = config.ensemble.method
    if ensemble_method == "lint":
      nmtrain_models = nmtrain.models.nmt_ensemble.EnsembleLinearInterpolateNMT(config.ensemble.linear_interpolation, nmtrain_models)
    else:
      raise ValueError("Unknown ensemble method:", ensemble_method)
  return nmtrain_models

if __name__ == "__main__":
  main(args)
