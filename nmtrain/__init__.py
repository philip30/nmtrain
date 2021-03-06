# Loading all protocol buffer
import zipimport
import os

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "protobuf.zip")
importer = zipimport.zipimporter(path)
algebra_pb = importer.load_module("algebra_pb2")
hack_pb = importer.load_module("hack_config_pb2")
corpus_pb = importer.load_module("corpus_pb2")
data_pb = importer.load_module("data_pb2")
update_pb = importer.load_module("updates_pb2")
ensemble_pb = importer.load_module("ensemble_pb2")
postprocess_pb = importer.load_module("post_process_pb2")
output_pb = importer.load_module("output_pb2")
optimizer_pb = importer.load_module("optimizer_pb2")
evaluation_pb = importer.load_module("evaluation_pb2")
learning_pb = importer.load_module("learning_pb2")
network_pb = importer.load_module("network_pb2")
test_config_pb = importer.load_module("test_config_pb2")
train_config_pb = importer.load_module("train_config_pb2")
dictionary_pb = importer.load_module("dictionaries_pb2")
state_pb = importer.load_module("state_pb2")

# Adaptation
adaptation_pb = importer.load_module("adaptation_pb2")
adaptation_config_pb = importer.load_module("train_adaptation_pb2")

# Loading Rest of the modules
import nmtrain.debug
import nmtrain.data
import nmtrain.minrisk
import nmtrain.structs
import nmtrain.environment
import nmtrain.third_party
import nmtrain.log
import nmtrain.components
import nmtrain.util
import nmtrain.serializers
import nmtrain.models
import nmtrain.outputers
import nmtrain.testers
import nmtrain.trainers

from nmtrain.structs.vocabulary import Vocabulary
from nmtrain.structs.nmtrain_model import NmtrainModel
from nmtrain.structs.nmtrain_state import NmtrainState
