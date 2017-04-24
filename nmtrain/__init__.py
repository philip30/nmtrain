# Loading all protocol buffer
import zipimport
import os

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "protobuf.zip")
importer = zipimport.zipimporter(path)
data_config_pb = importer.load_module("data_pb2")
update_pb = importer.load_module("updates_pb2")
postprocess_pb = importer.load_module("post_process_pb2")
output_pb = importer.load_module("output_pb2")
optimizer_pb = importer.load_module("optimizer_pb2")
evaluation_pb = importer.load_module("evaluation_pb2")
test_config_pb = importer.load_module("test_config_pb2")
train_config_pb = importer.load_module("train_config_pb2")
dictionary_pb = importer.load_module("dictionaries_pb2")
state_pb = importer.load_module("state_pb2")

# Loading Rest of the modules
import nmtrain.data
import nmtrain.structs
import nmtrain.environment
import nmtrain.third_party
import nmtrain.log
import nmtrain.chner
import nmtrain.util
import nmtrain.serializers
import nmtrain.models
import nmtrain.outputers
import nmtrain.testers


from nmtrain.structs.vocabulary import Vocabulary
from nmtrain.structs.nmtrain_model import NmtrainModel
from nmtrain.structs.nmtrain_state import NmtrainState
