import nmtrain
import chainer

class TrainModelReader(object):
  def __init__ (self, nmtrain_model):
    self.nmtrain_model = nmtrain_model

  def load(self, config):
    nmtrain.serializers.serializer.load(self, config)

  def load_config(self, config_obj, from_config):
    if type(from_config) == nmtrain.train_config_pb.TrainingConfig:
      assert_proto_equal(config_obj.seed, from_config.seed)
      assert_proto_equal(config_obj.corpus.train_data, from_config.corpus.train_data)
      assert_proto_equal(config_obj.network_config, from_config.network_config)
      assert_proto_equal(config_obj.data_config, from_config.data_config)
      assert_proto_equal(config_obj.bpe_config, from_config.bpe_config)
      self.flag_same_opt = config_obj.learning_config.optimizer == from_config.learning_config.optimizer

    self.nmtrain_model.config.MergeFrom(config_obj)

  def load_state(self, state_obj):
    self.nmtrain_model.state = nmtrain.NmtrainState()
    self.nmtrain_model.state.data.MergeFrom(state_obj)

  def load_vocabularies(self, src_vocab, trg_vocab):
    self.nmtrain_model.src_vocab = nmtrain.Vocabulary()
    self.nmtrain_model.src_vocab.data.MergeFrom(src_vocab)
    self.nmtrain_model.trg_vocab = nmtrain.Vocabulary()
    self.nmtrain_model.trg_vocab.data.MergeFrom(trg_vocab)

  def load_lexicon(self, lexicon_obj):
    self.nmtrain_model.lexicon = nmtrain.structs.Lexicon(self.nmtrain_model.src_vocab,
                                                         self.nmtrain_model.trg_vocab,
                                                         self.nmtrain_model.config.lexicon_config.alpha,
                                                         self.nmtrain_model.config.lexicon_config.method)
    self.nmtrain_model.lexicon.data = lexicon_obj

  def load_bpe(self, bpe_obj):
    self.nmtrain_model.bpe_codec = bpe_obj

  def load_chainer_objects(self, weight_path, opt_path):
    self.nmtrain_model.chainer_model = nmtrain.structs.nmtrain_model.from_spec(
                                          self.nmtrain_model.config.network_config,
                                          self.nmtrain_model.config.learning_config,
                                          self.nmtrain_model.src_vocab,
                                          self.nmtrain_model.trg_vocab,
                                          self.nmtrain_model.lexicon)
    # Setup Optimizer
    self.nmtrain_model.optimizer = nmtrain.structs.nmtrain_model.parse_optimizer(self.nmtrain_model.config.learning_config.optimizer)
    self.nmtrain_model.optimizer.setup(self.nmtrain_model.chainer_model)
    # Loading state of the models
    chainer.serializers.load_npz(weight_path, self.nmtrain_model.chainer_model)
    chainer.serializers.load_npz(opt_path, self.nmtrain_model.optimizer)

# Module for sanity check
def assert_proto_equal(proto_old, proto_new):
  if proto_old != proto_new:
    nmtrain.log.fatal("This configuration should not be changed! \nOld:\n%s\nNew:\n%s" % (str(proto_old), str(proto_new)))

