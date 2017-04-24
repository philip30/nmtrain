import sys
import os
import nmtrain
import chainer

from nmtrain.outputers.reporter import TrainReporter
from collections import defaultdict

class Outputer(object):
  def __init__(self, src_vocab, trg_vocab):
    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab
    self.outputers = []

  def register_outputer(self, name, config):
    config_type = type(config)
    if config_type == nmtrain.output_pb.TrainOutput:
      OutputerClass = TrainOutputer
    elif config_type == nmtrain.output_pb.TestOutput:
      OutputerClass = TestOutputer
    else:
      raise ValueError("Undefined type:", type(config))
    # Registering outputer
    self.__dict__[name] = OutputerClass(config, self.src_vocab, self.trg_vocab)
    self.outputers.append(self.__dict__[name])

  def close(self):
    for outputer in self.outputers:
      outputer.close()

class TrainOutputer(object):
  def __init__(self, config, src_vocab, trg_vocab):
    self.streams = []
    if config.report.generate:
      self.reporter = TrainReporter(add_stream(config.report.path, self.streams),
                                    config.report.attention,
                                    src_vocab, trg_vocab)

    self.model_out       = config.model_out
    self.save_models     = config.save_models
    self.generate_report = config.report.generate

  def begin_collection(self, src, ref):
    self.buffer = defaultdict(list)
    self.src_batch = src
    self.ref_batch = ref

  def end_collection(self):
    if self.generate_report:
      self.reporter(self.src_batch, self.ref_batch, self.buffer)

    # Cleaning up
    self.buffer.clear()
    self.src_batch = None
    self.ref_batch = None

  def __call__(self, output):
    if len(self.streams) != 0:
      self.collect_output(output, "y")
      self.collect_output(output, "a")

  def collect_output(self, output, key):
    if hasattr(output, key):
      targeted = getattr(output, key)
      targeted.to_cpu()
      self.buffer[key].append(targeted.data)

  def close(self):
    close_file_streams(self.streams)

class TestOutputer(object):
  def __init__(self, config, src_vocab, trg_vocab):
    self.prefix = config.output_prefix
    self.generate_attention = config.generate_attention
    self.attention_writer = None
    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab
    self.streams = []

    if not self.prefix:
      nmtrain.log.info("Skipping collection for dev/test.")
    else:
      nmtrain.log.info("Collecting output at: \"" + self.prefix + "\"")

  def begin_collection(self, epoch):
    if self.prefix:
      os.makedirs(os.path.abspath(os.path.join(self.prefix, os.pardir)), exist_ok=True)
      add_stream(self.prefix + "_" + str(epoch) + ".out", self.streams)
      if self.generate_attention:
        add_stream(self.prefix + "_" + str(epoch) + ".attn", self.streams)
        self.attention_writer = nmtrain.outputers.AttentionWriter(self.streams[-1], self.src_vocab, self.trg_vocab)

  def end_collection(self):
    close_file_streams(self.streams)
    self.streams.clear()

  def close(self):
    del self.streams

  def __call__(self, src, output, id):
    if len(self.streams) != 0:
      self.streams[0].write(output.prediction)
      self.streams[0].write("\n")
      if self.generate_attention and hasattr(output, "attention"):
        prediction_list = self.trg_vocab.parse_sentence(output.prediction_list)
        self.attention_writer(src, prediction_list, output.attention, id=id)

def add_stream(string, streams, mode="w"):
  if len(string) != 0:
    stream = open_stream_from_string(string, mode)
    streams.append(stream)
    return stream
  else:
    return None

def open_stream_from_string(string, mode="w"):
  if string == "STDOUT":
    return sys.stdout
  elif string == "STDERR":
    return sys.stderr
  else:
    return open(string, mode)

def close_file_streams(streams):
  for stream in streams:
    if stream != sys.stdout and stream != sys.stderr and stream != sys.stdin:
      if not stream.closed:
        stream.close()

