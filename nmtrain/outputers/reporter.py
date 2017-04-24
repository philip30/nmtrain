import sys
import numpy
import nmtrain

from nmtrain.outputers.attention_writer import AttentionWriter

class TrainReporter(object):
  def __init__(self, stream, write_attention, src_vocab, trg_vocab):
    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab
    self.write_attention = write_attention
    self.attention_writer = nmtrain.outputers.AttentionWriter(stream, src_vocab, trg_vocab)
    self.stream = stream

  def __call__(self, src_batch, ref_batch, output):
    if self.stream is None or not self.stream:
      return
    # Gather all informations
    word_prob = numpy.concatenate([numpy.expand_dims(out, axis=2) for out in output["y"]], axis=2)
    word      = numpy.argmax(word_prob, axis=1)
    attention = None
    if self.write_attention and "a" in output:
      attention = numpy.concatenate([numpy.expand_dims(out, axis=2) for out in output["a"]], axis=2)

    # Generating sentence based report
    for i, (src, ref, out) in enumerate(zip(src_batch.transpose(), ref_batch.transpose(), word)):
      print("SRC:", self.src_vocab.raw_sentence(src), file=self.stream)
      print("REF:", self.trg_vocab.raw_sentence(ref), file=self.stream)
      print("OUT:", self.trg_vocab.raw_sentence(out), file=self.stream, end="\n\n")
      self.stream.flush()

      if self.write_attention:
        self.attention_writer(src, out, attention[i])

