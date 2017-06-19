import sys
import numpy
import nmtrain
import chainer

from nmtrain.outputers.attention_writer import AttentionWriter

class TrainReporter(object):
  def __init__(self, stream, write_attention, src_vocab, trg_vocab, report_type=""):
    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab
    self.write_attention = write_attention
    self.attention_writer = nmtrain.outputers.AttentionWriter(stream, src_vocab, trg_vocab)
    self.stream = stream
    self.report_type = report_type

  def __call__(self, src_batch, ref_batch, output):
    if self.stream is None or not self.stream:
      return

    if self.report_type == "" or self.report_type == "nmt":
      return self.train_nmt_reporter(src_batch, ref_batch, output)
    elif self.report_type == "mrt":
      return self.train_mrt_reporter(src_batch, ref_batch, output)
    else:
      raise ValueError("Unrecognized report type:", self.report_type)

  def train_nmt_reporter(self, src_batch, ref_batch, output):
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
        src = [self.src_vocab.word(src_id) for src_id in src]
        self.attention_writer(src, out, attention[i])

  def train_mrt_reporter(self, src_batch, ref_batch, output):
    if "disc_out" in output:
      (src_batch, trg_batch), label = src_batch
      trg_batch = trg_batch.transpose()
      with chainer.no_backprop_mode():
        output = chainer.functions.argmax(output["disc_out"][0], axis=1).data

      print("Label =", label, file=self.stream)
      label = 1 if label else 0
      true = 0
      for item, disc_out, in zip(trg_batch, output):
        print("%5s %s" % (disc_out, self.trg_vocab.raw_sentence(item)) , file=self.stream)
        true += 1 if label == disc_out else 0
      print("Prec = %f" % (true / len(trg_batch)), file=self.stream)

    if "minrisk_sample" in output:
      minrisk_sample = output["minrisk_sample"][0]
      minrisk_delta = output["minrisk_delta"]
      minrisk_prob = output["minrisk_prob"]
      minrisk_item = output["minrisk_item"]

      src_batch = src_batch.transpose()

      if ref_batch is not None:
        ref_batch = ref_batch.transpose()
      for i, (src, item, delta, prob) in enumerate(zip(src_batch, minrisk_item, minrisk_delta, minrisk_prob)):
        print("SRC:", self.src_vocab.raw_sentence(src), file=self.stream)
        if ref_batch is not None:
          print("REF:", self.trg_vocab.raw_sentence(ref_batch[i]), file=self.stream)

        for j, (index, p_i, d_i) in enumerate(zip(item, prob, delta)):
          print(" s#%d prob=%.4f delta=%.4f: %s" % \
                (j, p_i, d_i, self.trg_vocab.raw_sentence(minrisk_sample[index][i])), file=self.stream)

    self.stream.flush()
