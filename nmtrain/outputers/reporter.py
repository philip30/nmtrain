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
      trg_batch, ground_truth = src_batch

      with chainer.no_backprop_mode():
        out = chainer.functions.argmax(output["disc_out"][0], axis=1)
        out.to_cpu()
        out = out.data

      tp = 0
      for trg_sent, label, prediction in zip(trg_batch, ground_truth, out):
        if label == prediction:
          tp += 1
          score = "C"
        else:
          score = "W"

        sign = "+" if label == 1 else "-"

        sent = self.trg_vocab.sentence(trg_sent)
        print("  %s %s [%d -> %d] %s" % (sign, score, label, prediction, sent), file=self.stream)
      print("Prec = %f" % (tp / len(trg_batch)), file=self.stream)

    if "minrisk_sample" in output:
      minrisk_sample = output["minrisk_sample"][0]
      minrisk_delta = output["minrisk_delta"]
      minrisk_prob = output["minrisk_prob"]
      minrisk_item = output["minrisk_item"]

      src_batch = src_batch.transpose()

      if ref_batch is not None:
        ref_batch = ref_batch.transpose()
      for i, (src, item, delta, prob) in enumerate(zip(src_batch, minrisk_item, minrisk_delta, minrisk_prob)):
        print("SRC:", self.src_vocab.sentence(src), file=self.stream)
        if ref_batch is not None:
          print("REF:", self.trg_vocab.sentence(ref_batch[i]), file=self.stream)

        for j, (index, p_i, d_i) in enumerate(zip(item, prob, delta)):
          print(" s#%d prob=%.4f delta=%.4f: %s" % \
                (j, p_i, d_i, self.trg_vocab.sentence(minrisk_sample[index][i])), file=self.stream)

    self.stream.flush()
