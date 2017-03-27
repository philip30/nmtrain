import sys

class TrainingReporter(object):
  def __init__(self, specification, src_vocab, trg_vocab):
    if hasattr(specification, "report") and specification.report:
      self.reporting = True
      self.src_vocab = src_vocab
      self.trg_vocab = trg_vocab
    else:
      self.reporting = False

  @property
  def is_reporting(self):
    return self.reporting

  def train_report(self, src_batch, ref_batch, output_buffer):
    if not self.is_reporting:
      return
    # Generating sentence based report
    for src, ref, out in zip(src_batch.transpose(), ref_batch.transpose(), output_buffer.transpose()):
      print("SRC:", self.src_vocab.raw_sentence(src), file=sys.stderr)
      print("REF:", self.trg_vocab.raw_sentence(ref), file=sys.stderr)
      print("OUT:", self.trg_vocab.raw_sentence(out), file=sys.stderr, end="\n\n")
