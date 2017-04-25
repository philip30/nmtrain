from tabulate import tabulate

class AttentionWriter(object):
  def __init__(self, stream, src_vocab, trg_vocab):
    self.stream = stream
    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab

  def __call__(self, src, out, attn_vector, id=None):
    if self.stream is None:
      return
    assert(attn_vector.shape == (len(src), len(out)))

    if id is not None:
      print(str(id) + ".", file=self.stream)

    src_sent = [self.src_vocab.word(src_word) for src_word in src]
    trg_sent = [" "] + [self.trg_vocab.word(out_word) for out_word in out]

    table = []
    table.append(trg_sent)
    for i in range(len(src_sent)):
      row = [src_sent[i]]
      row.extend(["%.3f" % num for num in attn_vector[i]])
      table.append(row)

    self.stream.write(tabulate(table, tablefmt="plain"))
    self.stream.write("\n\n")
