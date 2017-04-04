import numpy
import nmtrain

def post_process(prediction_output, trg_vocab, unk_lexicon, batch, src_vocab):
  batch_src_sent = batch.data[0].src_sent
  source_sent = batch_src_sent.tokenized + [src_vocab.eos()]
  prediction  = list(map(trg_vocab.word, prediction_output.prediction))
  # Check the availability of attention
  if hasattr(prediction_output, "attention"):
    attention_matrix = prediction_output.attention.transpose()
  else:
    attention_matrix = None

  # Merging BPE
  prediction, attention_matrix = merge_bpe(prediction, attention_matrix, normalize=True)
  merge_src, attention_matrix = merge_bpe(source_sent, attention_matrix.transpose() if attention_matrix is not None else None)
  batch_src_sent.annotate("bpe_merge", merge_src)
  # Unknown Replacement
  if attention_matrix is not None and unk_lexicon is not None:
    unknown_replace(merge_src, prediction, attention_matrix.transpose(), unk_lexicon, trg_vocab.unk())
  # Original output of the system
  prediction_list = [token for token in prediction]
  # Cutting OFF EOS
  prediction.append(trg_vocab.eos())
  prediction = prediction[:prediction.index(trg_vocab.eos())]
  # Assign the processed_output
  prediction_output.prediction_list = prediction_list
  prediction_output.prediction = " ".join(prediction)
  if hasattr(prediction_output, "attention"):
    prediction_output.attention = attention_matrix
  return prediction_output

def merge_bpe(word_vector, attention_matrix, normalize=False):
  if attention_matrix is not None:
    assert(len(attention_matrix) == len(word_vector))
  ## Handling the BPE output (target side)
  merge_flag = False
  prediction = []
  attention  = []
  for i, word in enumerate(word_vector):
    if merge_flag:
      if word.endswith("@@"):
        word = word[:-2]
      else:
        merge_flag = False
      prediction[-1] += word
      if attention_matrix is not None:
        attention[-1] += attention_matrix[i]
    else:
      if word.endswith("@@"):
        word = word[:-2]
        merge_flag = True
      prediction.append(word)
      if attention_matrix is not None:
        attention.append(attention_matrix[i])
  # Normalizing attention matrix
  if attention_matrix is not None:
    for i, attn_vector in enumerate(attention):
      if normalize:
        attention[i] = attention[i] / attention[i].sum()
      attention[i] = numpy.expand_dims(attention[i], axis=1)
    attention = numpy.concatenate(attention, axis=1).transpose()
  else:
    attention = None
  return prediction, attention

def unknown_replace(src, out, attn, lex, unk_sym):
  assert(len(out) == len(attn))
  for i, (word, attn) in enumerate(zip(out, attn)):
    if word == unk_sym:
      for src_index in reversed(numpy.argsort(attn)):
        src_word = src[src_index]
        if src_word in lex:
          out[i] = lex[src_word]
          break
  return out
