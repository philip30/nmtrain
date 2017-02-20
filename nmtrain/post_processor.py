import re
import nmtrain

def post_process(prediction_output, trg_vocab):
  src_codec, trg_codec = nmtrain.environment.bpe_codec
  prediction_output.prediction = trg_vocab.sentence(prediction_output.prediction)
  if src_codec and trg_codec:
    prediction_output.prediction = re.sub(r'@@ ', '', prediction_output.prediction)

