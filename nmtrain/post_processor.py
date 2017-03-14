import re
import nmtrain

def post_process(prediction_output, trg_vocab):
  prediction_output.prediction = trg_vocab.sentence(prediction_output.prediction)
  prediction_output.prediction = re.sub(r'@@ ', '', prediction_output.prediction)

