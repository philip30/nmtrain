import chainer
import chainer.functions as F

import nmtrain

# Implementation of Luong et al.
class LSTMAttentionalDecoder(chainer.Chain):
  def __init__(self, out_size, embed_size, hidden_size,
               dropout_ratio, lstm_depth, input_feeding,
               attention_type, lexicon):
    # Construct Appropriate Attention Chain
    if attention_type == "dot":
      attention = nmtrain.models.attentions.DotAttentionLayer()
    elif attention_type == "general":
      attention = nmtrain.models.attentions.GeneralAttentionLayer(hidden_size)
    elif attention_type == "mlp":
      attention = nmtrain.models.attentions.MLPAttentionLayer(hidden_size)
    else:
      raise ValueError("Unknown Attention Type:", attention_type)

    # Construct Appropriate Lexicon Chain
    if lexicon is not None:
      if lexicon.type == "bias":
        lexicon_model = nmtrain.models.lexicons.BiasedLexicon(lexicon.alpha)
      elif lexicon.type == "linear":
        lexicon_model = nmtrain.models.lexicons.LinearInterpolationLexicon(hidden_size)
      else:
        raise ValueError("Unknown Lexicon Type:", lexicon.type)

    # Register all 
    super(LSTMAttentionalDecoder, self).__init__(
      decoder         = nmtrain.chner.StackLSTM(embed_size, hidden_size, lstm_depth, dropout_ratio),
      context_project = chainer.links.Linear(hidden_size, hidden_size),
      affine_vocab    = chainer.links.Linear(hidden_size, out_size),
      output_embed    = chainer.links.EmbedID(out_size, embed_size),
      attention       = attention,
    )

    if lexicon is not None:
      self.add_link("lexicon_model", lexicon_model)

    if input_feeding:
      self.add_link("feeding_transform", chainer.links.Linear(hidden_size, embed_size))

    self.input_feeding = input_feeding
    self.dropout_ratio = dropout_ratio
    self.use_lexicon   = lexicon is not None

  def init(self, h):
    h, S, lexicon_matrix = h
    self.decoder.reset_state()
    self.S = S
    self.h = self.decoder(F.dropout(h,
                                    ratio=self.dropout_ratio,
                                    train=nmtrain.environment.is_train()))
    self.lexicon_matrix = lexicon_matrix

  def __call__(self):
    mem_optimize = nmtrain.optimization.chainer_mem_optimize
    # Calculate Attention vector
    a = self.attention(self.S, self.h)
    # Calculate context vector
    c = F.squeeze(F.batch_matmul(self.S, a, transa=True), axis=2)
    # Calculate hidden vector + context
    self.ht = self.context_project(self.h) + c
    # Calculate Word probability distribution
    y = mem_optimize(self.affine_vocab, F.tanh(self.ht), level=1)
    if self.use_lexicon:
      y_lex_enhanced, is_probability = self.lexicon_model(y, a, self.ht, self.lexicon_matrix)
    else:
      y_lex_enhanced, is_probability = y, False

    if not is_probability:
      y_lex_enhanced = F.softmax(y_lex_enhanced)

    if nmtrain.environment.is_train():
      return nmtrain.models.decoders.Output(y=y_lex_enhanced)
    else:
      # Return the vocabulary size output projection
      return nmtrain.models.decoders.Output(y=y_lex_enhanced, a=a)

  def update(self, next_word):
    # embed_size + hidden size -> input feeding approach
    decoder_update = self.output_embed(next_word)
    if self.input_feeding:
      decoder_update = self.feeding_transform(self.ht) + decoder_update
    self.h = self.decoder(decoder_update)

  def set_state(self, state):
    self.h, self.ht, state = state
    self.decoder.set_state(state)

  def state(self):
    return self.h, self.ht, self.decoder.state()

