import nmtrain

UNK = "<UNK>"
EOS = "<EOS>"

class Vocabulary(object):
  """ Class to represent static vocabulary of neural machine translation.
      vocab    = Vocabulary()
      sentence = "This is a test."
      parsed_sent = vocab.parse_sentence(sentence) # SentenceID
      print(vocab.sentence(parsed_sentence)) # Original sent
      print(vocab.word(parse_sent[0])) # First word
  """

  def __init__(self, add_unk=False, add_eos=False):
    # The real data goes here
    self.data = nmtrain.dictionary_pb.Vocabulary()
    # Pointer to data
    self.word_to_id = self.data.word_to_id
    self.id_to_word = self.data.id_to_word
    # Attributes
    self.check_rare = None
    self.frozen = False
    if add_unk:
      self.add_word(UNK, include_rare=True)

    if add_eos:
      self.add_word(EOS, include_rare=True)

  def add_word(self, word, include_rare=False):
    if not include_rare and self.check_rare is not None and self.check_rare(word):
      return self.unk_id()
    else:
      word_id = self.word_to_id.get(word, len(self.word_to_id))
      self.word_to_id[word]    = word_id
      self.id_to_word[word_id] = word
      return word_id

  def add_sentence(self, sentence, include_rare=False):
    return list(self.add_word(word, include_rare) for word in sentence)

  def sentence(self, word_ids, append_eos = True):
    ret = []
    for word_id in word_ids:
      ret.append(self.word(word_id))
    if append_eos:
      ret.append(EOS)
      ret = ret[:ret.index(EOS)]
    return " ".join(ret)

  def raw_sentence(self, word_ids):
    return " ".join([self.word(word_id) for word_id in word_ids])

  def parse_sentence(self, words):
    return list(self.parse_word(word) for word in words)

  def word(self, word_id):
    return self.id_to_word.get(int(word_id), UNK)

  def parse_word(self, word):
    if word not in self:
      return self.unk_id()
    else:
      return self[word]

  def check_special_id(self, word_id):
    return word_id == self.unk_id() or word_id == self.eos_id()

  def check_special_word(self, word):
    return word == self.unk() or word == self.eos()

  def unk_id(self): return self[UNK]
  def eos_id(self): return self[EOS]
  def unk(self): return UNK
  def eos(self): return EOS

  def set_check_rare(self, check_rare):
    self.check_rare = check_rare

  def set_frozen(self, frozen):
    self.frozen = frozen

  def is_frozen(self):
    return self.frozen

  # Operators
  def __getitem__(self, word):
    return self.word_to_id[word]

  def __contains__(self, word):
    return word in self.word_to_id

  def __iter__(self):
    return iter(self.word_to_id)

  def __len__(self):
    return len(self.word_to_id)

  def __reversed__(self):
    return reversed(self.word_to_id)

  def __str__(self):
    return str(sorted(self.word_to_id.items(), key=lambda item: item[1]))

  def __eq__(self, other):
    if type(self) != type(other):
      return False
    else:
      return len(self) == len(other) and \
          self.id_to_word == other.id_to_word and \
          self.word_to_id == other.word_to_id
