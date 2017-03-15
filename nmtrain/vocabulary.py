
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
    self.word_to_id = {}
    self.id_to_word = {}
    self.check_rare = None

    if add_unk:
      self.add_word(UNK)
    if add_eos:
      self.add_word(EOS)

  def add_word(self, word, include_rare=False):
    if not include_rare and self.check_rare(word):
      return self.unk_id()
    else:
      word_id = self.word_to_id.get(word, len(self.word_to_id))
      self.word_to_id[word]    = word_id
      self.id_to_word[word_id] = word
      return word_id

  def add_sentence(self, sentence, include_rare=False):
    return list(self.add_word(word, include_rare) for word in sentence)

  def set_word(self, word, word_id):
    self.word_to_id[word]    = word_id
    self.id_to_word[word_id] = word

  def sentence(self, word_ids, append_eos = True):
    ret = []
    for word_id in word_ids:
      ret.append(self.word(word_id))
    if append_eos:
      ret.append(EOS)
      ret = ret[:ret.index(EOS)]
    return " ".join(ret)

  def parse_sentence(self, words):
    return list(self.parse_word(word) for word in words)

  def word(self, word_id):
    return self.id_to_word.get(word_id, UNK)

  def parse_word(self, word):
    if word not in self:
      return self.unk_id()
    else:
      return self[word]

  def unk_id(self): return self[UNK]
  def eos_id(self): return self[EOS]
  def unk(self): return UNK
  def eos(self): return EOS

  def set_check_rare(self, check_rare):
    self.check_rare = check_rare

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
