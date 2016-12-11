
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
    self.rare_words = set()

    if add_unk:
      self.add_word(UNK)
    if add_eos:
      self.add_word(EOS)

  def add_word(self, word):
    word_id = self.word_to_id.get(word, len(self.word_to_id))
    self.word_to_id[word]    = word_id
    self.id_to_word[word_id] = word
    return word_id
  
  def add_sentence(self, sentence):
    return list(self.add_word(word) for word in sentence)

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

  def parse_sentence(self, words, ignore_rare=True):
    return list(self.parse_word(word, ignore_rare) for word in words)

  def word(self, word_id):
    return self.id_to_word.get(word_id, UNK)

  def parse_word(self, word, ignore_rare=True):
    if word not in self:
      return self.unk_id()
    else:
      word_id = self[word]
      if ignore_rare and word_id in self.rare_words:
        word_id = self.unk_id()
      return word_id

  def set_rare_word(self, word_id):
    """ Mark that this word_id is a rare word.
        Normally this rare words will not be included in training due to its low frequency.
        Every word that is parsed there after will be marked as UNK.
        Note:
          The word still in vocabulary, just it is marked as rare.
    """
    self.rare_words.add(word_id)
    return self.unk_id()

  def remap_unknown(self):
    """ This method should be called only by the transfomer.
        It remaps the ids in the vocabulary, based on the unknown words.
        The known words will get an early ids while the unknown words will get the last ids.
    """
    mapping_id   = {}
    mapping_rare = {}
    # Handle EOS, UNK
    has_unk = UNK in self.word_to_id
    has_eos = EOS in self.word_to_id
    now_id = len([flag for flag in [has_unk, has_eos] if flag])
    default_id = [self.eos_id(), self.unk_id()]
    # Create the mapping
    # For the content
    for word_id, word in self.id_to_word.items():
      if word_id not in self.rare_words and word_id not in default_id:
        mapping_id[word_id] = now_id, word
        now_id += 1
    # For the artificially generated token
    for id in default_id:
      mapping_id[id] = id, self.word(id)
    # For the rare words
    for word_id in self.rare_words:
      mapping_rare[word_id] = now_id, self.word(word_id)
      now_id += 1
    # Remap the whole vocabulary
    self.rare_words.clear()
    self.id_to_word.clear()
    self.word_to_id.clear()
    if has_unk: self.add_word(UNK)
    if has_eos: self.add_word(EOS)
    for _, (new_id, word) in mapping_id.items():
      self.set_word(word, new_id)
    for _, (new_id, word) in mapping_rare.items():
      self.set_word(word, new_id)
      self.rare_words.add(new_id)
    # Return the mapping so the client will aware of the changed word_ids
    return mapping_id

  def unk_id(self): return self[UNK]
  def eos_id(self): return self[EOS]
  def unk(self): return UNK
  def eos(self): return EOS

  # Operators
  def __getitem__(self, word):
    return self.word_to_id[word]

  def __contains__(self, word):
    return word in self.word_to_id

  def __iter__(self):
    return iter(self.word_to_id)

  def __len__(self):
    return len(self.word_to_id) - len(self.rare_words)

  def __reversed__(self):
    return reversed(self.word_to_id)

  def __str__(self):
    return str(self.word_to_id)

  def __equal__(self, other):
    if type(self) != type(other):
      return False
    else:
      return self.id_to_word == other.id_to_word and \
          self.word_to_id == other.word_to_id and \
          self.rare_words == other.rare_words
