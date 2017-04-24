# Modified from Rico Sennrich

import codecs

class BPE(object):

  def __init__(self, codes, separator='@@'):
    with codecs.open(codes, encoding='utf-8') as codes:
      self.bpe_codes = [tuple(item.split()) for item in codes]
    # some hacking to deal with duplicates (only consider first instance)
    self.bpe_codes = dict([(code,i) for (i,code) in reversed(list(enumerate(self.bpe_codes)))])
    self.cache     = {}
    self.separator = separator

  def segment(self, sentence):
    """segment single sentence (whitespace-tokenized string) with BPE encoding"""

    output = []
    for word in sentence:
      new_word = self.encode(word, self.bpe_codes)

      for item in new_word[:-1]:
        output.append(item + self.separator)
      output.append(new_word[-1])

    return output


  def encode(self, orig, bpe_codes):
    """Encode word based on list of BPE merge operations, which are applied consecutively"""

    if orig in self.cache:
      return self.cache[orig]

    word = tuple(orig) + ('</w>',)
    pairs = get_pairs(word)

    while True:
      bigram = min(pairs, key = lambda pair: bpe_codes.get(pair, float('inf')))
      if bigram not in bpe_codes:
        break
      first, second = bigram
      new_word = []
      i = 0
      while i < len(word):
        try:
          j = word.index(first, i)
          new_word.extend(word[i:j])
          i = j
        except:
          new_word.extend(word[i:])
          break

        if word[i] == first and i < len(word)-1 and word[i+1] == second:
          new_word.append(first+second)
          i += 2
        else:
          new_word.append(word[i])
          i += 1
      new_word = tuple(new_word)
      word = new_word
      if len(word) == 1:
        break
      else:
        pairs = get_pairs(word)

    # don't print end-of-word symbols
    if word[-1] == '</w>':
      word = word[:-1]
    elif word[-1].endswith('</w>'):
      word = word[:-1] + (word[-1].replace('</w>',''),)

    self.cache[orig] = word
    return word

def get_pairs(word):
  """Return set of symbol pairs in a word.
  word is represented as tuple of symbols (symbols being variable-length strings)
  """
  pairs = set()
  prev_char = word[0]
  for char in word[1:]:
    pairs.add((prev_char, char))
    prev_char = char
  return pairs

def codec_from_file(self, codes, separator="@@"):
  return BPE(codes, separator)

