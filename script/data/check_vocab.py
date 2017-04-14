import argparse
import tempfile
import zipfile
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument("--words", nargs="+", type=str, default=[])
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--side", type=str, choices=["src","trg"], default="src")
args = parser.parse_args()

# Loading model
tmpdir = tempfile.mkdtemp()
zf = zipfile.ZipFile(args.model, mode="r")
zf.extract("src.vocab", tmpdir)
zf.extract("trg.vocab", tmpdir)

vocab = "src.vocab" if args.side == "src" else "trg.vocab"
with open(os.path.join(tmpdir, vocab), "rb") as vocab_file:
  vocab = pickle.load(vocab_file)

occurence = {}
for word in args.words:
  occurence[word] = word in vocab.word_to_id

for word, occ in sorted(occurence.items()):
  print(word, occ)
