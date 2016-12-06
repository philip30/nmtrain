#!/usr/bin/env python3

import sys
import argparse
import statistics
from collections import defaultdict
from tabulate import tabulate

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True)
parser.add_argument("--trg", required=True)
args = parser.parse_args()

def analyze(corpus_file_dir):
  count = defaultdict(int)
  length = []
  with open(corpus_file_dir) as corpus_file:
    for line in corpus_file:
      line = line.strip().split()
      for word in line:
        count[word] += 1
      length.append(len(line))
  ret = lambda: None
  ret.count = count
  ret.length = length
  return ret

src_corpus = analyze(args.src)
trg_corpus = analyze(args.trg)

data = []
data.append(("Word Count", str(sum(src_corpus.count.values())), str(sum(trg_corpus.count.values()))))
data.append(("Distinct Word", len(src_corpus.count), len(trg_corpus.count)))
data.append(("Max Sent Length", max(src_corpus.length), max(trg_corpus.length)))
data.append(("Total Sent", str(len(src_corpus.length)), str(len(trg_corpus.length))))
data.append(("Avg Sent Length", statistics.mean(src_corpus.length), statistics.mean(trg_corpus.length)))
data.append(("Var Sent Length", statistics.variance(src_corpus.length), statistics.mean(trg_corpus.length)))
print(tabulate(data, headers=["", "SRC", "TRG"], numalign="left"))
