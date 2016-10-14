import datetime
import sys

import nmtrain.environment as environment

def info(*message, verbosity_level=0):
  if environment.verbosity >= verbosity_level:
    print("[INFO]", datetime.datetime.now(), *message, file=sys.stderr)
  
def warning(*message, verbosity_level=0):
  if environment.verbosity >= verbosity_level:
    print("[WARNING]", datetime.datetime.now(), *message, file=sys.stderr)

def fatal(*message, verbosity_level=0):
  if environment.verbosity >= verbosity_level:
    print("[ERROR]", datetime.datetime.now(), *message, file=sys.stderr)