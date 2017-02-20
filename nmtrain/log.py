import datetime
import sys

import nmtrain.environment as environment

# Global variable for this module
SILENCE = False

def info(*message, verbosity_level=0):
  if environment.verbosity >= verbosity_level and not SILENCE:
    print("[INFO]", datetime.datetime.now(), *message, file=sys.stderr)
    sys.stderr.flush()

def warning(*message, verbosity_level=0):
  if environment.verbosity >= verbosity_level and not SILENCE:
    print("[WARNING]", datetime.datetime.now(), *message, file=sys.stderr)
    sys.stderr.flush()

def fatal(*message, verbosity_level=0):
  if environment.verbosity >= verbosity_level and not SILENCE:
    print("[ERROR]", datetime.datetime.now(), *message, file=sys.stderr)
    sys.stderr.flush()

def silence(value=True):
  global SILENCE
  SILENCE = value
