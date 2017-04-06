def add_gpu(parser):
  parser.add_argument("--gpu", type=int, default=-1, help="Specify GPU to be used, negative for using CPU.")

def add_max_sent_length(parser):
  parser.add_argument("--max_sent_length", type=int, default=-1, help="Maximum length of training sentences in both sides")

def add_seed(parser):
  parser.add_argument("--seed", type=int, default=0, help="Seed for RNG. 0 for totally random seed.")

def add_sort_method(parser):
  parser.add_argument("--sort_method", type=str, choices=["lentrg"], default="lentrg")

def add_batch(parser):
  parser.add_argument("--batch", type=int, default=64, help="Number of items in batch.")
  parser.add_argument("--batch_strategy", type=str, choices=["word", "sent"], default="sent")

def add_generation_limit(parser):
  parser.add_argument("--gen_limit", type=int, default=50, help="Maximum Target Output Length.")

def add_memory_optimization(parser):
  parser.add_argument("--memory_optimization", type=int, default=0, help="Memory optimization level, 0 for no optimization. Effect running time.")

