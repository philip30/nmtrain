class LenTargetSorter(object):
  def __call__(self, parallel_data):
    parallel_data.sort(key=lambda ps: (len(ps.trg_sent), len(ps.src_sent)))
    return parallel_data

def from_string(string):
  if string == "lentrg":
    return LenTargetSorter()
