class LenTargetSorter(object):
  def __call__(self, parallel_data):
    parallel_data.sort(key=lambda ps: \
        (len(ps.trg_sent) if ps.trg_sent is not None else 0, \
         len(ps.src_sent) if ps.src_sent is not None else 0))
    return parallel_data

def from_string(string):
  if string == "lentrg":
    return LenTargetSorter()
