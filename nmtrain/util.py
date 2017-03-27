def parse_parameter(opt_param, param_mapping):
  if len(opt_param) == 0:
    return {}
  param = {}
  for param_str in opt_param.split(","):
    param_str = param_str.split("=")
    assert len(param_str) == 2, "Bad parameter line: %s" % (opt_param)
    if param_str[0] not in param_mapping:
      raise ValueError("Unrecognized parameter:", param_str)
    else:
      param[param_str[0]] = param_mapping[param_str[0]](param_str[1])
  return param
