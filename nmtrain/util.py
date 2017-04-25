import google.protobuf as protobuf

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

def parse_proto_str(proto_str, proto_object):
  protobuf.text_format.Merge(str(proto_str), proto_object)
  return proto_object

def open_proto_str(proto_dir, proto_object):
  with open(proto_dir, "r") as fp:
    return parse_proto_str(fp.read(), proto_object)

