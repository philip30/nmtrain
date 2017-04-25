#!/bin/bash

protoc --proto_path nmtrain/proto --python_out nmtrain/protobuf.zip nmtrain/proto/*.proto
