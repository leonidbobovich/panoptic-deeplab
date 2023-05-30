import importlib
import sys
import os
from pathlib import Path
import torch

directory_path, file_name = os.path.split(sys.argv[1])
sys.path.insert(0, directory_path)
print(Path(file_name).stem)
print(directory_path)
network = importlib.import_module( Path(file_name).stem).network
model = network()
model.eval()
model_input = [1,3,768,1536]
print('model input:', model_input)
model(torch.rand(model_input))
op=11
torch.onnx.export(model=model.cpu(), args=torch.rand(model_input), f='test.onnx',
                  verbose=False, do_constant_folding=False, opset_version=op, input_names=["input"], output_names=["semantic","offset_m","center_h"])