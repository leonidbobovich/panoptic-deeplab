import os
import sys
import onnx
import torch
from pathlib import Path
import importlib
import onnxsim

directory_path, file_name = os.path.split(__file__)
os.chdir(os.path.join(directory_path, '..'))
sys.path.insert(0, os.path.join(directory_path, '..'))
wrapped = importlib.import_module( Path('ec16e4e22cf5451aa456af5aceb0afaa_evaluate.py').stem).network

class network(torch.nn.Module):
    def __init__(self):
        super(network, self).__init__()
        # self.bn = torch.nn.BatchNorm2d(num_features=3)
        self.wrapped = wrapped()

    def forward(self,  input_var_0):
        return self.wrapped( input_var_0.to(torch.float32) )
        #[x, y, z] = self.wrapped( torch.permute(input_var_0, [ 0, 3, 1, 2 ] ))
        #return torch.permute(x, [ 0, 2, 3, 1 ] ), torch.permute(y, [ 0, 2, 3, 1 ] ), torch.permute(z, [ 0, 2, 3, 1 ] )

model = network()
model.eval()
model_input = [1,3,768,1536]
print('model input:', model_input)
# input_tensor = torch.permute(torch.rand(model_input), [ 0, 2, 3, 1 ] )
input_tensor = torch.rand(model_input).to(torch.int32)
# model(input_tensor)
op=9
torch.onnx.export(model=model.cpu(), args=input_tensor, f='wrapped.onnx',
                  verbose=False, do_constant_folding=True, opset_version=op, input_names=["input"], output_names=["semantic","offset","center"])
model = onnx.load('wrapped.onnx')
model = onnxsim.simplify(model, overwrite_input_shapes={model.graph.input[0].name:[1,3,768,1536]})
onnx.save(model[0], 'wrapped.onnx')

sys.exit(0)