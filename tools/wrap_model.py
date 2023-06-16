import os
import sys
import onnx
import torch
from pathlib import Path
import importlib
import onnxsim
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

directory_path, file_name = os.path.split(__file__)
os.chdir(os.path.join(directory_path, '..'))
sys.path.insert(0, os.path.join(directory_path, '..'))
wrapped = importlib.import_module( Path('006ec4570cefa992966ad2447407c39d_framerate.py').stem).network

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
model = model.cpu()
model_input = [1,3,768,1536]
print('model input:', model_input)
# input_tensor = torch.permute(torch.rand(model_input), [ 0, 2, 3, 1 ] )
input_tensor = torch.rand(model_input).to(torch.float32)
with profile(activities=[ProfilerActivity.CPU], profile_memory=True, with_flops=True,  with_stack=True, record_shapes=True) as prof:
    with record_function("model_inference"):
        model(input_tensor)

# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=10))

prof.export_chrome_trace("trace.json")
# with profile(activities=[ProfilerActivity.CPU],
#         profile_memory=True, record_shapes=True) as prof:
#     model(input_tensor)
# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))


sys.exit(0)
# model(input_tensor)
op=11
torch.onnx.export(model=model.cpu(), args=input_tensor, f='framerate_test.onnx',
                  verbose=False, do_constant_folding=True, opset_version=op, input_names=["input"], output_names=["semantic","offset","center"])
model = onnx.load('framerate_test.onnx')
model = onnxsim.simplify(model, overwrite_input_shapes={model.graph.input[0].name:[1,3,768,1536]})
onnx.save(model[0], 'framerate_test.onnx')

sys.exit(0)