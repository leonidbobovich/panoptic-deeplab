import os
import sys
import onnx
import torch
from pathlib import Path
import importlib
import onnxsim
import torch
import torch.nn.utils.prune as prune

import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

directory_path, file_name = os.path.split(__file__)
os.chdir(os.path.join(directory_path, '..'))
sys.path.insert(0, os.path.join(directory_path, '..'))
network = importlib.import_module( Path('1ccd96976f0771588d3247cdcb1e3080_test_framerate.py').stem).network

model = network()
model.eval()
model = model.cpu()
model_input = [1,3,768,1536]
print('model input:', model_input)
input_tensor = torch.rand(model_input).to(torch.float32)
# model = model.to('mps')
# model(input_tensor.to('mps'))
model(input_tensor)
# new_model = model
# for name, module in new_model.named_modules():
#     # prune 20% of connections in all 2D-conv layers
#     if isinstance(module, torch.nn.Conv2d):
#         prune.l1_unstructured(module, name='weight', amount=0.5)
#     # # prune 40% of connections in all linear layers
#     # elif isinstance(module, torch.nn.Linear):
#     #     prune.l1_unstructured(module, name='weight', amount=0.4)
#
# print(dict(new_model.named_buffers()).keys())  # to verify that all masks exist

# parameters_to_prune = (
#     (model.conv_operator_5, 'weight'),
# )

# prune.global_unstructured(
#     parameters_to_prune,
#     pruning_method=prune.L1Unstructured,
#     amount=0.5,
# )

# for name, module in model.named_modules():
#     if isinstance(module, torch.nn.Conv2d):
#         print(
#             "Sparsity in {}: {:.2f}%".format(name,
#                 100. * float(torch.sum(module.weight == 0))
#                 / float(module.weight.nelement())
#             )
#         )

torch.onnx.export(model.cpu(), input_tensor.cpu(), 'analyze.onnx', opset_version=11)
model = onnx.load('analyze.onnx')
onnx.checker.check_model(model)
model = onnxsim.simplify(model, overwrite_input_shapes={model.graph.input[0].name:[1,3,768,1536]})
onnx.save(model[0], 'analyze_sim.onnx')
sys.exit(0)