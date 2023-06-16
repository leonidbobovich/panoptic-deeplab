import sys

import numpy
import numpy as np
import torch
import torch.nn as nn
import torchvision
import time
from torch.quantization import QuantStub, DeQuantStub
from torch.utils.mobile_optimizer import optimize_for_mobile

'''
class network(torch.nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.transpose_operator_0 = None   # StatefulPartitionedCall/model_1/conv1_conv/Conv2D__10
        self.conv_operator_1 = torch.nn.quantized.Conv2d(3,16,(3,3))  # Conv__721
        #self.conv_operator_1 = torch.nn.Conv2d(3,16,(3,3))
        #self.relu_operator_2 = torch.nn.ReLU()



    def forward(self,
        input_var_0,
    ):
        #transpose_tensor_0 = torch.permute(input_var_0, dims=[0, 3, 1, 2])    # StatefulPartitionedCall/model_1/conv1_conv/Conv2D__10



        transpose_tensor_0 = input_var_0
        conv_tensor_1 = self.conv_operator_1(transpose_tensor_0)  # Conv__721
        #relu_tensor_2 = self.relu_operator_2(conv_tensor_1)



        return conv_tensor_1



#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device=torch.device('cpu')
batch_size = 1



m = network().to(device)
m.eval()



s=0



for i in range(10):
    weight = torch.FloatTensor(16,3,3,3).uniform_(0, 1)
    weight = torch.quantize_per_channel(weight, torch.ones(16)/200, torch.zeros(16, dtype=torch.int64), axis=0, dtype=torch.qint8)
    bias = torch.zeros(16)

    m.conv_operator_1.set_weight_bias(weight, bias)
    inp=torch.quantize_per_tensor(torch.FloatTensor(1,3,5000,5000).uniform_(0, 100), 0.1, 10, torch.quint8)



    t1=time.process_time_ns()
    out = m(inp)
    t2=time.process_time_ns()
    s+=t2-t1





#for i in range(10):
#    weight = torch.FloatTensor(16,3,3,3).uniform_(0, 1)
#    bias = torch.zeros(16)
#    
#    with torch.no_grad():
#        m.conv_operator_1.weight.copy_(weight)
#        m.conv_operator_1.bias.copy_(bias)
#    inp=torch.FloatTensor(1,3,5000,5000).uniform_(0, 100)
#    t1=time.process_time_ns()
#    out = m(inp)
#    t2=time.process_time_ns()
#    s+=t2-t1



print(s)
'''

import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
from torch.utils.mobile_optimizer import optimize_for_mobile


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=[3, 3], bias=True)
        self.dequant = DeQuantStub()

    def forward(self, x):
        return self.dequant(self.conv1(self.quant(x)))

torch.backends.quantized.engine = "qnnpack"

model_fp32 = TestModel()
model_fp32.eval()

model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')

# Fuse the activations to preceding layers, where applicable.
# This needs to be done manually depending on the model architecture.
# Common fusions include `conv + relu` and `conv + batchnorm + relu`
# model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, [['conv1']])


# Prepare the model for static quantization. This inserts observers in
# the model that will observe activation tensors during calibration.
model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)

# calibrate the prepared model to determine quantization parameters for activations
# in a real world setting, the calibration would be done with a representative dataset
input_fp32 = torch.randn(1, 3, 5000, 5000)
model_fp32_prepared(input_fp32)

# print(model_fp32_prepared.state_dict()['conv1.weight'][0])


# Convert the observed model to a quantized model. This does several things:
# quantizes the weights, computes and stores the scale and bias value to be
# used with each activation tensor, and replaces key operators with quantized
# implementations.
model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

model_int8.eval()
# print(model_int8.state_dict()['conv1.weight'][0])
s = 0

for i in range(10):
    weight = torch.FloatTensor(16,3,3,3).uniform_(0, 1)
    bias = torch.zeros(16)

    with torch.no_grad():
        model_fp32_prepared.conv1.weight.copy_(weight)
        model_fp32_prepared.conv1.bias.copy_(bias)
    inp=torch.FloatTensor(1,3,5000,5000).uniform_(0, 10)
    t1=time.process_time_ns()
    out = model_fp32_prepared(inp)
    # print("out",out[0,0,:10,:10])
    t2=time.process_time_ns()
    s+=t2-t1
    print('float', t2 - t1, s/1000/1000/1000/(i + 1))

s = 0
for i in range(10):
    # weight = torch.FloatTensor(16, 3, 3, 3).uniform_(0, 1)
    # weight = torch.quantize_per_channel(weight, torch.ones(16), torch.zeros(16, dtype=torch.int64), axis=0,
    #                                     dtype=torch.qint8)
    # weight = torch.quantize_per_tensor_dynamic(weight, dtype=torch.qint8, reduce_range=False)
    # bias = torch.zeros(16)

    # model_int8.conv1.set_weight_bias(weight, bias)
    # print(model_int8.state_dict()['conv1.weight'][0, 0, :10, :10])
    # inp=torch.quantize_per_tensor(torch.FloatTensor(1,3,5000,5000).uniform_(0, 100), 0.1, 10, torch.quint8)
    inp = torch.FloatTensor(1, 3, 5000, 5000).uniform_(0, 10)

    t1 = time.process_time_ns()
    out = model_int8(inp)
    # print('out', out[0, 0, :10, :10])
    t2 = time.process_time_ns()
    s += t2 - t1
    print('int', t2 - t1, s/1000/1000/1000/(i + 1))
sys.exit(0)