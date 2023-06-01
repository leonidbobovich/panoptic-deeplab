import glob
import os
import sys

import numpy
import onnx
import math
import torch
import struct
import ptflops
import hashlib
import numpy as np
import importlib
from onnx_to_pytorch_helper import *
from torch.onnx import TrainingMode
from typing import List, Dict


trainable = True


def is_input(onnx_model, node):
    for n in onnx_model.graph.input:
        if n.name in node.input:
            return True
    return False


def is_output(onnx_model, node):
    for n in onnx_model.graph.output:
        if n.name in node.name:
            return True
    return False


def is_real_input(node, weights: onnx.GraphProto, input_name: str):
    if input_name in node.input:
        for w in weights:
            if input_name == w.name:
                return False
        return True
    else:
        assert False
    return False


def convert_weight(v):
    r = 0
    if v.data_type == 1:  # float 32-bit
        if len(v.raw_data) == 0:
            return v.float_data
        if False and len(v.dims) == 1 and v.dims[0] == 1:
            # struct.unpack('>f', v.raw_data[0:3])
            # big endian https://stackoverflow.com/questions/56396524/how-to-convert-a-byte-array-to-float-in-python
            r = struct.unpack('=f', v.raw_data[0:4])
            r = r[0]
        else:
            r = []
            assert len(v.raw_data) // 4 * 4 == len(v.raw_data)
            for d in range(0, len(v.raw_data) // 4):
                r.append(struct.unpack('f', v.raw_data[(d * 4 + 0):(d * 4 + 4)])[0])
            s = v.dims[0]
            for d in range(1, len(v.dims)):
                s = s * v.dims[d]
            assert s == len(r)
            r = np.reshape(r, newshape=v.dims)
    elif v.data_type == 7:  # int64 ???
        if False and len(v.dims) == 1 and v.dims[0] == 1:
            r = struct.unpack('=q', v.raw_data[0:8])
            r = r[0]
        else:
            r = []
            assert len(v.raw_data) // 8 * 8 == len(v.raw_data)
            for d in range(0, len(v.raw_data) // 8):
                r.append(struct.unpack('=q', v.raw_data[(d * 8 + 0):(d * 8 + 8)])[0])
            s = v.dims[0]
            for d in range(1, len(v.dims)):
                s = s * v.dims[d]
            assert s == len(r)
            r = np.reshape(r, newshape=v.dims)
    else:
        print(v)
        print('v.data_type', v.data_type)
        assert False
    return r


def node_get_attribute_int(node: onnx.NodeProto, name: str, default: int):
    for a in node.attribute:
        if a.name == name:
            return a.i
    return default


def node_get_attribute_ints(node: onnx.NodeProto, name: str, default: List[int]):
    for a in node.attribute:
        if a.name == name:
            return a.ints
    return default


def node_get_attribute_float(node: onnx.NodeProto, name: str, default: float):
    for a in node.attribute:
        if a.name == name:
            return a.f
    return default


def node_get_attribute_string(node: onnx.NodeProto, name: str, default: str):
    for a in node.attribute:
        if a.name == name:
            return a.s
    return default


def onnx_node_to_torch(md5: str, node: onnx.NodeProto, inputs: List[str], counter: int, weights,
                       onnxrpt_name: Dict[str, str]):
    name = node.op_type.lower() + '_operator' + '_' + str(counter)
    output = node.op_type.lower() + '_tensor' + '_' + str(counter)
    comment = '' #  # ' + node.name
    tab = ' ' * 4
    offset = tab * 2
    prefix = offset + 'self.'
    prefix_in_init = prefix + name
    prefix_in_body = offset + output
    self_name = prefix + name
    nograd = tab + prefix
    post_stats = '' #'{}\n'.format(comment)
    # post_stats = '{}\n{}self.compute_post_stats(\'{}\',\'{}\',self.{},\'{}\',{})'.format(comment, offset, node.op_type,
    #                                                                                      name,
    #                                                                                      name, output, output)
    input_files = {}
    print(node.op_type, node.name, '<-', node.input, inputs)
    for ni in node.input:
        is_real = is_real_input(node, weights, ni)
        print(node.op_type, node.name, 'node input:', ni, 'is real:', is_real)
        if is_real_input(node, weights, ni):
            continue
        for w in weights:
            if w.name == ni:
                print(node.op_type, node.name, 'weight name:', w.name)
                input_files[ni] = os.path.join(md5,
                                               name + '_' + str(w.name).replace(':', '_').replace('/', '_').lower()
                                               + '.npy')
                data = convert_weight(w)
                np.save(input_files[ni], data)
                print(node.op_type, node.name, 'node input:', ni, 'saved in file:', input_files[ni])
                break
    print(node.op_type, node.name, 'output:', node.output, output)

    assert len(node.output) == 1
    if node.op_type == 'Add':
        # assert len(input_files.keys()) == 0
        assert len(inputs) > 0
        print(node.op_type, node.name, 'inputs=', inputs, 'input_files=', input_files)
        init = ''
        ni2tensor = {}
        for c in input_files.keys():
            var_name = 'self.tensor_{}'.format(counter)
            counter = counter + 1
            init = init + offset + '{} = torch.from_numpy(numpy.load(\'{}\')).float()\n'.format(var_name,
                                                                                                input_files[c])
            ni2tensor[c] = var_name
        body = ''
        for ni in node.input:
            if ni in onnxrpt_name.keys():
                t_name = onnxrpt_name[ni]
            elif ni in input_files.keys():
                t_name = ni2tensor[ni]
            else:
                print(node.op_type, node.name, 'input {} not found', ni)
                sys.exit(0)
            body = body + (' + ' if body else '') + t_name + (
                '.to({}.device)'.format(inputs[0]) if t_name != inputs[0] else '')
        print(node.op_type, node.name, body)
        return name, output, '{} = None {}\n{} {}'.format(prefix_in_init, comment, init, comment), \
            '{} = {} {}'.format(prefix_in_body, body, post_stats)
    elif node.op_type == 'Concat':
        axis = node_get_attribute_int(node, 'axis', 0)
        print(node.op_type, node.name, 'axis=', axis, 'inputs=', inputs, 'input_files=', input_files)
        init = ''
        ni2tensor = {}
        for c in input_files.keys():
            var_name = 'self.tensor_{}'.format(counter)
            counter = counter + 1
            init = init + offset + '{} = torch.from_numpy(numpy.load(\'{}\'))\n'.format(var_name, input_files[c])
            ni2tensor[c] = var_name
        body = ''
        for ni in node.input:
            if ni in onnxrpt_name.keys():
                body = body + onnxrpt_name[ni] + ', '
            elif ni in input_files.keys():
                body = body + ni2tensor[ni] + ', '
            else:
                print('input {} not found', ni)
                sys.exit(0)
        return name, output, '{} = None {}\n{} {}'.format(prefix_in_init, comment, init, comment), \
            '{} = torch.cat(dim={},tensors=({})) {}'.format(prefix_in_body, axis, body, post_stats)
    elif node.op_type == 'AveragePool':
        assert len(inputs) == 1 and len(node.input) == 1
        auto_pad = node_get_attribute_string(node, 'auto_pad', None)
        ceil_mode = node_get_attribute_int(node, 'ceil_mode', 0)
        count_include_pad = node_get_attribute_int(node, 'count_include_pad', 0)
        dilations = node_get_attribute_ints(node, 'dilations', None)
        kernel_shape = node_get_attribute_ints(node, 'kernel_shape', [])
        pads = node_get_attribute_ints(node, 'pads', None)
        strides = node_get_attribute_ints(node, 'strides', None)
        # TODO: add attributes to PT AvgPool
        if len(kernel_shape) == 1:
            return torch.nn.AvgPool1d(kernel_size=kernel_shape, stride=strides, padding=0)
        if len(kernel_shape) == 2:
            return name, output, prefix + name + '=torch.nn.AvgPool2d( ' + \
                                 'kernel_size=' + str(tuple(kernel_shape)) + ', ' + \
                                 'stride=' + str(tuple(strides)) + ', ' + \
                                 'padding=(0, 0)' + ', ' + \
                                 ')' + comment, \
                                 prefix_in_body + '=self.' + name + '(' + inputs[0] + ')' + post_stats
        if len(kernel_shape) == 3:
            return name, output, prefix + name + '=torch.nn.AvgPool3d( ' + \
                                 'kernel_size=' + str(kernel_shape) + ', ' + \
                                 'stride=' + str(strides) + ', ' + \
                                 'padding=(0, 0)' + ', ' + \
                                 ')' + comment, \
                                 prefix_in_body + '=' + name + '(' + inputs[0] + ')' + post_stats
    elif node.op_type == 'BatchNormalization':
        eps = node_get_attribute_float(node, 'epsilon', 1e-05)
        momentum = node_get_attribute_float(node, 'momentum', 0.9)
        training_mode = node_get_attribute_float(node, 'training_mode', 0)
        scale = input_files[node.input[1]]
        B = input_files[node.input[2]]
        mean = input_files[node.input[3]]
        var = input_files[node.input[4]]
        return name, output, \
            prefix + name + ' = torch.nn.BatchNorm2d(' + \
            'num_features=' + str(len(np.load(scale))) + ', ' + \
            'eps=' + str(eps) + ', ' + \
            'momentum=' + str(momentum) + ', ' + \
            'affine=' + str(scale is not None or B is not None) + ', track_running_stats=True' + \
            ')' + comment + '\n' + \
            offset + 'with torch.no_grad():\n' + \
            (nograd + name + '.weight.copy_(torch.from_numpy(numpy.load(\'' +
             scale + '\')).float())\n' if os.path.exists(scale)
             else ' # ' + nograd + name + '.weight.copy_(torch.zeros(' + prefix + name + '.weight.shape))\n') + \
            (nograd + name + '.bias.copy_(torch.from_numpy(numpy.load(\'' +
             B + '\')).float())\n' if os.path.exists(B)
             else ' # ' + nograd + name + '.bias.copy_(torch.zeros(' + prefix + name + '.bias.shape))\n') + \
            (nograd + name + '.running_mean.copy_(torch.from_numpy(numpy.load(\'' +
             mean + '\')).float())\n' if os.path.exists(mean)
             else nograd + name + '.running_mean.copy_(torch.zeros(' + prefix + name + '.running_means.shape))\n') + \
            (nograd + name + '.running_var.copy_(torch.from_numpy(numpy.load(\'' +
             var + '\')).float())\n' if os.path.exists(var)
             else nograd + name + '.running_var.copy_(torch.zeros(' + prefix + name + '.running_var.shape))\n'), \
            offset + output + '= self.' + name + '(' + inputs[0] + ')' + post_stats
        # elif len(n.kernel_shape) == 3:
        #     seq.append(torch.nn.BatchNorm3d())
    elif node.op_type == 'Conv':
        dilation = node_get_attribute_ints(node, 'dilations', [])
        stride = node_get_attribute_ints(node, 'strides', [])
        kernel_shape = node_get_attribute_ints(node, 'kernel_shape', [])
        pads = node_get_attribute_ints(node, 'pads', [])
        if len(pads) == 0:
            pads = node_get_attribute_int(node, 'pads', 0)
            pads = [pads, pads, pads, pads]
        group = node_get_attribute_int(node, 'group', 0)
        _w = None
        _b = None
        if len(node.input) > 1:
            for w in weights:
                if w.name == node.input[1]:
                    _w = w
        if len(node.input) > 2:
            for b in weights:
                if b.name == node.input[2]:
                    _b = b
        is_deepwise = _w.dims[1] == 1 and _w.dims[0] // group * group == _w.dims[0]
        if len(kernel_shape) == 1:
            return torch.nn.Conv1d(in_channels=_w.dims[0] if is_deepwise else _w.dims[1],
                                   out_channels=_w.dims[0], kernel_size=_w.dims[2:],
                                   stride=stride, dilation=dilation, padding=pads, groups=group)
        elif len(kernel_shape) == 2:
            # assert len(tensor_shape) == 4
            pads = pads[0:2]
            if trainable and _b is not None:
                return name, output, \
                    prefix + name + ' = torch.nn.Conv2d( ' + \
                    'in_channels=' + str(_w.dims[0] if is_deepwise else _w.dims[1]) + ', ' + \
                    'out_channels=' + str(_w.dims[0]) + ', ' + \
                    'kernel_size=' + str(tuple(_w.dims[2:])) + ', ' + \
                    'stride=' + str(tuple(stride)) + ', ' + \
                    'dilation=' + str(tuple(dilation)) + ', ' + \
                    'padding=' + str(tuple(pads)) + ', ' + \
                    'groups=' + str(group) + ', ' + \
                    'bias=False' \
                    ')' + comment + '\n' + \
                    prefix + name + '_bn = torch.nn.BatchNorm2d(num_features=' + str(_w.dims[0]) + ', affine=True)\n' + \
                    offset + 'with torch.no_grad():\n' + \
                    (nograd + name + '.weight.copy_(torch.from_numpy(numpy.load(\'' + input_files[
                        node.input[1]] + '\')).float())\n'
                     if _w is not None
                     else nograd + name + '.weight.copy_(torch.zeros(' + prefix + name + '.weight.shape))\n') + \
                    (nograd + name + '_bn.bias.copy_(torch.from_numpy(numpy.load(\'' + input_files[
                        node.input[2]] + '\')).float())\n'
                     if _b is not None else '') + '\n', \
                    offset + output + '_bn = self.' + name + '(' + inputs[0] + ')' + post_stats + '\n' + \
                    offset + output + ' = self.' + name + '_bn(' + offset + output + '_bn)' + post_stats
            else:
                return name, output, \
                    prefix + name + ' = torch.nn.Conv2d( ' + \
                    'in_channels=' + str(_w.dims[0] if is_deepwise else _w.dims[1]) + ', ' + \
                    'out_channels=' + str(_w.dims[0]) + ', ' + \
                    'kernel_size=' + str(tuple(_w.dims[2:])) + ', ' + \
                    'stride=' + str(tuple(stride)) + ', ' + \
                    'dilation=' + str(tuple(dilation)) + ', ' + \
                    'padding=' + str(tuple(pads)) + ', ' + \
                    'groups=' + str(group) + ', ' + \
                    'bias=' + ('True' if _b is not None else 'False') + \
                    ')' + comment + '\n' + \
                    offset + 'with torch.no_grad():\n' + \
                    (nograd + name + '.weight.copy_(torch.from_numpy(numpy.load(\'' + input_files[
                        node.input[1]] + '\')).float())\n'
                     if _w is not None
                     else nograd + name + '.weight.copy_(torch.zeros(' + prefix + name + '.weight.shape))\n') + \
                    (nograd + name + '.bias.copy_(torch.from_numpy(numpy.load(\'' + input_files[
                        node.input[2]] + '\')).float())\n'
                     if _b is not None else '') + '\n', \
                    offset + output + ' = self.' + name + '(' + inputs[0] + ')' + post_stats
        elif len(kernel_shape) == 3:
            return torch.nn.Conv3d(in_channels=_w.dims[0] if is_deepwise else _w.dims[1],
                                   out_channels=_w.dims[0], kernel_size=_w.dims[2:],
                                   stride=stride, dilation=dilation, padding=pads, groups=group)
        else:
            assert False
    elif node.op_type == 'MaxPool':
        auto_pad = node_get_attribute_string(node, 'auto_pad', None)
        kernel_shape = node_get_attribute_ints(node, 'kernel_shape', None)
        pads = node_get_attribute_ints(node, 'pads', None)
        strides = node_get_attribute_ints(node, 'strides', None)
        assert kernel_shape is not None
        arg = 'kernel_size=' + str(tuple(kernel_shape))
        if pads is not None:
            if pads[0] == pads[1] and pads[0] == pads[2] and pads[0] == pads[3]:
                arg = '{}, padding={}'.format(arg, pads[0])
            else:
                arg = arg + ', padding=' + str(tuple(pads))
        if strides is not None:
            arg = arg + ', stride=' + str(tuple(strides))
        return name, output, prefix + name + ' = torch.nn.MaxPool2d( ' + arg + ')' + comment, \
                             offset + output + ' = self.' + name + '(' + inputs[0] + ')' + post_stats
    elif node.op_type == 'Pad':
        if not (len(inputs) <= 4 and 1 <= len(node.input) <= 4 and len(input_files) <= 4):
            print(node.op_type, node.name, len(inputs), len(node.input), len(input_files))
            sys.exit(0)
        assert len(inputs) <= 4 and 1 <= len(node.input) <= 4 and len(input_files) <= 4
        mode = node_get_attribute_string(node, 'mode', 'constant')
        pads = node_get_attribute_ints(node, 'pads', None)
        constant_value = node_get_attribute_float(node, 'constant_value', 0.0)
        axes = node_get_attribute_ints(node, 'axes', None)
        if pads is None and len(inputs) == 1 and len(node.input) >= 2:
            print(node.op_type, node.name,
                  'loading pads from {} input name {}'.format(input_files[node.input[1]], node.input[1]))
            pads = np.load(input_files[node.input[1]])
        print(node.op_type, node.name, 'mode={} pads={} input_files={}'.format(mode, pads, input_files))

        return name, output, prefix + name + ' = torch.nn.ConstantPad2d( ' + \
                             'padding=[' + str(pads[2]) + ', ' + str(pads[3]) + ', ' + str(pads[6]) + ', ' + str(
            pads[7]) + '], ' + \
                             'value=' + str(0.0) + ', ' + ')' + comment, \
                             '        ' + output + '=self.' + name + '(' + inputs[0] + ')' + post_stats
    elif node.op_type == 'Resize':
        print(node.op_type, node.name, inputs, node.input)
        assert 1 <= len(inputs) and len(node.input) <= 4
        antialias = node_get_attribute_int(node, 'antialias', 0)
        axes = node_get_attribute_ints(node, 'axes', None)
        coordinate_transformation_mode = node_get_attribute_string(node, 'coordinate_transformation_mode',
                                                                   'half_pixel').decode('utf-8')
        cubic_coeff_a = node_get_attribute_float(node, 'cubic_coeff_a', -0.75)
        exclude_outside = node_get_attribute_int(node, 'exclude_outside', 0)
        extrapolation_value = node_get_attribute_float(node, 'extrapolation_value', 0.0)
        keep_aspect_ratio_policy = node_get_attribute_string(node, 'keep_aspect_ratio_policy', 'stretch')
        mode = node_get_attribute_string(node, 'mode', 'nearest').decode('utf-8')
        nearest_mode = node_get_attribute_string(node, 'nearest_mode', 'round_prefer_floor').decode('utf-8')
        print(node.op_type, node.name, input_files)
        # print(node.op_type, node.name, node.input)
        roi = np.load(input_files[node.input[1]]) if len(node.input) > 1 and node.input[
            1] in input_files.keys() else None
        scale = np.load(input_files[node.input[2]]) if len(node.input) > 2 and node.input[
            2] in input_files.keys() else None
        sizes = np.load(input_files[node.input[3]]) if len(node.input) > 3 and node.input[
            3] in input_files.keys() else None
        # sizes_string = 'tuple({}.numpy().astype(int)[2:])'.format(onnxrpt_name[node.input[3]]) if sizes is None else str(sizes)
        sizes_string = None if sizes is None else  '{}[2:].tolist()'.format(
                onnxrpt_name[node.input[3]]) if sizes is None else np.array2string(sizes[2:], separator=',')

        print(node.op_type, node.name, 'inputs = {} node.input = {}'.format(inputs, node.input))
        print(node.op_type, node.name,
              'mode = {} coordinate_transformation_mode = {}'.format(mode, coordinate_transformation_mode,
                                                                     exclude_outside))
        print(node.op_type, node.name,
              'nearest_mode = {} roi = {} scale={} sizes={}'.format(nearest_mode, roi, scale, sizes))
        print(node.op_type, node.name, 'input_files = {}'.format(input_files))
        print(node.op_type, node.name, 'sizes_string = {}'.format(sizes_string))

        # TODO: Fix generation below ...
        if sizes is not None:
            return name, output, '{} = None {}'.format(prefix_in_init, comment), \
                '{} = torch.nn.functional.interpolate({}, mode=\'{}\', size={})  {}'.format(prefix_in_body, inputs[0], mode,
                                                                                            sizes_string, post_stats)
        elif scale is not None:
            return name, output, '{} = None {}'.format(prefix_in_init, comment), \
                '{} = torch.nn.functional.interpolate({}, mode=\'{}\', scale_factor=({},{}))  {}'.format(prefix_in_body, inputs[0], mode,
                                                                                            scale[2], scale[3], post_stats)
        else:
            assert False
    elif node.op_type == 'Reshape':
        assert len(inputs) <= 2 and len(node.input) == 2 and len(input_files) <= 1
        # (Optional) By default, when any value in the 'shape' input is equal to zero the corresponding dimension
        # value is copied from the input tensor dynamically. allowzero=1 indicates that if any value in the 'shape'
        # input is set to zero, the zero value is honored, similar to NumPy.
        allowzero = node_get_attribute_int(node, 'allowzero', 0)
        init = ''
        body = ''
        shape = None
        for k in input_files.keys():
            # init = 'torch.from_numpy(numpy.load(\'' + input_files[k] + '\'))\n'
            init = 'None'
            body = name
            shape = np.load(input_files[k])
            break
        return name, output, '{} = {} {}'.format(prefix_in_init, init, comment), \
            '{} = torch.reshape({}, shape={}) {}'.format(prefix_in_body, inputs[0], str([x for x in shape]), post_stats)
        # '{} = torch.reshape({}, shape=tuple(self.{}.numpy().astype(int))) {}'.format(prefix_in_body, inputs[0],
        #                                                                             name, post_stats)
    elif node.op_type == 'Relu':
        assert len(inputs) == 1 and len(node.input) == 1 and len(input_files) == 0
        return name, output, '{} = torch.nn.ReLU() {}'.format(prefix_in_init, comment), \
            '{} = self.{}({}) {}'.format(prefix_in_body, name, inputs[0], post_stats)
        # return name, output, prefix + name + '=torch.nn.ReLU(' + ')' + comment, \
        #                      offset + output + '=self.' + name + '(' + inputs[0] + ')' + comment
    elif node.op_type == 'Shape':
        assert len(inputs) == 1 and len(node.input) == 1 and len(input_files) == 0
        start = node_get_attribute_ints(node, 'start', None)
        end = node_get_attribute_ints(node, 'end', None)
        return name, output, '{} = None {}'.format(prefix_in_init, comment), \
            '{} = {}.size()[{}:{}]  {}'.format(prefix_in_body, inputs[0], start if start is not None else '',
                                               end if end is not None else '', post_stats)
    elif node.op_type == 'Slice':
        print(node.op_type, node.name, 'inputs={} node.input={} input_files={}'.format(inputs, node.input, input_files))
        assert len(inputs) == 1 and len(node.input) == 3 and len(input_files) == 2
        starts = np.load(input_files[node.input[1]])[0] if node.input[1] in input_files.keys() else 0
        ends = np.load(input_files[node.input[2]])[0] if node.input[2] in input_files.keys() else 0
        print(node.op_type, node.name, 'start={} end={}', starts, ends)
        arg = ''
        for i in range(starts, ends):
            arg = arg + '{}[{}], '.format(inputs[0], i)
        return name, output, '{} = None {}'.format(prefix_in_init, comment), \
            '{} = torch.tensor([{}])  {}'.format(prefix_in_body, arg, post_stats)
    elif node.op_type == 'Transpose':
        assert len(inputs) == 1 and len(node.input) == 1 and len(input_files) == 0
        perm = node_get_attribute_ints(node, 'perm', None)
        assert perm is not None
        return name, output, '{} = None {}'.format(prefix_in_init, comment), \
            '{} = torch.permute({}, dims={})  {}'.format(prefix_in_body, inputs[0], perm, post_stats)
    elif node.op_type == 'Upsample':
        assert len(inputs) == 1
        mode = node_get_attribute_string(node, 'mode', 'nearest')
        if len(node.input) == 1 and len(input_files) == 0:
            height_scale = node_get_attribute_float(node, 'height_scale', 1.0)
            assert height_scale >= 1.0
            mode = node_get_attribute_string(node, 'mode', 'nearest')
            width_scale = node_get_attribute_float(node, 'width_scale', 1.0)
            assert width_scale >= 1.0
            if width_scale == height_scale:
                arg = 'scale_factor={}'.format(width_scale * 1.0)
            else:
                arg = 'size=[int({}.size()[-2] * {}), int({}.size()[-1] * {})]'.format(inputs[0],
                                                                                       width_scale, inputs[0],
                                                                                       height_scale)
        elif len(node.input) > 1 and len(input_files) == 1:
            scale: numpy.ndarray = np.load(input_files[node.input[1]]) if node.input[1] in input_files.keys() else None
            # arg = 'scale_factor={}'.format(scale[2:]).replace('. ', ', ').replace('.]', ']')
            if len(scale) == 3 or (len(scale) == 4 and (scale[-1] == scale[-2])):
                arg = 'scale_factor={}'.format(scale[-1] * 1.0)
            elif len(scale) == 4:
                arg = 'size=[int({}.size()[-2] * {}), int({}.size()[-1] * {})]'.format(inputs[0], scale[-2], inputs[0],
                                                                                       scale[-1])
            else:
                assert False
        else:
            print(node.op_type, node.name, node.input, input_files)
            assert False
        return name, output, '{} = None {}'.format(prefix_in_init, comment), \
            '{} = torch.nn.functional.interpolate({}, mode=\'{}\', {})  {}'.format(prefix_in_body, inputs[0],
                                                                                mode.decode('utf-8'),
                                                                                arg, post_stats)
    elif node.op_type == 'Identity':
        assert len(inputs) == 1
        assert len(node.input) == 1
        assert len(input_files) <= 1
        arg = ''
        for k in input_files.keys():
            arg = 'torch.from_numpy(numpy.load(\'' + input_files[k] + '))\'\n'
            break
        return name, output, '{} = torch.nn.Identity({}) {}'.format(prefix_in_init, arg, comment), \
            '{} = self.{}({}) {}'.format(prefix_in_body, name, inputs[0], post_stats)
    elif node.op_type == 'GlobalAveragePool':
        assert len(inputs) == 1 and len(node.input) == 1 and len(input_files) == 0
        return name, output, '{} = torch.nn.AdaptiveAvgPool2d(1) {}'.format(prefix_in_init, comment), \
            '{} = self.{}({}) {}'.format(prefix_in_body, name, inputs[0], post_stats)
    elif node.op_type == 'Flatten':
        assert len(inputs) == 1 and len(node.input) == 1 and len(input_files) == 0
        axis = node_get_attribute_int(node, 'axis', 0)
        return name, output, '{} = torch.nn.Flatten({}) {}'.format(prefix_in_init, axis, comment), \
            '{} = self.{}({}) {}'.format(prefix_in_body, name, inputs[0], post_stats)
    elif node.op_type == 'Gemm':
        print(node.op_type, node.name, node.input, inputs, input_files.keys())
        assert len(node.input) == 3 and len(inputs) + len(input_files.keys()) == 3
        alpha = node_get_attribute_float(node, 'alpha', 0)
        beta = node_get_attribute_float(node, 'beta', 0)
        transA = node_get_attribute_int(node, 'transA', 0)
        transB = node_get_attribute_int(node, 'transB', 0)
        print('GEMM operation is not finished', node.input)
        input_idx = []
        for i in node.input:
            input_idx.append(i)
        return name, output, \
            '{} = Gemm({}, {}, {}, {}, {}, {}, {})'.format(self_name,
                                                           alpha, beta, transA, transB,
                                                           'None' if input_idx[
                                                                         0] not in input_files.keys() else 'numpy.load(\'{}\')'.format(
                                                               input_files[input_idx[0]]),
                                                           'None' if input_idx[
                                                                         1] not in input_files.keys() else 'numpy.load(\'{}\')'.format(
                                                               input_files[input_idx[1]]),
                                                           'None' if input_idx[
                                                                         2] not in input_files.keys() else 'numpy.load(\'{}\')'.format(
                                                               input_files[input_idx[2]]),
                                                           ) + comment, \
            offset + output + '=self.' + name + '(' + str(inputs).replace('\'', '') + ')' + post_stats
    else:
        print(node.name, node.op_type)
        assert False
    print('onnx_node_to_torch out')


def process_nodes(md5: str, inputs: List[str], inputs_shapes: List[List[int]], outputs: List[str],
                  nodes: Dict[str, onnx.NodeProto],
                  weights: onnx.GraphProto) -> str:
    counter = 0
    init = ''
    forward = ''
    onnx2torch_name: Dict[str, str] = {}
    forward_args = ''
    forward_rets = ''
    print('process_node inputs', inputs)
    for i in range(0, len(inputs)):
        print(i)
        onnx2torch_name[inputs[i]] = 'input_var_' + str(counter)
        forward_args = forward_args + onnx2torch_name[inputs[i]] + ', # ' + str(inputs_shapes[i]) + ',\n'
        print('process_node input \'', onnx2torch_name[inputs[i]], '\'', inputs_shapes[i])
    while True:
        node = None
        for n in nodes.keys():
            found = True
            if nodes[n] is None:
                continue
            for i in nodes[n].input:
                if is_real_input(nodes[n], weights, i) and i not in onnx2torch_name.keys():
                    found = False
                    break
            if found:
                node = nodes[n]
                break
        if node is None:
            print('process_node no node found')
            break

        inputs = []
        for i in node.input:
            if is_real_input(node, weights, i):
                inputs.append(onnx2torch_name[i])
        print(node.op_type, node.name, 'start processing')
        torch_node_name, torch_node_output, torch_node_init, torch_node_forward = \
            onnx_node_to_torch(md5, node, inputs, counter, weights, onnx2torch_name)
        # print('done ', node.name)
        init = init + torch_node_init + '\n'
        forward = forward + torch_node_forward + '\n'
        onnx2torch_name[node.name] = torch_node_name
        onnx2torch_name[node.output[0]] = torch_node_output
        counter = counter + 1
        for o in node.output:
            if o in outputs:
                forward_rets = forward_rets + onnx2torch_name[o] + ','
        # if node.name in outputs:
        #     for o in node.output:
        #         if o not in input2nodes.keys():
        #             forward_rets = forward_rets + onnx2torch_name[o] + ','
        print(node.op_type, node.name, 'finish processing')
        del nodes[node.name]
        if len(nodes) == 0:
            break
    assert len(nodes) == 0
    forward_rets = '        return ' + forward_rets
    with open(md5 + '.py', 'w') as f:
        f.write('import numpy\n')
        f.write('import torch\n')
        #  f.write('from onnx_to_pytorch_lib import *\n')
        f.write('\n')
        f.write('class network(torch.nn.Module):\n')
        f.write('    def __init__(self):\n')
        f.write('        super(network, self).__init__()\n')
        f.write(init)
        f.write('\n')
        f.write('\n')
        f.write('    def forward(self,\n        ' + forward_args + '    ):\n')
        f.write(forward)
        f.write(forward_rets)
        # f.write(
        #     '\n\n    def compute_post_stats(self, op_type: str, op_name: str, op, tensor_name: str, tensor_data):\n')
        # f.write('        pass\n\n')
        # f.write('        if isinstance(tensor_data, torch.Tensor):\n')
        # f.write('            print(\'Tensor\', tensor_name, numpy.prod(tensor_data.shape), tensor_data.shape)\n\n')
    return


def add_node_to_input(inputs2nodes: Dict[str, List[onnx.NodeProto]], name: str, node: onnx.NodeProto) \
        -> Dict[str, List[onnx.NodeProto]]:
    inputs2nodes.setdefault(name, []).append(node)
    return inputs2nodes
    # if name in inputs2nodes.keys() and len(inputs2nodes[name]):
    #     inputs2nodes[name].append(node)
    # else:
    #     inputs2nodes[name] = [node]
    # return inputs2nodes


def onnx_file_convert(filename: str):
    update = True
    md5 = hashlib.md5(open(filename, 'rb').read()).hexdigest()
    md5 = md5 + '_' + filename.split('/')[-1].split('.')[0]
    print('md5', md5)
    if update:
        if os.path.exists(md5):
            for f in glob.glob(os.path.join(md5, '*')):
                os.remove(f)
        else:
            os.mkdir(md5)
    inputs = []
    outputs = []
    nodes: Dict[str, onnx.NodeProto] = {}
    inputs2nodes: Dict[str, List[onnx.NodeProto]] = {}
    onnx_model = onnx.load(filename)
    print('version', onnx_model.model_version)
    print('is initialized', onnx_model.IsInitialized())
    onnx_input, onnx_output, _ = get_input_and_output_sizes(filename)
    for node in onnx_model.graph.node:
        nodes[node.name] = node
        if len(node.input) == 0:
            add_node_to_input(inputs2nodes, '', node)  # No input nodes, some constants for example
        else:
            for i in node.input:
                add_node_to_input(inputs2nodes, i, node)
    for i in onnx_input:
        inputs.append(i[0])
    for i in onnx_output:
        outputs.append(i[0])
    print('model inputs:', inputs, 'model outputs:', outputs)
    if update:
        process_nodes(md5, inputs, [onnx_input[0][1]], outputs, nodes, onnx_model.graph.initializer)
    # from converted_net import network
    sys.path.insert(0, os.path.curdir)
    network = importlib.import_module(md5).network
    model = network()
    model.eval()
    model_input = onnx_input[0][1]
    print('model input:', model_input)
    model(torch.rand(model_input))
    op=11
    torch.onnx.export(model=model.cpu(), args=torch.rand(model_input), f='{}_{}_infer.onnx'.format(md5, op),
                      verbose=False, do_constant_folding=False, opset_version=op, input_names=["input"], output_names=["semantic","offset_m","center_h"])
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # torch.save(model, '{}.pt'.format(md5))
    print(torch.jit.script(model))
    sys.exit(0)
    torch.jit.save(torch.jit.script(model), '{}_jit_eval.pt'.format(md5))
    sys.exit(0)
    # model_scripted = torch.jit.script(model)  # Export to TorchScript
    # model_scripted.save('model_jit.pt')  # Save
    ptflops_input = tuple(onnx_input[0][1][1:])
    print('ptflops input:', ptflops_input)
    complexity = ptflops.get_model_complexity_info(model=model, input_res=ptflops_input, print_per_layer_stat=True)
    print(complexity)
    model.train()
    torch.jit.save(torch.jit.trace(model), '{}_jit_train.pt'.format(md5))
    # for op in range(9, 18):
    #     torch.onnx.export(model=model.cpu(), args=torch.rand(model_input), f='{}_{}_train.onnx'.format(md5, op),
    #                       verbose=False, do_constant_folding=True, opset_version=op)
    return


def onnx_dir_convert(pathname):
    for f in sorted(os.listdir(pathname)):
        if f.lower().endswith('.onnx'):
            try:
                onnx_file_convert(os.path.join(pathname, f))
            except:
                print('Failed to rename:', f)
            finally:
                pass
    return


if __name__ == "__main__":
    if sys.argv[1] is not None and os.path.exists(sys.argv[1]) and os.path.isdir(sys.argv[1]):
        onnx_dir_convert(sys.argv[1])
    else:
        onnx_file_convert(sys.argv[1])
    sys.exit(0)
