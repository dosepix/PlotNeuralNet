#!/usr/bin/env python
import sys
import json
import numpy as np
sys.path.append('../')
from pycore.tikzeng import *

interprDict = {'InputLayer': None,
               'Conv1D': to_ConvRelu,
               'Conv2D': to_ConvRelu,
               'AveragePooling1D': None,
               'AveragePooling2D': None,
               'Dense': to_FcRelu} 

def generateArchFromDict(net_info, scale_x, scale_y, scale_fc, offs, min_fc):
    arch = [to_head('..'),
            to_cor(),
            to_begin()]

    last_name = ''
    offset0 = '(0,0,0)'
    offset = '(%f,0,0)' % offs
    for idx, layer in enumerate(net_info['layers']):
        class_name = layer['class_name']
        # Check if input
        if class_name == 'InputLayer':
            # First dimension is batch size, neglect
            input_size = layer['config']['batch_input_shape'][1:-1]
            layer_img = to_Conv('Input', [input_size[1], input_size[0], 1], 1, 
                          offset=offset0, to=offset0,
                          height=input_size[0], depth=input_size[-1]*scale_x, 
                          width=1*scale_y)
            last_name = 'Input'
            arch.append(layer_img)
            continue

        # Check if pooling
        if 'Pooling' in class_name or class_name == 'Conv1D' or class_name == 'Conv2D':
            print(class_name, layer['config'].keys())
            strides = layer['config']['strides']

            if 'Pooling' in class_name:
                pool = layer['config']['pool_size']
            else:
                pool = layer['config']['kernel_size']

            padding = layer['config']['padding']
            input_size = np.ceil(np.asarray(input_size, dtype=float) / np.asarray(strides))
            if padding == 'valid':
                pad = np.zeros(input_size.shape)
                pad[-1] = 1
                input_size = input_size - pad
            input_size = np.asarray(input_size, dtype=int)
            print(pool, strides, input_size)

        if class_name not in interprDict.keys():
            # Unknown layer
            f = None
        else:
            f = interprDict[class_name]
        # Skip layer
        if f is None:
            continue

        name = class_name + '_%d' % idx

        # Create layer in image
        if idx == 0 or not last_name:
            off = offset0
            to = offset0
        else:
            off = offset
            to = '(%s-east)' % last_name

        if class_name == 'Conv1D':
            filters = layer['config']['filters']
            layer_img = f(name, input_size, filters, 
                          offset=off, to=to,
                          height=input_size[-1], depth=input_size[0]*scale_x, width=filters*scale_y)
        elif class_name == 'Conv2D':
            filters = layer['config']['filters']
            print(filters)
            layer_img = f(name, list(reversed(input_size)), filters, 
                          offset=off, to=to,
                          height=input_size[0], depth=input_size[-1]*scale_x, width=filters*scale_y)
        elif class_name == 'Dense':
            units = layer['config']['units']
            depth = units * scale_fc
            depth = np.clip(depth, min_fc, 1.e3)
            layer_img = f(name, units,
                          offset=off, to=to,
                          height=1, depth=depth, width=2)
        else:
            continue
        arch.append(layer_img)

        if last_name:
            if class_name =='Dense':
                conn_img = to_connectionDashed(last_name, name)
            else:
                conn_img = to_connection(last_name, name)
            arch.append(conn_img)

        # Store name for next iteration
        last_name = name

    # End arch
    arch.append(to_end())
    return arch

def main():
    # inFile = '../fluence_config.json'
    # outFile = 'fluence.tex'
    inFile = '../spec_char_config.json'
    outFile = 'spec_char.tex'
    # inFile = '../spec_deconv_config.json'
    # outFile = 'spec_deconv.tex'

    net_info = json.load(open(inFile, 'r'))
    arch = generateArchFromDict(net_info, scale_x=.2, scale_y=.05, scale_fc=.15, offs=1.2, min_fc=3)
    # for a in arch:
    #     print a
    to_generate(arch, outFile)

if __name__ == '__main__':
    main()
