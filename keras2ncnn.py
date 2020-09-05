import sys, os
import logging, argparse
import json

import numpy as np
import h5py

class H5dfParser:
    def __init__(self, h5_file):
        f = h5py.File(h5_file, mode='r')
        self.f = f
        model_config_raw = f.attrs.get('model_config')
        self.model_config = json.loads(model_config_raw.decode('utf-8'))

        self.keras_version = self.get_keras_version()

    def get_h5df_file(self):
        return self.f

    def get_model_config(self):
        return self.model_config

    def get_keras_version(self):       
        if 'keras_version' in self.f['model_weights'].attrs:
            original_keras_version = self.f['model_weights'].attrs['keras_version'].decode('utf8')
            return original_keras_version
        else:
            return '1'

    def get_backend_version(self):
        if 'backend' in self.f['model_weights'].attrs:
            original_backend = self.f['model_weights'].attrs['backend'].decode('utf8')
            return original_backend
        else:
            return None

    def find_weights_root(self, layer_name):
        if self.keras_version != '1':
            layer = self.f['model_weights']
        else:
            layer = self.f

        while True:
            layer = layer[layer_name]
            if (not hasattr(layer, "keys")) or len(layer.keys()) > 1:
                break
            layer_keys = list(layer.keys())
            if len(layer_keys) < 1:
                return None
            else:
                layer_name = list(layer.keys())[0] 

        return layer

    def get_if_sequential(self):
        if self.model_config['class_name'] == 'Sequential':
            return True
        else:
            return False

    def join_inbound_nodes(self, layer):
        inbound_nodes = []
        if 'inbound_nodes' in layer.keys():
            if len(layer['inbound_nodes']) > 0:
                for inbound in layer['inbound_nodes'][0]:
                    inbound_nodes.append(inbound[0])
        return inbound_nodes

    def parse_graph(self, graph_helper):
        if self.get_if_sequential():
            self.parse_sequential_graph(graph_helper)
        else:
            self.parse_model_graph(self.get_model_config()['config']['layers'], graph_helper)

    def parse_sequential_graph(self, graph_helper):        
        self.joined_layers = []
        for layers in self.model_config['config']['layers']:
            if layers['class_name'] == 'Model':
                self.parse_model_graph(layers['config']['layers'], graph_helper)
            else:
                if layers['class_name'] + '_helper' in dir(KerasParser):
                    tails = graph_helper.get_graph_tail() 
                    if len(tails) != 1:
                        raise NotImplementedError
                    else:
                        graph_helper.node(layers['config']['name'], tails)
                        graph_helper.set_node_attr(layers['config']['name'], layers)
                else:
                    raise NotImplementedError

    def parse_model_graph(self, model_layers, graph_helper):
        for layer in model_layers:
            inbound_nodes = self.join_inbound_nodes(layer)

            graph_helper.node(layer['name'], inbound_nodes)
            graph_helper.set_node_attr(layer['name'], {'layer': layer, 
                                'weight': self.find_weights_root(layer['name'])})

class Grapher:
    def __init__(self):
        self.graph = {}
        
    def node(self, name, inbound_nodes=None):
        self.graph[name] = {}
        if inbound_nodes != None:
            self.graph[name]['inbounds'] = inbound_nodes
            for node in inbound_nodes:
                if node not in self.graph.keys():
                    self.graph[node] = {}
                if 'outbounds' not in self.graph[node].keys():
                    self.graph[node]['outbounds'] = []
                self.graph[node]['outbounds'].append(name)
                
    def refresh(self):
        for name in self.graph.keys():
            self.graph[name]['outbounds'] = []

        for name in self.graph.keys():
            for node in self.graph[name]['inbounds']:
                if node not in self.graph.keys():
                    raise NotImplementedError

                if 'outbounds' not in self.graph[node].keys():
                    self.graph[node]['outbounds'] = []

                self.graph[node]['outbounds'].append(name)

    def get_graph(self):
        return self.graph

    def get_node_inbounds(self, name):
        if 'inbounds' in self.graph[name]:
            return self.graph[name]['inbounds']
        else:
            return []

    def get_node_outbounds(self, name):
        if 'outbounds' in self.graph[name]:
            return self.graph[name]['outbounds']
        else:
            return []

    def set_node_inbounds(self, name, inbounds):
        self.graph[name]['inbounds'] = inbounds

    def set_node_outbounds(self, name, outbounds):
        self.graph[name]['outbounds'] = outbounds

    def remove_node_inbounds(self, name, inbound):
        if inbound in self.graph[name]['inbounds']:
            self.graph[name]['inbounds'].remove(inbound)

    def remove_node_outbounds(self, name, outbound):
        if outbound in self.graph[name]['outbound']:
            self.graph[name]['outbounds'].remove(outbound)

    def add_node_inbounds(self, name, inbound):
        self.graph[name]['inbounds'].append(inbound)

    def add_node_outbounds(self, name, outbound):
        self.graph[name]['outbounds'].append(outbound)

    def get_graph_head(self):
        self.heads = []
        for (key, value) in self.graph.items():
            if 'inbounds' not in value.keys() or len(value['inbounds']) == 0:
                self.heads.append(key)
        return self.heads

    def get_graph_tail(self):
        self.tails = []
        for (key, value) in self.graph.items():
            if 'outbounds' not in value.keys() or len(value['outbounds']) == 0:
                self.tails.append(key)
        return self.tails

    def set_node_attr(self, name, attr):
        if name not in self.graph.keys():
            self.graph[name] = {}
        self.graph[name]['attr'] = attr

    def get_node_attr(self, name):
        if name in self.graph.keys():
            return self.graph[name]['attr']
        else:
            return None

    def plot_graphs(self, comment='Network Grapher View'):
        from graphviz import Digraph

        dot = Digraph(comment=comment)
        for (key, value) in self.graph.items():
            dot.node(key, key)
            if 'inbounds' in value.keys():
                for node in value['inbounds']:
                    dot.edge(node, key)
        dot.render('kears2ncnn.gv', view=True)

class KerasParser:
    MULTI_OUTPUT_OP = []

    def InputLayer_helper(self, layer, keras_graph_helper, ncnn_graph_helper, ncnn_helper):
        input_w = layer['layer']['config']['batch_input_shape'][1]
        input_h = layer['layer']['config']['batch_input_shape'][2]
        input_c = layer['layer']['config']['batch_input_shape'][3]

        ncnn_graph_attr = ncnn_helper.dump_args('Input', w=input_w, h=input_h, c=input_c)

        ncnn_graph_helper.node(layer['layer']['name'], keras_graph_helper.get_node_inbounds(layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(layer['layer']['name'], {'type': 'Input', 
                    'param': ncnn_graph_attr, 'binary': []})
       

    def Conv2D_helper(self, layer, keras_graph_helper, ncnn_graph_helper, ncnn_helper):
        # Reshape weight, Thanks github.com/Tencent/ncnn/tools/mlir/mlir2ncnn.cpp
        weight = np.insert(np.transpose(layer['weight'], [3, 2, 0, 1]).flatten(), 0, 0)
        num_output = layer['layer']['config']['filters']
        kernel_w, kernel_h = layer['layer']['config']['kernel_size']
        dilation_w, dilation_h = layer['layer']['config']['dilation_rate']
        stride_w, stride_h = layer['layer']['config']['strides']

        if layer['layer']['config']['padding'] == 'valid':
            pad_left = 0
        elif layer['layer']['config']['padding'] == 'same':
            pad_left = -233
        else:
            raise NotImplementedError
        bias_term = layer['layer']['config']['use_bias']
        if bias_term == True:
            raise NotImplementedError

        weight_data_size = int(layer['weight'].size)
        #print(weight_data_size, layer['weight'].shape, weight.shape)
        ncnn_graph_attr = ncnn_helper.dump_args('Convolution', num_output=num_output, kernel_w=kernel_w, 
            dilation_w=dilation_w, stride_w=stride_w, pad_left=pad_left, bias_term=bias_term, 
            weight_data_size=weight_data_size, kernel_h=kernel_h, dilation_h=dilation_h, stride_h=stride_h)

        ncnn_graph_helper.node(layer['layer']['name'], keras_graph_helper.get_node_inbounds(layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(layer['layer']['name'], {'type': 'Convolution', 
            'param': ncnn_graph_attr, 'binary': [weight]})  
       
    def DepthwiseConv2D_helper(self, layer, keras_graph_helper, ncnn_graph_helper, ncnn_helper):
        # Reshape weight, Thanks github.com/Tencent/ncnn/tools/mlir/mlir2ncnn.cpp
        weight = np.insert(np.transpose(layer['weight'], [3, 2, 0, 1]).flatten(), 0, 0)

        num_output = layer['weight'].shape[2] * layer['layer']['config']['depth_multiplier']
        group = layer['weight'].shape[2]

        kernel_w, kernel_h = layer['layer']['config']['kernel_size']

        dilation_w, dilation_h = layer['layer']['config']['dilation_rate']

        stride_w, stride_h = layer['layer']['config']['strides']

        if layer['layer']['config']['padding'] == 'valid':
            pad_left = 0
        elif layer['layer']['config']['padding'] == 'same':
            pad_left = -233
        else:
            raise NotImplementedError
        bias_term = layer['layer']['config']['use_bias']
        if bias_term == True:
            raise NotImplementedError

        weight_data_size = int(layer['weight'].size)

        ncnn_graph_attr = ncnn_helper.dump_args('ConvolutionDepthWise', num_output=num_output, kernel_w=kernel_w, 
            dilation_w=dilation_w, stride_w=stride_w, pad_left=pad_left, bias_term=bias_term, 
            weight_data_size=weight_data_size, group=group, kernel_h=kernel_h, dilation_h=dilation_h, stride_h=stride_h)

        ncnn_graph_helper.node(layer['layer']['name'], keras_graph_helper.get_node_inbounds(layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(layer['layer']['name'], {'type': 'ConvolutionDepthWise', 
            'param': ncnn_graph_attr, 'binary': [weight]})  
       
    def BatchNormalization_helper(self, layer, keras_graph_helper, ncnn_graph_helper, ncnn_helper):
        num_output = layer['weight']['beta:0'].shape[0]
        bn_eps = layer['layer']['config']['epsilon']

        bn_params = {}
        bn_params['bn_beta'] = np.full([num_output,], 0, dtype=np.float)
        bn_params['bn_gamma'] = np.full([num_output,], 1, dtype=np.float)
        bn_params['bn_moving_mean'] = np.full([num_output,], 0, dtype=np.float)
        bn_params['bn_moving_variance'] = np.full([num_output,], 1, dtype=np.float)

        for weight_name in layer['weight'].keys():
            bn_params['bn_' + weight_name.replace(':0', '')] = layer['weight'][weight_name]

        ncnn_graph_attr = ncnn_helper.dump_args('BatchNorm', channels=num_output, eps=bn_eps)

        ncnn_graph_helper.node(layer['layer']['name'], keras_graph_helper.get_node_inbounds(layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(layer['layer']['name'], {'type': 'BatchNorm', 
            'param': ncnn_graph_attr, 'binary': [bn_params['bn_gamma'], bn_params['bn_moving_mean'],
                bn_params['bn_moving_variance'], bn_params['bn_beta']]})  
       

    def Add_helper(self, layer, keras_graph_helper, ncnn_graph_helper, ncnn_helper):
        ncnn_graph_attr = ncnn_helper.dump_args('Eltwise', op_type=1)

        ncnn_graph_helper.node(layer['layer']['name'], keras_graph_helper.get_node_inbounds(layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(layer['layer']['name'], {'type': 'Eltwise', 
            'param': ncnn_graph_attr, 'binary': []})
       

    def ZeroPadding2D_helper(self, layer, keras_graph_helper, ncnn_graph_helper, ncnn_helper):
        padding_top = layer['layer']['config']['padding'][0][0]
        padding_bottom = layer['layer']['config']['padding'][0][1]
        padding_left = layer['layer']['config']['padding'][1][0]
        padding_right = layer['layer']['config']['padding'][1][1]

        ncnn_graph_attr = ncnn_helper.dump_args('Padding', top=padding_top, bottom=padding_bottom,
                                            left=padding_left, right=padding_right)

        ncnn_graph_helper.node(layer['layer']['name'], keras_graph_helper.get_node_inbounds(layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(layer['layer']['name'], {'type': 'Padding', 
            'param': ncnn_graph_attr, 'binary': []})
       

    def ReLU_helper(self, layer, keras_graph_helper, ncnn_graph_helper, ncnn_helper):
        if layer['layer']['config']['threshold'] != 0:
            raise NotImplementedError

        if 'max_value' in layer['layer']['config'].keys():
            ncnn_graph_attr = ncnn_helper.dump_args('Clip', max = layer['layer']['config']['max_value'])
            ncnn_graph_helper.node(layer['layer']['name']+'_Clip', keras_graph_helper.get_node_inbounds(layer['layer']['name']))
            ncnn_graph_helper.set_node_attr(layer['layer']['name']+'_Clip', {'type': 'Clip', 
                'param': ncnn_graph_attr, 'binary': [], 'output_blobs': layer['layer']['name']+'_Clip_blob'})

            ncnn_graph_attr = ncnn_helper.dump_args('ReLU', slope = layer['layer']['config']['negative_slope'])
            ncnn_graph_helper.node(layer['layer']['name'], [layer['layer']['name']+'_Clip', ])
            ncnn_graph_helper.set_node_attr(layer['layer']['name'], {'type': 'ReLU', 
                'param': ncnn_graph_attr, 'binary': []})
        else:
            ncnn_graph_attr = ncnn_helper.dump_args('ReLU', slope = layer['layer']['config']['negative_slope'])
            ncnn_graph_helper.node(layer['layer']['name'], keras_graph_helper.get_node_inbounds(layer['layer']['name']))
            ncnn_graph_helper.set_node_attr(layer['layer']['name'], {'type': 'ReLU', 
                'param': ncnn_graph_attr, 'binary': []})
           

    def Dense_helper(self, layer, keras_graph_helper, ncnn_graph_helper, ncnn_helper):
        num_output = layer['weight']['kernel:0'].shape[1]

        bn_params = {}
        for weight_name in layer['weight'].keys():
            bn_params['bn_' + weight_name.replace(':0', '')] = layer['weight'][weight_name]
        bn_params['bn_kernel'] = np.transpose(bn_params['bn_kernel'])
        weight_data_size = int(bn_params['bn_kernel'].size)

        bn_params['bn_bias'] = np.asarray(bn_params['bn_bias'])
        bn_params['bn_kernel'] = np.insert(bn_params['bn_kernel'].flatten(), 0, 0)

        ncnn_graph_attr = ncnn_helper.dump_args('InnerProduct', num_output=num_output, bias_term=1,
                                                                weight_data_size = weight_data_size)
        ncnn_graph_helper.node(layer['layer']['name'], keras_graph_helper.get_node_inbounds(layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(layer['layer']['name'], {'type': 'InnerProduct', 
            'param': ncnn_graph_attr, 'binary': [bn_params['bn_kernel'], bn_params['bn_bias']]
            })
       

    def Permute_helper(self, layer, keras_graph_helper, ncnn_graph_helper, ncnn_helper):
        print(layer)
        raise NotImplementedError 

    def Concatenate_helper(self, layer, keras_graph_helper, ncnn_graph_helper, ncnn_helper):
        print(layer)
        raise NotImplementedError 

    def Dropout_helper(self, layer, keras_graph_helper, ncnn_graph_helper, ncnn_helper):
        print(layer)
        raise NotImplementedError 

    def GlobalAveragePooling2D_helper(self, layer, keras_graph_helper, ncnn_graph_helper, ncnn_helper):
        ncnn_graph_attr = ncnn_helper.dump_args('Pooling', pooling_type=1, global_pooling=1)
        ncnn_graph_helper.node(layer['layer']['name'], keras_graph_helper.get_node_inbounds(layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(layer['layer']['name'], {'type': 'Pooling', 
            'param': ncnn_graph_attr, 'binary': []})
       
    def GlobalMaxPooling2D_helper(self, layer, keras_graph_helper, ncnn_graph_helper, ncnn_helper):
        ncnn_graph_attr = ncnn_helper.dump_args('Pooling', pooling_type=0, global_pooling=1)
        ncnn_graph_helper.node(layer['layer']['name'], keras_graph_helper.get_node_inbounds(layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(layer['layer']['name'], {'type': 'Pooling', 
            'param': ncnn_graph_attr, 'binary': []})
       

    def AveragePooling2D_helper(self, layer, keras_graph_helper, ncnn_graph_helper, ncnn_helper):
        kernel_w, kernel_w = layer['layer']['config']['kernel_size']

        dilation_w, dilation_h = layer['layer']['config']['dilation_rate']

        stride_w, stride_h = layer['layer']['config']['strides']

        if layer['layer']['config']['padding'] == 'valid':
            pad_left = 0
        elif layer['layer']['config']['padding'] == 'same':
            pad_left = -233
        else:
            raise NotImplementedError

        ncnn_graph_attr = ncnn_helper.dump_args('Pooling', pooling_type=1, kernel_w=kernel_w, 
            dilation_w=dilation_w, stride_w=stride_w, pad_left=pad_left, kernel_h=kernel_h, 
            dilation_h=dilation_h, stride_h=stride_h)

        ncnn_graph_helper.node(layer['layer']['name'], keras_graph_helper.get_node_inbounds(layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(layer['layer']['name'], {'type': 'Pooling', 
            'param': ncnn_graph_attr, 'binary': []})  
       
    def MaxPooling2D_helper(self, layer, keras_graph_helper, ncnn_graph_helper, ncnn_helper):
        kernel_w, kernel_w = layer['layer']['config']['kernel_size']

        dilation_w, dilation_h = layer['layer']['config']['dilation_rate']

        stride_w, stride_h = layer['layer']['config']['strides']

        if layer['layer']['config']['padding'] == 'valid':
            pad_left = 0
        elif layer['layer']['config']['padding'] == 'same':
            pad_left = -233
        else:
            raise NotImplementedError

        ncnn_graph_attr = ncnn_helper.dump_args('Pooling', pooling_type=0, kernel_w=kernel_w, 
            dilation_w=dilation_w, stride_w=stride_w, pad_left=pad_left, kernel_h=kernel_h, 
            dilation_h=dilation_h, stride_h=stride_h)

        ncnn_graph_helper.node(layer['layer']['name'], keras_graph_helper.get_node_inbounds(layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(layer['layer']['name'], {'type': 'Pooling', 
            'param': ncnn_graph_attr, 'binary': []})  


    def insert_split(self, layer_name, keras_graph_helper, ncnn_graph_helper, ncnn_helper):
        outbound_layers = []
        for name in keras_graph_helper.get_graph().keys():
            for node in keras_graph_helper.get_graph()[name]['inbounds']:
                if layer_name == node:
                    outbound_layers.append(name)

        if len(outbound_layers) > 1:
            
            ncnn_graph_attr = ncnn_helper.dump_args('Split')
            ncnn_graph_helper.node(layer_name+'_Split', [layer_name, ])
            ncnn_graph_helper.set_node_attr(layer_name+'_Split', {'type': 'Split', 
                'param': ncnn_graph_attr, 'binary': []})  
            
            keras_graph_helper.node(layer_name+'_Split', [layer_name, ])

            for outbound_layer in outbound_layers:
                keras_graph_helper.remove_node_inbounds(outbound_layer, layer_name)
                keras_graph_helper.add_node_inbounds(outbound_layer, layer_name+'_Split')

    def parse_keras_graph(self, keras_graph_helper, ncnn_graph_helper, ncnn_helper):
        keras_graph_nodes = list(keras_graph_helper.get_graph().keys())
        for node_name in keras_graph_nodes:
            node_helper_name = keras_graph_helper.get_node_attr(node_name)['layer']['class_name'] + '_helper'
            if node_helper_name in dir(self):
                eval('self.' + node_helper_name)(keras_graph_helper.get_node_attr(node_name), 
                    keras_graph_helper, ncnn_graph_helper, ncnn_helper)

                if keras_graph_helper.get_node_attr(node_name)['layer']['class_name'] not in self.MULTI_OUTPUT_OP:
                    self.insert_split(keras_graph_helper.get_node_attr(node_name)['layer']['name'], 
                        keras_graph_helper, ncnn_graph_helper, ncnn_helper)
            else:
                print(node_helper_name)
                raise NotImplementedError

        keras_graph_helper.refresh()
        ncnn_graph_helper.refresh()

class NcnnParamDispatcher:
    operation_param_table = {
        'BatchNorm': {
            0: {'channels': 0},
            1: {'eps': 0},
        },

        'BinaryOp': {
            0: {'op_type': 0},
            1: {'with_scalar': 0},
            2: {'b': 0},
        },

        'Clip': {  
            0: {'min': -sys.float_info.max},
            1: {'max': sys.float_info.max},
        },

        'Concat': {  
            0: {'axis': 0},
        },

        'Convolution':{
            0: {'num_output': 0},
            1: {'kernel_w': 0},
            2: {'dilation_w': 1},
            3: {'stride_w': 0},
            4: {'pad_left': 0},
            5: {'bias_term': 0},
            6: {'weight_data_size': 0},

            11: {'kernel_h': 0},
            12: {'dilation_h': 1},
            13: {'stride_h': 1},
        },

        'ConvolutionDepthWise':{
            0: {'num_output': 0},
            1: {'kernel_w': 0},
            2: {'dilation_w': 1},
            3: {'stride_w': 0},
            4: {'pad_left': 0},
            5: {'bias_term': 0},
            6: {'weight_data_size': 0},
            7: {'group': 1},

            11: {'kernel_h': 0},
            12: {'dilation_h': 1},
            13: {'stride_h': 1},
        },

        'Eltwise': {  
            0: {'op_type': 0},
            # 1: {'coeffs': []},
        },

        'InnerProduct':{
            0: {'num_output': 0},
            1: {'bias_term': 0},
            2: {'weight_data_size': 0},
        },

        'Input': {  
            0: {'w': 0},
            1: {'h': 0},
            2: {'c': 0},
        },

        'Padding': {  
            0: {'top': 0},
            1: {'bottom': 0},
            2: {'left': 0},
            3: {'right': 0},
        },

        'Pooling': {  
            0: {'pooling_type': 0},
            1: {'kernel_w': 0},
            11: {'kernel_h': 0},
            2: {'stride_w': 1},
            12: {'stride_h': 1},
            3: {'pad_left': 0},
            4: {'global_pooling': 0},    
        },

        'ReLU': {  
            0: {'slope': 0},
            1: {'stride': 0},
        },  

        'Reshape': {
            0: {'w': -233},
            1: {'h': -233},
            2: {'c': -233},
        },

        'Sigmoid':{
        },

        'Softmax':{
            0: {'axis': 0},
        },

        'Split':{

        },

    }

    # [layer type] [layer name] [input count] [output count] [input blobs] [output blobs] [layer specific params]
    # integer array or float array key : -23300 minus index 0 ~ 19 [array size],int,int,...,int
    def dump_args(self, operator, **kwargs):
        params = self.operation_param_table[operator]
        ncnn_args_phrase = ''
        for arg in params.keys():
            arg_name = list(params[arg].keys())[0]
            if arg_name in kwargs:
                params[arg][arg_name] = kwargs[arg_name]

            params_arg = params[arg][arg_name]

            if isinstance(params_arg, str):
                ncnn_args_phrase = ncnn_args_phrase + '%d=%s ' % (arg, params_arg)

            elif isinstance(params_arg, int):
                ncnn_args_phrase = ncnn_args_phrase + '%d=%d ' % (arg, params_arg)
            
            elif isinstance(params_arg, float):
                ncnn_args_phrase = ncnn_args_phrase + '%d=%e ' % (arg, params_arg)

            elif isinstance(params_arg, list):
                ncnn_args_phrase = ncnn_args_phrase + '%d=%d,%s ' % (-23300 - arg, len(params_arg)
                                                                    ,','.join(list(map(str, params_arg))))
            else:
                print(arg_name, params_arg, type(params_arg))
                raise NotImplementedError
        return ncnn_args_phrase

class NcnnEmitter:

    def __init__(self, ncnn_graph):
        self.MAGGGGGIC = 7767517
        self.ncnn_graph = ncnn_graph
    def emit_param(self, file_name):
        ncnn_param_file = open(file_name, 'w+')

        ncnn_param_file.write('%d\n' % self.MAGGGGGIC)

        layer_count = len(ncnn_graph.get_graph())
        blob_count = len(ncnn_graph.get_graph()) * 2
        ncnn_param_file.write('%d %d\n' % (layer_count, blob_count))

        for layer_name in self.ncnn_graph.get_graph().keys():
            layer_type = self.ncnn_graph.get_node_attr(layer_name)['type']
            input_count = len(self.ncnn_graph.get_node_inbounds(layer_name))

            output_count = len(self.ncnn_graph.get_node_outbounds(layer_name))
            output_count = 1 if output_count == 0 else output_count

            input_blobs = []
            inbound_nodes = self.ncnn_graph.get_node_inbounds(layer_name)
            for in_node in inbound_nodes:
                #print(in_node)
                if len(self.ncnn_graph.get_node_outbounds(in_node)) > 1:
                    input_blobs.append('%s_blob_idx_%d' % (in_node, 
                        self.ncnn_graph.get_node_outbounds(in_node).index(layer_name)))
                else:
                    input_blobs.append('%s_blob' % in_node)

            output_blobs = []
            if output_count > 1:
                for i in range(output_count):
                    output_blobs.append('%s_blob_idx_%d' % (layer_name, i))
            else:
                output_blobs.append('%s_blob' % layer_name)

            ncnn_param_file.write(('%s'+(25 - len(layer_type))*' '+'%s'+(40 - len(layer_name))*' '+'%d %d %s %s %s\n') % 
                            (layer_type, layer_name, input_count, output_count, ' '.join(input_blobs), 
                            ' '.join(output_blobs), self.ncnn_graph.get_node_attr(layer_name)['param']))
    def emit_binary(self, file_name):
        output_blob = b''
        for layer_name in self.ncnn_graph.get_graph().keys():
            is_weight_blob = 1
            for weight in self.ncnn_graph.get_node_attr(layer_name)['binary']:
                output_blob = output_blob + np.asarray(weight, dtype=np.float32).tobytes()#bitstruct.pack('f32' * len(weight), *weight)

        open(file_name, 'w+b').write(output_blob)

# Create a source graph and a dest graph
keras_graph = Grapher()
ncnn_graph = Grapher()

# Read and parse keras file to graph
H5dfParser(sys.argv[1]).parse_graph(keras_graph)

# Convert keras to ncnn representations
KerasParser().parse_keras_graph(keras_graph, ncnn_graph, NcnnParamDispatcher())

# keras_graph.plot_graphs()

emitter = NcnnEmitter(ncnn_graph)
emitter.emit_param('ncnn_weight.param')
emitter.emit_binary('ncnn_weight.bin')
# Emit the graph to params and bin

