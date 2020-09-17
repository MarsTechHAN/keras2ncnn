import sys
import os
import math
import argparse
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
            original_keras_version = self.f['model_weights']\
                .attrs['keras_version'].decode('utf8')
            return original_keras_version
        else:
            return '1'

    def get_backend_version(self):
        if 'backend' in self.f['model_weights'].attrs:
            original_backend = self.f['model_weights']\
                .attrs['backend'].decode('utf8')
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
            self.parse_model_graph(
                self.get_model_config()['config']['layers'],
                graph_helper)

    def parse_sequential_graph(self, graph_helper):
        self.joined_layers = []
        for layers in self.model_config['config']['layers']:
            if layers['class_name'] == 'Model':
                self.parse_model_graph(
                    layers['config']['layers'], graph_helper)
            else:
                if layers['class_name'] + '_helper' in dir(KerasParser):
                    tails = graph_helper.get_graph_tail()
                    if len(tails) != 1:
                        raise NotImplementedError
                    else:
                        graph_helper.node(layers['config']['name'], tails)
                        graph_helper.set_node_attr(
                            layers['config']['name'], layers)
                else:
                    raise NotImplementedError

    def parse_model_graph(self, model_layers, graph_helper):
        for layer in model_layers:
            inbound_nodes = self.join_inbound_nodes(layer)

            graph_helper.node(layer['name'], inbound_nodes)
            graph_helper.set_node_attr(
                layer['name'], {
                    'layer': layer, 'weight': self.find_weights_root(
                        layer['name'])})


class Grapher:
    def __init__(self):
        self.graph = {}

    def node(self, name, inbound_nodes=None):
        self.graph[name] = {}
        if inbound_nodes is not None:
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
            if 'inbounds' not in value.keys()\
                    or len(value['inbounds']) == 0:
                self.heads.append(key)
        return self.heads

    def get_graph_tail(self):
        self.tails = []
        for (key, value) in self.graph.items():
            if 'outbounds' not in value.keys()\
                    or len(value['outbounds']) == 0:
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

    def InputLayer_helper(self, layer, keras_graph_helper,
                          ncnn_graph_helper, ncnn_helper):
        input_w = layer['layer']['config']['batch_input_shape'][1]
        input_h = layer['layer']['config']['batch_input_shape'][2]
        input_c = layer['layer']['config']['batch_input_shape'][3]

        ncnn_graph_attr = ncnn_helper.dump_args(
            'Input', w=input_w, h=input_h, c=input_c)

        ncnn_graph_helper.node(
            layer['layer']['name'],
            keras_graph_helper.get_node_inbounds(
                layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(
            layer['layer']['name'], {
                'type': 'Input', 'param': ncnn_graph_attr, 'binary': []})

    def Conv2D_helper(self, layer, keras_graph_helper,
                      ncnn_graph_helper, ncnn_helper):
        # Reorder weight, h-w-i-o to o-i-h-w
        weight = np.insert(np.transpose(layer['weight'],
                                        [3, 2, 0, 1]).flatten(), 0, 0)
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
        if bias_term:
            raise NotImplementedError

        weight_data_size = int(layer['weight'].size)
        ncnn_graph_attr = ncnn_helper.dump_args(
            'Convolution',
            num_output=num_output,
            kernel_w=kernel_w,
            dilation_w=dilation_w,
            stride_w=stride_w,
            pad_left=pad_left,
            bias_term=bias_term,
            weight_data_size=weight_data_size,
            kernel_h=kernel_h,
            dilation_h=dilation_h,
            stride_h=stride_h)

        ncnn_graph_helper.node(
            layer['layer']['name'],
            keras_graph_helper.get_node_inbounds(
                layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(
            layer['layer']['name'], {
                'type': 'Convolution', 'param': ncnn_graph_attr, 'binary': [weight]})

    def DepthwiseConv2D_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):
        # Reorder weight, h-w-i-o to o-i-h-w
        weight = np.insert(
            np.transpose(
                layer['weight'], [
                    3, 2, 0, 1]).flatten(), 0, 0)

        num_output = layer['weight'].shape[2] * \
            layer['layer']['config']['depth_multiplier']
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
        if bias_term:
            raise NotImplementedError

        weight_data_size = int(layer['weight'].size)

        ncnn_graph_attr = ncnn_helper.dump_args(
            'ConvolutionDepthWise',
            num_output=num_output,
            kernel_w=kernel_w,
            dilation_w=dilation_w,
            stride_w=stride_w,
            pad_left=pad_left,
            bias_term=bias_term,
            weight_data_size=weight_data_size,
            group=group,
            kernel_h=kernel_h,
            dilation_h=dilation_h,
            stride_h=stride_h)

        ncnn_graph_helper.node(
            layer['layer']['name'],
            keras_graph_helper.get_node_inbounds(
                layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(
            layer['layer']['name'], {
                'type': 'ConvolutionDepthWise', 'param': ncnn_graph_attr, 'binary': [weight]})

    def BatchNormalization_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):
        num_output = layer['weight']['beta:0'].shape[0]
        bn_eps = layer['layer']['config']['epsilon']

        bn_params = {}
        bn_params['bn_beta'] = np.full([num_output, ], 0, dtype=np.float)
        bn_params['bn_gamma'] = np.full([num_output, ], 1, dtype=np.float)
        bn_params['bn_moving_mean'] = np.full(
            [num_output, ], 0, dtype=np.float)
        bn_params['bn_moving_variance'] = np.full(
            [num_output, ], 1, dtype=np.float)

        for weight_name in layer['weight'].keys():
            bn_params['bn_' +
                      weight_name.replace(':0', '')] = layer['weight'][weight_name]

        ncnn_graph_attr = ncnn_helper.dump_args(
            'BatchNorm', channels=num_output, eps=bn_eps)

        ncnn_graph_helper.node(
            layer['layer']['name'],
            keras_graph_helper.get_node_inbounds(
                layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(
            layer['layer']['name'],
            {
                'type': 'BatchNorm',
                'param': ncnn_graph_attr,
                'binary': [
                    bn_params['bn_gamma'],
                    bn_params['bn_moving_mean'],
                    bn_params['bn_moving_variance'],
                    bn_params['bn_beta']]})

    def Add_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):
        ncnn_graph_attr = ncnn_helper.dump_args('Eltwise', op_type=1)

        ncnn_graph_helper.node(
            layer['layer']['name'],
            keras_graph_helper.get_node_inbounds(
                layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(
            layer['layer']['name'], {
                'type': 'Eltwise', 'param': ncnn_graph_attr, 'binary': []})

    def ZeroPadding2D_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):
        padding_top = layer['layer']['config']['padding'][0][0]
        padding_bottom = layer['layer']['config']['padding'][0][1]
        padding_left = layer['layer']['config']['padding'][1][0]
        padding_right = layer['layer']['config']['padding'][1][1]

        ncnn_graph_attr = ncnn_helper.dump_args(
            'Padding',
            top=padding_top,
            bottom=padding_bottom,
            left=padding_left,
            right=padding_right)

        ncnn_graph_helper.node(
            layer['layer']['name'],
            keras_graph_helper.get_node_inbounds(
                layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(
            layer['layer']['name'], {
                'type': 'Padding', 'param': ncnn_graph_attr, 'binary': []})

    def ReLU_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):
        if layer['layer']['config']['threshold'] != 0:
            raise NotImplementedError

        if 'max_value' in layer['layer']['config'].keys():
            ncnn_graph_attr = ncnn_helper.dump_args(
                'Clip', max=layer['layer']['config']['max_value'])
            ncnn_graph_helper.node(
                layer['layer']['name'] + '_Clip',
                keras_graph_helper.get_node_inbounds(
                    layer['layer']['name']))
            ncnn_graph_helper.set_node_attr(
                layer['layer']['name'] + '_Clip',
                {
                    'type': 'Clip',
                    'param': ncnn_graph_attr,
                    'binary': [],
                    'output_blobs': layer['layer']['name'] + '_Clip_blob'})

            ncnn_graph_attr = ncnn_helper.dump_args(
                'ReLU', slope=layer['layer']['config']['negative_slope'])
            ncnn_graph_helper.node(
                layer['layer']['name'], [
                    layer['layer']['name'] + '_Clip', ])
            ncnn_graph_helper.set_node_attr(
                layer['layer']['name'], {
                    'type': 'ReLU', 'param': ncnn_graph_attr, 'binary': []})
        else:
            ncnn_graph_attr = ncnn_helper.dump_args(
                'ReLU', slope=layer['layer']['config']['negative_slope'])
            ncnn_graph_helper.node(
                layer['layer']['name'],
                keras_graph_helper.get_node_inbounds(
                    layer['layer']['name']))
            ncnn_graph_helper.set_node_attr(
                layer['layer']['name'], {
                    'type': 'ReLU', 'param': ncnn_graph_attr, 'binary': []})

    def Dense_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):

        SUPPORTED_ACTIVATION = ['', 'softmax']
        if layer['layer']['config']['activation'] not in SUPPORTED_ACTIVATION:
            print(layer)
            raise NotImplementedError

        num_output = layer['weight']['kernel:0'].shape[1]

        bn_params = {}
        for weight_name in layer['weight'].keys():
            bn_params['bn_' +
                      weight_name.replace(':0', '')] = layer['weight'][weight_name]
        bn_params['bn_kernel'] = np.transpose(bn_params['bn_kernel'])
        weight_data_size = int(bn_params['bn_kernel'].size)

        bn_params['bn_bias'] = np.asarray(bn_params['bn_bias'])
        bn_params['bn_kernel'] = np.insert(
            bn_params['bn_kernel'].flatten(), 0, 0)

        if layer['layer']['config']['activation'] == '':
            ncnn_graph_attr = ncnn_helper.dump_args(
                'InnerProduct',
                num_output=num_output,
                bias_term=1,
                weight_data_size=weight_data_size)
            ncnn_graph_helper.node(
                layer['layer']['name'],
                keras_graph_helper.get_node_inbounds(
                    layer['layer']['name']))
            ncnn_graph_helper.set_node_attr(
                layer['layer']['name'], {
                    'type': 'InnerProduct', 'param': ncnn_graph_attr, 'binary': [
                        bn_params['bn_kernel'], bn_params['bn_bias']]})
        else:
            ncnn_graph_attr = ncnn_helper.dump_args(
                'InnerProduct',
                num_output=num_output,
                bias_term=1,
                weight_data_size=weight_data_size)
            ncnn_graph_helper.node(
                layer['layer']['name'],
                keras_graph_helper.get_node_inbounds(
                    layer['layer']['name']))
            ncnn_graph_helper.set_node_attr(
                layer['layer']['name'], {
                    'type': 'InnerProduct', 'param': ncnn_graph_attr, 'binary': [
                        bn_params['bn_kernel'], bn_params['bn_bias']]})

            outbound_layers = []

            for name in keras_graph_helper.get_graph().keys():
                for node in keras_graph_helper.get_graph()[name]['inbounds']:
                    if layer['layer']['name'] == node:
                        outbound_layers.append(name)

            ncnn_graph_attr = ncnn_helper.dump_args('Softmax')
            ncnn_graph_helper.node(
                layer['layer']['name'] + '_Softmax', [layer['layer']['name'], ])
            ncnn_graph_helper.set_node_attr(
                layer['layer']['name'] + '_Softmax', {
                    'type': 'Softmax', 'param': ncnn_graph_attr, 'binary': []})

            keras_graph_helper.node(
                layer['layer']['name'] + '_Softmax', [layer['layer']['name'], ])

            for outbound_layer in outbound_layers:
                keras_graph_helper.remove_node_inbounds(
                    outbound_layer, layer['layer']['name'])
                keras_graph_helper.add_node_inbounds(
                    outbound_layer, layer['layer']['name'] + '_Softmax')

    def Permute_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):
        print(layer)
        raise NotImplementedError

    def Concatenate_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):
        print(layer)
        raise NotImplementedError

    def Dropout_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):
        print(layer)
        raise NotImplementedError

    def GlobalAveragePooling2D_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):
        ncnn_graph_attr = ncnn_helper.dump_args(
            'Pooling', pooling_type=1, global_pooling=1)
        ncnn_graph_helper.node(
            layer['layer']['name'],
            keras_graph_helper.get_node_inbounds(
                layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(
            layer['layer']['name'], {
                'type': 'Pooling', 'param': ncnn_graph_attr, 'binary': []})

    def GlobalMaxPooling2D_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):
        ncnn_graph_attr = ncnn_helper.dump_args(
            'Pooling', pooling_type=0, global_pooling=1)
        ncnn_graph_helper.node(
            layer['layer']['name'],
            keras_graph_helper.get_node_inbounds(
                layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(
            layer['layer']['name'], {
                'type': 'Pooling', 'param': ncnn_graph_attr, 'binary': []})

    def AveragePooling2D_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):
        kernel_w, kernel_w = layer['layer']['config']['kernel_size']

        dilation_w, dilation_h = layer['layer']['config']['dilation_rate']

        stride_w, stride_h = layer['layer']['config']['strides']

        if layer['layer']['config']['padding'] == 'valid':
            pad_left = 0
        elif layer['layer']['config']['padding'] == 'same':
            pad_left = -233
        else:
            raise NotImplementedError

        ncnn_graph_attr = ncnn_helper.dump_args(
            'Pooling',
            pooling_type=1,
            kernel_w=kernel_w,
            dilation_w=dilation_w,
            stride_w=stride_w,
            pad_left=pad_left,
            kernel_h=kernel_h,
            dilation_h=dilation_h,
            stride_h=stride_h)

        ncnn_graph_helper.node(
            layer['layer']['name'],
            keras_graph_helper.get_node_inbounds(
                layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(
            layer['layer']['name'], {
                'type': 'Pooling', 'param': ncnn_graph_attr, 'binary': []})

    def MaxPooling2D_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):
        kernel_w, kernel_w = layer['layer']['config']['kernel_size']

        dilation_w, dilation_h = layer['layer']['config']['dilation_rate']

        stride_w, stride_h = layer['layer']['config']['strides']

        if layer['layer']['config']['padding'] == 'valid':
            pad_left = 0
        elif layer['layer']['config']['padding'] == 'same':
            pad_left = -233
        else:
            raise NotImplementedError

        ncnn_graph_attr = ncnn_helper.dump_args(
            'Pooling',
            pooling_type=0,
            kernel_w=kernel_w,
            dilation_w=dilation_w,
            stride_w=stride_w,
            pad_left=pad_left,
            kernel_h=kernel_h,
            dilation_h=dilation_h,
            stride_h=stride_h)

        ncnn_graph_helper.node(
            layer['layer']['name'],
            keras_graph_helper.get_node_inbounds(
                layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(
            layer['layer']['name'], {
                'type': 'Pooling', 'param': ncnn_graph_attr, 'binary': []})

    def insert_split(
            self,
            layer_name,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):
        outbound_layers = []
        for name in keras_graph_helper.get_graph().keys():
            for node in keras_graph_helper.get_graph()[name]['inbounds']:
                if layer_name == node:
                    outbound_layers.append(name)

        if len(outbound_layers) > 1:

            ncnn_graph_attr = ncnn_helper.dump_args('Split')
            ncnn_graph_helper.node(layer_name + '_Split', [layer_name, ])
            ncnn_graph_helper.set_node_attr(
                layer_name + '_Split', {'type': 'Split', 'param': ncnn_graph_attr, 'binary': []})

            keras_graph_helper.node(layer_name + '_Split', [layer_name, ])

            for outbound_layer in outbound_layers:
                keras_graph_helper.remove_node_inbounds(
                    outbound_layer, layer_name)
                keras_graph_helper.add_node_inbounds(
                    outbound_layer, layer_name + '_Split')

    def parse_keras_graph(
            self,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):
        keras_graph_nodes = list(keras_graph_helper.get_graph().keys())
        for node_name in keras_graph_nodes:
            node_helper_name = keras_graph_helper.get_node_attr(
                node_name)['layer']['class_name'] + '_helper'
            if node_helper_name in dir(self):
                eval(
                    'self.' +
                    node_helper_name)(
                    keras_graph_helper.get_node_attr(node_name),
                    keras_graph_helper,
                    ncnn_graph_helper,
                    ncnn_helper)

                if keras_graph_helper.get_node_attr(
                        node_name)['layer']['class_name'] not in self.MULTI_OUTPUT_OP:
                    self.insert_split(
                        keras_graph_helper.get_node_attr(node_name)['layer']['name'],
                        keras_graph_helper,
                        ncnn_graph_helper,
                        ncnn_helper)
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

        'Convolution': {
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

        'ConvolutionDepthWise': {
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

        'InnerProduct': {
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

        'Sigmoid': {
        },

        'Softmax': {
            0: {'axis': 0},
        },

        'Split': {

        },

    }

    def dump_args(self, operator, **kwargs):
        params = self.operation_param_table[operator]
        ncnn_args_phrase = ''
        for arg in params.keys():
            arg_name = list(params[arg].keys())[0]
            if arg_name in kwargs:
                params[arg][arg_name] = kwargs[arg_name]

            params_arg = params[arg][arg_name]

            if isinstance(params_arg, str):
                ncnn_args_phrase = ncnn_args_phrase + \
                    '%d=%s ' % (arg, params_arg)

            elif isinstance(params_arg, int):
                ncnn_args_phrase = ncnn_args_phrase + \
                    '%d=%d ' % (arg, params_arg)

            elif isinstance(params_arg, float):
                ncnn_args_phrase = ncnn_args_phrase + \
                    '%d=%e ' % (arg, params_arg)

            elif isinstance(params_arg, list):
                ncnn_args_phrase = ncnn_args_phrase + \
                    '%d=%d,%s ' % (-23300 - arg, len(params_arg),
                                   ','.join(list(map(str, params_arg))))
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
                # print(in_node)
                if len(self.ncnn_graph.get_node_outbounds(in_node)) > 1:
                    input_blobs.append(
                        '%s_blob_idx_%d' %
                        (in_node, self.ncnn_graph.get_node_outbounds(in_node).index(layer_name)))
                else:
                    input_blobs.append('%s_blob' % in_node)

            output_blobs = []
            if output_count > 1:
                for i in range(output_count):
                    output_blobs.append('%s_blob_idx_%d' % (layer_name, i))
            else:
                output_blobs.append('%s_blob' % layer_name)

            ncnn_param_file.write(
                ('%s' + (
                    25 - len(layer_type)) * ' ' + '%s' + (
                    40 - len(layer_name)) * ' ' + '%d %d %s %s %s\n') % (layer_type,
                                                                         layer_name,
                                                                         input_count,
                                                                         output_count,
                                                                         ' '.join(input_blobs),
                                                                         ' '.join(output_blobs),
                                                                         self.ncnn_graph.get_node_attr(layer_name)['param']))

    def emit_binary(self, file_name):
        output_blob = b''
        for layer_name in self.ncnn_graph.get_graph().keys():
            is_weight_blob = 1
            for weight in self.ncnn_graph.get_node_attr(layer_name)['binary']:
                # bitstruct.pack('f32' * len(weight), *weight)
                output_blob = output_blob + \
                    np.asarray(weight, dtype=np.float32).tobytes()

        open(file_name, 'w+b').write(output_blob)


class KerasDebugger:
    payload = '1f8b0800a916625f02ff85536b6bdb3014fdee5f71e7a6c3493cb7c947a70dac'\
        '9075813694ac6583361855966381253bb2f2e846fffbae643b71b24185d1eb9e'\
        '7b7cee43675cd26c1d337025d341ea3a67cdc515c996b9e23a15e3d6655e3049'\
        '37c30b9a2b66a7202d8aff0152be4c976bdeaca7b052c73c0fd2b1e36c721e43'\
        'bc16452488f6682e4b0d924a1986f7447f06e143754753a2a027224904eb3a7f'\
        '1cc051282e75e2b92f726cc779ecd75ff9225d1f4440cd949a698b53e53cb2be'\
        '49aec04377585d5f8e607585585cfafdaeb556fc66547f4fb29ce81e145ac1b5'\
        'a14d89942cf35635d911e19b217c43c2149786f098f408bf33f81de2b7b8b4f1'\
        'fffab4833e4f00434445cfbb454b8619ef472723ba6f546f0fa80af17e92c52b'\
        '3b3075c8f7ee38a5269a53301a63a619d591294c849d52d7896eea2abd2e5553'\
        '93aa76335655d18047cede608e01a6328e0aa288f0dcceb7e9dd64f6f57ed209'\
        'ec8d5b47720c16798cc96e835fb915d9fa23ea40a5589dfd310c13958ba8e03b'\
        '9695916225ffcd3c541ac44413bf8d7b98fe9adc4537b7f3e1fcf6c637d10434'\
        'cfca6aa7f22dee3ad3d9c3d363f4bdb3dffeecb4b486e164a715a11a8bca76b5'\
        '0a1b00558c6816b1c6ec35b291e6c764fed8a94e8ae9b5927069136f322e0897'\
        'b63d885a62179bf6eff5cc61d3649a27e019237cba86e169db267555f1a131a5'\
        '7c709f4ab264219c97f0cc056e0ba2d3857d2586f3f9b2dd43b5982f8351dd26'\
        'cee1255821b0a7c050adff60518755f70408b4983d179880d8db3bf830a8ff64'\
        'f48b808942bf79dd0ff51fb84c0c09e1198badfc3df387019c36b1684ad14efe'\
        '5fd479cbba12050000'

    target_operators = [
        'Convolution',
        'Conv2D',

        'ConvolutionDepthWise',
        'DepthwiseConv2D',

        'BatchNorm',
        'BatchNormalization',

        'ReLU',
        'Softmax',

        'InnerProduct',
        'Dense']

    input_extractor_template = 'ex.input("$layer_name$_blob", in);'
    output_extractor_template = '    ncnn::Mat $layer_name$; '\
                                'ex.extract("$layer_name$_blob", $layer_name$); '\
                                'dump_mat($layer_name$, "$layer_name$");'

    def dump2c(self, file_name, ncnn_graph):
        import gzip
        import binascii  # pylint: disable=import-outside-toplevel
        c_payload = gzip.decompress(
            binascii.unhexlify(
                self.payload)).decode('utf-8')

        extractor_list = []

        for layer_name in ncnn_graph.get_graph().keys():
            layer_type = ncnn_graph.get_node_attr(layer_name)['type']
            if layer_type == 'Input':
                extractor_list.append(
                    self.input_extractor_template.replace(
                        '$layer_name$', layer_name))
                # extractor_list.append(
                #    self.output_extractor_template.replace(
                #        '$layer_name$', layer_name))

            if layer_type in self.target_operators:
                extractor_list.append(
                    self.output_extractor_template.replace(
                        '$layer_name$', layer_name))

        c_payload = c_payload.replace('$FILENAME$', file_name)
        c_payload = c_payload.replace('$INPUT_W$', '224')
        c_payload = c_payload.replace('$INPUT_H$', '224')
        c_payload = c_payload.replace('$INSERT$', '\n'.join(extractor_list))

        open('%s.c' % file_name, 'w+').write(c_payload)

    def decode(self, log_file):
        from tensorflow.python import keras  # pylint: disable=import-outside-toplevel
        from tensorflow.python.keras import backend as K  # pylint: disable=import-outside-toplevel
        K.set_learning_phase(0)

        from PIL import Image  # pylint: disable=import-outside-toplevel
        from matplotlib import pyplot as plt  # pylint: disable=import-outside-toplevel
        from numpy import linalg  # pylint: disable=import-outside-toplevel
        from scipy.spatial import distance  # pylint: disable=import-outside-toplevel

        # Read results from ncnn log file
        lines = open(log_file, 'r').readlines()
        line_idx = 0
        ncnn_layer_dumps = {}

        while True:
            if line_idx >= len(lines):
                break
            if '>>>>>>' in lines[line_idx]:
                layer_config = lines[line_idx].strip(
                    '>>>>>>').strip('\n').split(',')
                ncnn_layer_dumps[layer_config[3]] = np.fromstring(
                    lines[line_idx + 1], dtype=np.float32, sep=' ') .reshape(*list(map(int, layer_config[0:3])))
                line_idx = line_idx + 2
            else:
                line_idx = line_idx + 1

        # Inference using keras
        model = keras.models.load_model(sys.argv[1])
        test_img = np.asarray(Image.open("cat.jpg").resize((224, 224)))
        output_node_name = []
        output_nodes = []

        for layer_idx in range(len(model.layers)):
            op_type = str(type(model.layers[layer_idx])).strip(
                '\'>').split('.')[-1]
            if op_type in self.target_operators:
                output_node_name.append(model.layers[layer_idx].name)
                output_nodes.append(model.layers[layer_idx].output)

        functor = K.function([model.input], output_nodes)
        layer_outs = functor([test_img[np.newaxis, ...], 1.])

        keras_layer_dumps = dict(zip(output_node_name, layer_outs))

        for ncnn_layer_name in ncnn_layer_dumps.keys():

            if '_Split' in ncnn_layer_name:
                continue

            if '_Softmax' in ncnn_layer_name:
                layer_name = ncnn_layer_name.strip('_Softmax')
            else:
                if ncnn_layer_name + '_Softmax' in ncnn_layer_dumps.keys():
                    continue
                layer_name = ncnn_layer_name

            print('==================================')

            print(
                'Layer Name: %s, Layer Shape: keras->%s ncnn->%s' %
                (ncnn_layer_name, str(
                    keras_layer_dumps[layer_name].shape), str(
                    ncnn_layer_dumps[ncnn_layer_name].shape)))
            print(
                'Max: \tkeras->%.03f ncnn->%.03f \tMin: keras->%.03f ncnn->%.03f' %
                (keras_layer_dumps[layer_name].flatten().max(),
                 ncnn_layer_dumps[ncnn_layer_name].flatten().max(),
                 keras_layer_dumps[layer_name].flatten().min(),
                 ncnn_layer_dumps[ncnn_layer_name].flatten().min()))
            print(
                'Mean: \tkeras->%.03f ncnn->%.03f \tVar: keras->%.03f ncnn->%.03f' %
                (keras_layer_dumps[layer_name].flatten().mean(),
                 ncnn_layer_dumps[ncnn_layer_name].flatten().mean(),
                 keras_layer_dumps[layer_name].flatten().std(),
                 ncnn_layer_dumps[ncnn_layer_name].flatten().std()))

            if keras_layer_dumps[layer_name][0].ndim == 3:
                print(
                    'Cosine Similarity: %.05f' %
                    distance.cosine(
                        keras_layer_dumps[layer_name][0].transpose(
                            (2, 0, 1)).flatten(), ncnn_layer_dumps[ncnn_layer_name].flatten()))

                print('Keras Feature Map: \t%s' % np.array2string(
                    keras_layer_dumps[layer_name][0][0:10, 0, 0], suppress_small=True, precision=3))
                print('Ncnn Feature Map: \t%s' % np.array2string(
                    ncnn_layer_dumps[ncnn_layer_name][0, 0:10, 0], suppress_small=True, precision=3))

            elif keras_layer_dumps[layer_name][0].ndim == 2:
                print(
                    'Cosine Similarity: %.05f' %
                    distance.cosine(
                        keras_layer_dumps[layer_name][0].transpose(
                            (1, 0)).flatten(), ncnn_layer_dumps[ncnn_layer_name].flatten()))

                print('Keras Feature Map: \t%s' % np.array2string(
                    keras_layer_dumps[layer_name][0][0:10, 0], suppress_small=True, precision=3))
                print('Ncnn Feature Map: \t%s' % np.array2string(
                    ncnn_layer_dumps[ncnn_layer_name][0, 0:10], suppress_small=True, precision=3))

            elif keras_layer_dumps[layer_name][0].ndim == 1\
                    and (ncnn_layer_dumps[ncnn_layer_name].shape[:2] == (1, 1)
                         or ncnn_layer_dumps[ncnn_layer_name].ndim == 1):

                print(
                    'Cosine Similarity: %.05f' %
                    distance.cosine(
                        keras_layer_dumps[layer_name][0].flatten(),
                        ncnn_layer_dumps[ncnn_layer_name].flatten()))

                print('Keras Feature Map: \t%s' % np.array2string(
                    keras_layer_dumps[layer_name][0][0:10], suppress_small=True, precision=3))

                if ncnn_layer_dumps[ncnn_layer_name].ndim == 3:
                    print('Ncnn Feature Map: \t%s' % np.array2string(
                        ncnn_layer_dumps[ncnn_layer_name][0, 0, 0:10], suppress_small=True, precision=3))

                keras_index = keras_layer_dumps[layer_name][0].argsort(
                )[-10:][::-1]
                keras_top_value = keras_layer_dumps[layer_name][0][keras_index]
                keras_topk = dict(zip(keras_index, keras_top_value))
                keras_topk_str = ", ".join(
                    ("%d:%.03f" % i for i in keras_topk.items()))

                if ncnn_layer_dumps[ncnn_layer_name].ndim == 3:
                    ncnn_index = ncnn_layer_dumps[ncnn_layer_name][0, 0].argsort(
                    )[-10:][::-1]
                    ncnn_top_value = ncnn_layer_dumps[ncnn_layer_name][0,
                                                                       0][ncnn_index]
                    ncnn_topk = dict(zip(ncnn_index, ncnn_top_value))

                ncnn_topk_str = ", ".join(
                    ("%d:%.03f" % i for i in ncnn_topk.items()))

                print(
                    'Top-k:\nKeras Top-k: \t%s\nncnn Top-k: \t%s' %
                    (keras_topk_str, ncnn_topk_str))

        # Make fancy plots
        fig = plt.figure()
        fig.tight_layout()

        display_features = min(100, len(ncnn_layer_dumps.keys()))
        columns = math.ceil(math.sqrt(display_features))
        rows = math.ceil(display_features / columns)

        for i in range(1, columns * rows + 1):
            if(i < len(ncnn_layer_dumps.keys())):
                title = list(ncnn_layer_dumps.keys())[i - 1]
                featuremap = ncnn_layer_dumps[title]

                fig.add_subplot(rows, columns, i).set_title(title)
                plt.imshow(featuremap[0])
            else:
                break

        plt.draw()

        fig = plt.figure()
        fig.tight_layout()

        columns = math.ceil(math.sqrt(display_features))
        rows = math.ceil(display_features / columns)

        for i in range(1, columns * rows + 1):
            if i < len(layer_outs):
                title = output_node_name[i - 1]
                featuremap = layer_outs[i - 1][0]

                fig.add_subplot(rows, columns, i).set_title(title)
                plt.imshow(featuremap[:, :, 0])
            else:
                break

        plt.show()


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

KerasDebugger().dump2c('ncnn_weight', ncnn_graph)

if os.path.exists('cat.log'):
    KerasDebugger().decode('cat.log')
