import inspect
import numpy as np
import sys


class KerasConverter:
    MULTI_OUTPUT_OP = []

    def InputLayer_helper(self, layer, keras_graph_helper,
                          ncnn_graph_helper, ncnn_helper):

        def replaceNone(x): return -1 if x is None else x

        input_w = replaceNone(layer['layer']['config']['batch_input_shape'][1])
        input_h = replaceNone(layer['layer']['config']['batch_input_shape'][2])
        input_c = replaceNone(layer['layer']['config']['batch_input_shape'][3])

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

        CONV2D_SUPPORTED_ACTIVATION = ['softmax']
        CONV2D_FUSED_ACTIVATION_TYPE = {
            '': 0,
            'linear': 0,
            'relu': 1,
            'sigmoid': 4
        }

        num_output = layer['layer']['config']['filters']
        kernel_w, kernel_h = layer['layer']['config']['kernel_size']
        dilation_w, dilation_h = layer['layer']['config']['dilation_rate']
        stride_w, stride_h = layer['layer']['config']['strides']

        if layer['layer']['config']['padding'] == 'valid':
            pad_left = 0
        elif layer['layer']['config']['padding'] == 'same':
            pad_left = -233
        else:
            print('[ERROR] Explicit padding is not supported yet.')
            frameinfo = inspect.getframeinfo(inspect.currentframe())
            print('Failed to convert at %s:%d %s()' %
                  (frameinfo.filename, frameinfo.lineno, frameinfo.function))
            sys.exit(-1)

        bias_term = layer['layer']['config']['use_bias']
        if bias_term:
            weight_data_size = int(layer['weight']['kernel:0'].size)
            # Reorder weight, h-w-i-o to o-i-h-w
            kernel_weight = np.insert(
                np.transpose(
                    layer['weight']['kernel:0'], [
                        3, 2, 0, 1]).flatten(), 0, 0)
            bias_weight = layer['weight']['bias:0']
        else:
            # Reorder weight, h-w-i-o to o-i-h-w
            weight_data_size = int(layer['weight']['kernel:0'].size)
            # Reorder weight, h-w-i-o to o-i-h-w
            weight = np.insert(np.transpose(layer['weight']['kernel:0'],
                                            [3, 2, 0, 1]).flatten(), 0, 0)

        if 'activation' in layer['layer']['config']:
            if layer['layer']['config']['activation'] in CONV2D_FUSED_ACTIVATION_TYPE.keys():
                activation_type = CONV2D_FUSED_ACTIVATION_TYPE[layer['layer'][
                    'config']['activation']]
            else:
                activation_type = -1
        else:
            activation_type = 0

        if activation_type > -1:
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
                stride_h=stride_h,
                activation_type=activation_type)
        
            ncnn_graph_helper.node(
                layer['layer']['name'],
                keras_graph_helper.get_node_inbounds(
                    layer['layer']['name']))

            if bias_term:
                ncnn_graph_helper.set_node_attr(
                    layer['layer']['name'], {
                        'type': 'Convolution', 'param': ncnn_graph_attr, 'binary': [
                            kernel_weight, bias_weight]})
            else:
                ncnn_graph_helper.set_node_attr(
                    layer['layer']['name'], {
                        'type': 'Convolution', 'param': ncnn_graph_attr, 'binary': [weight]})
        else:
            if layer['layer']['config']['activation'] in CONV2D_SUPPORTED_ACTIVATION:
                if layer['layer']['config']['activation'] == 'softmax':
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

                    if bias_term:
                        ncnn_graph_helper.set_node_attr(
                            layer['layer']['name'], {
                                'type': 'Convolution', 'param': ncnn_graph_attr, 'binary': [
                                    kernel_weight, bias_weight]})
                    else:
                        ncnn_graph_helper.set_node_attr(
                            layer['layer']['name'], {
                                'type': 'Convolution', 'param': ncnn_graph_attr, 'binary': [weight]})

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
            else:
                print(
                    '[ERROR] Activation type %s is is not supported yet.' %
                    layer['layer']['config']['activation'])
                frameinfo = inspect.getframeinfo(inspect.currentframe())
                print('Failed to convert at %s:%d %s()' %
                    (frameinfo.filename, frameinfo.lineno, frameinfo.function))
                sys.exit(-1)
                   

    def Conv2DTranspose_helper(self, layer, keras_graph_helper,
                               ncnn_graph_helper, ncnn_helper):

        CONV2D_T_ACTIVATION_TYPE = {
            'linear': 0,
            'relu': 1,
            'sigmoid': 4
        }

        num_output = layer['layer']['config']['filters']
        kernel_w, kernel_h = layer['layer']['config']['kernel_size']
        dilation_w, dilation_h = layer['layer']['config']['dilation_rate']
        stride_w, stride_h = layer['layer']['config']['strides']

        if layer['layer']['config']['padding'] == 'valid':
            print('[WARN] Valid padding is not tested yet.')
            frameinfo = inspect.getframeinfo(inspect.currentframe())
        elif layer['layer']['config']['padding'] == 'same':
            pad_left = kernel_w - stride_w
            pad_top = kernel_h - stride_h
            if pad_left < 0 or pad_top < 0:
                print('[ERROR] Failed to calculate output shape.')
                frameinfo = inspect.getframeinfo(inspect.currentframe())
                print('Failed to convert at %s:%d %s()' %
                      (frameinfo.filename, frameinfo.lineno, frameinfo.function))
                sys.exit(-1)
        else:
            print('[ERROR] Explicit padding is not supported yet.')
            frameinfo = inspect.getframeinfo(inspect.currentframe())
            print('Failed to convert at %s:%d %s()' %
                  (frameinfo.filename, frameinfo.lineno, frameinfo.function))
            sys.exit(-1)

        bias_term = layer['layer']['config']['use_bias']
        if bias_term:
            weight_data_size = int(layer['weight']['kernel:0'].size)
            # Reorder weight, h-w-i-o to o-i-h-w
            kernel_weight = np.insert(
                np.transpose(
                    layer['weight']['kernel:0'], [
                        2, 3, 0, 1]).flatten(), 0, 0)
            bias_weight = layer['weight']['bias:0']
        else:
            # Reorder weight, h-w-i-o to o-i-h-w
            weight_data_size = int(layer['weight']['kernel:0'].size)
            # Reorder weight, h-w-i-o to o-i-h-w
            weight = np.insert(np.transpose(layer['weight']['kernel:0'],
                                            [2, 3, 0, 1]).flatten(), 0, 0)

        if 'activation' in layer['layer']['config']:
            if layer['layer']['config']['activation'] in CONV2D_T_ACTIVATION_TYPE.keys():
                activation_type = CONV2D_T_ACTIVATION_TYPE[layer['layer'][
                    'config']['activation']]
            else:
                print(
                    '[ERROR] Activation type %s is is not supported yet.' %
                    layer['layer']['config']['activation'])
                frameinfo = inspect.getframeinfo(inspect.currentframe())
                print('Failed to convert at %s:%d %s()' %
                      (frameinfo.filename, frameinfo.lineno, frameinfo.function))
                sys.exit(-1)
        else:
            activation_type = 0

        ncnn_graph_attr = ncnn_helper.dump_args(
            'Deconvolution',
            num_output=num_output,
            kernel_w=kernel_w,
            dilation_w=dilation_w,
            stride_w=stride_w,
            pad_left=pad_left,
            pad_top=pad_top,
            bias_term=bias_term,
            weight_data_size=weight_data_size,
            kernel_h=kernel_h,
            dilation_h=dilation_h,
            stride_h=stride_h,
            activation_type=activation_type)

        ncnn_graph_helper.node(
            layer['layer']['name'],
            keras_graph_helper.get_node_inbounds(
                layer['layer']['name']))

        if bias_term:
            ncnn_graph_helper.set_node_attr(
                layer['layer']['name'], {
                    'type': 'Deconvolution', 'param': ncnn_graph_attr, 'binary': [
                        kernel_weight, bias_weight]})
        else:
            ncnn_graph_helper.set_node_attr(
                layer['layer']['name'], {
                    'type': 'Deconvolution', 'param': ncnn_graph_attr, 'binary': [weight]})

    def DepthwiseConv2D_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):
        # Reorder weight, h-w-i-o to o-i-h-w
        weight = np.insert(
            np.transpose(
                layer['weight']['depthwise_kernel:0'], [
                    3, 2, 0, 1]).flatten(), 0, 0)

        num_output = layer['weight']['depthwise_kernel:0'].shape[2] * \
            layer['layer']['config']['depth_multiplier']
        group = layer['weight']['depthwise_kernel:0'].shape[2]

        kernel_w, kernel_h = layer['layer']['config']['kernel_size']

        dilation_w, dilation_h = layer['layer']['config']['dilation_rate']

        stride_w, stride_h = layer['layer']['config']['strides']

        if layer['layer']['config']['padding'] == 'valid':
            pad_left = 0
        elif layer['layer']['config']['padding'] == 'same':
            pad_left = -233
        else:
            print('[ERROR] Explicit padding is not supported yet.')
            frameinfo = inspect.getframeinfo(inspect.currentframe())
            print('Failed to convert at %s:%d %s()' %
                  (frameinfo.filename, frameinfo.lineno, frameinfo.function))
            sys.exit(-1)

        bias_term = layer['layer']['config']['use_bias']

        if bias_term:
            bias_weight = layer['weight']['bias:0']

        weight_data_size = int(layer['weight']['depthwise_kernel:0'].size)

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

        if bias_term:
            ncnn_graph_helper.set_node_attr(
                layer['layer']['name'], {
                    'type': 'ConvolutionDepthWise', 'param': ncnn_graph_attr, 'binary': [
                        weight, bias_weight]})
        else:
            ncnn_graph_helper.set_node_attr(
                layer['layer']['name'], {
                    'type': 'ConvolutionDepthWise', 'param': ncnn_graph_attr, 'binary': [weight]})

    def SeparableConv2D_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):

        SEPCONV2D_ACTIVATION_TYPE = {
            'linear': 0,
            'relu': 1,
            'sigmoid': 4
        }

        # Fetch weight 
        dw_weight = np.insert(
            np.transpose(
                layer['weight']['depthwise_kernel:0'], [
                    3, 2, 0, 1]).flatten(), 0, 0)

        pw_weight = np.insert(
            np.transpose(
                layer['weight']['pointwise_kernel:0'], [
                    3, 2, 0, 1]).flatten(), 0, 0)

        # Insert dwconv
        num_output = layer['weight']['depthwise_kernel:0'].shape[2] * \
            layer['layer']['config']['depth_multiplier']
        group = layer['weight']['depthwise_kernel:0'].shape[2]

        kernel_w, kernel_h = layer['layer']['config']['kernel_size']

        dilation_w, dilation_h = layer['layer']['config']['dilation_rate']

        stride_w, stride_h = layer['layer']['config']['strides']

        if layer['layer']['config']['padding'] == 'valid':
            pad_left = 0
        elif layer['layer']['config']['padding'] == 'same':
            pad_left = -233
        else:
            print('[ERROR] Explicit padding is not supported yet.')
            frameinfo = inspect.getframeinfo(inspect.currentframe())
            print('Failed to convert at %s:%d %s()' %
                  (frameinfo.filename, frameinfo.lineno, frameinfo.function))
            sys.exit(-1)

        weight_data_size = int(layer['weight']['depthwise_kernel:0'].size)

        ncnn_graph_attr = ncnn_helper.dump_args(
            'ConvolutionDepthWise',
            num_output=num_output,
            kernel_w=kernel_w,
            dilation_w=dilation_w,
            stride_w=stride_w,
            pad_left=pad_left,
            weight_data_size=weight_data_size,
            group=group,
            kernel_h=kernel_h,
            dilation_h=dilation_h,
            stride_h=stride_h)

        ncnn_graph_helper.node(
            layer['layer']['name'] + '_dw',
            keras_graph_helper.get_node_inbounds(
                layer['layer']['name']))

        ncnn_graph_helper.set_node_attr(
            layer['layer']['name']  + '_dw', {
                'type': 'ConvolutionDepthWise', 'param': ncnn_graph_attr, 'binary': [dw_weight]})

        # Fill pwconv params
        num_output = layer['layer']['config']['filters']
        bias_term = layer['layer']['config']['use_bias']
        if bias_term:
            bias_weight = layer['weight']['bias:0']

        if 'activation' in layer['layer']['config']:
            if layer['layer']['config']['activation'] in SEPCONV2D_ACTIVATION_TYPE.keys():
                activation_type = SEPCONV2D_ACTIVATION_TYPE[layer['layer'][
                    'config']['activation']]
            else:
                print(
                    '[ERROR] Activation type %s is is not supported yet.' %
                    layer['layer']['config']['activation'])
                frameinfo = inspect.getframeinfo(inspect.currentframe())
                print('Failed to convert at %s:%d %s()' %
                      (frameinfo.filename, frameinfo.lineno, frameinfo.function))
                sys.exit(-1)
        else:
            activation_type = 0

        weight_data_size = int(layer['weight']['pointwise_kernel:0'].size)

        ncnn_graph_attr = ncnn_helper.dump_args(
            'Convolution',
            num_output=num_output,
            kernel_w=1,
            dilation_w=1,
            stride_w=1,
            pad_left=pad_left,
            bias_term=bias_term,
            weight_data_size=weight_data_size,
            kernel_h=1,
            dilation_h=1,
            stride_h=1,
            activation_type=activation_type)

        ncnn_graph_helper.node(
            layer['layer']['name'],
            [layer['layer']['name']+'_dw'])

        if bias_term:
            ncnn_graph_helper.set_node_attr(
                layer['layer']['name'], {
                    'type': 'Convolution', 'param': ncnn_graph_attr, 'binary': [
                        pw_weight, bias_weight]})
        else:
            ncnn_graph_helper.set_node_attr(
                layer['layer']['name'], {
                    'type': 'Convolution', 'param': ncnn_graph_attr, 'binary': [pw_weight]})

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
        ncnn_graph_attr = ncnn_helper.dump_args('BinaryOp', op_type=0, with_scalar=0)

        ncnn_graph_helper.node(
            layer['layer']['name'],
            keras_graph_helper.get_node_inbounds(
                layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(
            layer['layer']['name'], {
                'type': 'BinaryOp', 'param': ncnn_graph_attr, 'binary': []})

    def Multiply_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):
        ncnn_graph_attr = ncnn_helper.dump_args('BinaryOp', op_type=2, with_scalar=0)

        ncnn_graph_helper.node(
            layer['layer']['name'],
            keras_graph_helper.get_node_inbounds(
                layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(
            layer['layer']['name'], {
                'type': 'Eltwise', 'param': ncnn_graph_attr, 'binary': []})

    def Activation_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):

        SUPPORTED_ACTIVATION = ['relu', 'sigmoid', 'softmax']

        if layer['layer']['config']['activation'] not in SUPPORTED_ACTIVATION:
            print(
                '[ERROR] Activation type %s is is not supported yet.' %
                layer['layer']['config']['activation'])
            frameinfo = inspect.getframeinfo(inspect.currentframe())
            print('Failed to convert at %s:%d %s()' %
                  (frameinfo.filename, frameinfo.lineno, frameinfo.function))
            sys.exit(-1)

        if layer['layer']['config']['activation'] == 'relu':
            if 'alpha' in layer['layer']['config'].keys():
                negative_slope = layer['layer']['config']['alpha']
            else:
                negative_slope = 0.0

            if 'max_value' in layer['layer']['config'].keys():
                if layer['layer']['config']['max_value'] is not None:
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
                        'ReLU', slope=negative_slope)
                    ncnn_graph_helper.node(
                        layer['layer']['name'], [
                            layer['layer']['name'] + '_Clip', ])
                    ncnn_graph_helper.set_node_attr(
                        layer['layer']['name'], {
                            'type': 'ReLU', 'param': ncnn_graph_attr, 'binary': []})
                else:
                    ncnn_graph_attr = ncnn_helper.dump_args(
                        'ReLU', slope=negative_slope)
                    ncnn_graph_helper.node(
                        layer['layer']['name'],
                        keras_graph_helper.get_node_inbounds(
                            layer['layer']['name']))
                    ncnn_graph_helper.set_node_attr(
                        layer['layer']['name'], {
                            'type': 'ReLU', 'param': ncnn_graph_attr, 'binary': []})
            else:
                ncnn_graph_attr = ncnn_helper.dump_args(
                    'ReLU', slope=negative_slope)
                ncnn_graph_helper.node(
                    layer['layer']['name'],
                    keras_graph_helper.get_node_inbounds(
                        layer['layer']['name']))
                ncnn_graph_helper.set_node_attr(
                    layer['layer']['name'], {
                        'type': 'ReLU', 'param': ncnn_graph_attr, 'binary': []})
            return

        if layer['layer']['config']['activation'] == 'sigmoid':
            ncnn_graph_attr = ncnn_helper.dump_args(
                'Sigmoid')
            ncnn_graph_helper.node(
                layer['layer']['name'],
                keras_graph_helper.get_node_inbounds(
                    layer['layer']['name']))
            ncnn_graph_helper.set_node_attr(
                layer['layer']['name'], {
                    'type': 'Sigmoid', 'param': ncnn_graph_attr, 'binary': []})

        if layer['layer']['config']['activation'] == 'softmax':
            ncnn_graph_attr = ncnn_helper.dump_args(
                'Softmax')
            ncnn_graph_helper.node(
                layer['layer']['name'],
                keras_graph_helper.get_node_inbounds(
                    layer['layer']['name']))
            ncnn_graph_helper.set_node_attr(
                layer['layer']['name'], {
                    'type': 'Softmax', 'param': ncnn_graph_attr, 'binary': []})

    def Flatten_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):

        ncnn_graph_attr = ncnn_helper.dump_args(
            'Reshape', w=-1)
        ncnn_graph_helper.node(
            layer['layer']['name'],
            keras_graph_helper.get_node_inbounds(
                layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(
            layer['layer']['name'], {
                'type': 'Reshape', 'param': ncnn_graph_attr, 'binary': []})

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

        if 'threshold' in layer['layer']['config'].keys():
            if layer['layer']['config']['threshold'] != 0:
                print('[ERROR] Leaky Clip ReLU is supported by ncnn.')
                frameinfo = inspect.getframeinfo(inspect.currentframe())
                print('Failed to convert at %s:%d %s()' %
                      (frameinfo.filename, frameinfo.lineno, frameinfo.function))
                sys.exit(-1)

        if 'negative_slope' in layer['layer']['config'].keys():
            negative_slope = layer['layer']['config']['negative_slope']
        else:
            negative_slope = 0.0

        if 'max_value' in layer['layer']['config'].keys():
            if layer['layer']['config']['max_value'] is not None:
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
                    'ReLU', slope=negative_slope)
                ncnn_graph_helper.node(
                    layer['layer']['name'], [
                        layer['layer']['name'] + '_Clip', ])
                ncnn_graph_helper.set_node_attr(
                    layer['layer']['name'], {
                        'type': 'ReLU', 'param': ncnn_graph_attr, 'binary': []})
            else:
                ncnn_graph_attr = ncnn_helper.dump_args(
                    'ReLU', slope=negative_slope)
                ncnn_graph_helper.node(
                    layer['layer']['name'],
                    keras_graph_helper.get_node_inbounds(
                        layer['layer']['name']))
                ncnn_graph_helper.set_node_attr(
                    layer['layer']['name'], {
                        'type': 'ReLU', 'param': ncnn_graph_attr, 'binary': []})
        else:
            ncnn_graph_attr = ncnn_helper.dump_args(
                'ReLU', slope=negative_slope)
            ncnn_graph_helper.node(
                layer['layer']['name'],
                keras_graph_helper.get_node_inbounds(
                    layer['layer']['name']))
            ncnn_graph_helper.set_node_attr(
                layer['layer']['name'], {
                    'type': 'ReLU', 'param': ncnn_graph_attr, 'binary': []})

    def LeakyReLU_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):

        ncnn_graph_attr = ncnn_helper.dump_args(
            'ReLU', slope=layer['layer']['config']['alpha'])
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

        SUPPORTED_ACTIVATION = ['', 'linear', 'softmax']
        SUPPORTED_FUSED_ACTIVATION_TYPE = {
            'relu': 1,
            'sigmoid': 4
        }

        if layer['layer']['config']['activation'] not in SUPPORTED_ACTIVATION and \
                layer['layer']['config']['activation'] not in SUPPORTED_FUSED_ACTIVATION_TYPE:
            print(
                '[ERROR] Activation type %s is is not supported yet.' %
                layer['layer']['config']['activation'])
            frameinfo = inspect.getframeinfo(inspect.currentframe())
            print('Failed to convert at %s:%d %s()' %
                  (frameinfo.filename, frameinfo.lineno, frameinfo.function))
            sys.exit(-1)

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

        if layer['layer']['config']['activation'] == '' or layer['layer']['config']['activation'] == 'linear':
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

        if layer['layer']['config']['activation'] == 'softmax':
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
                for node in keras_graph_helper.get_graph()[
                        name]['inbounds']:
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

        if layer['layer']['config']['activation'] in SUPPORTED_FUSED_ACTIVATION_TYPE:
            ncnn_graph_attr = ncnn_helper.dump_args(
                'InnerProduct',
                num_output=num_output,
                bias_term=1,
                activation_type=SUPPORTED_FUSED_ACTIVATION_TYPE[layer['layer']['config']['activation']],
                weight_data_size=weight_data_size)
            ncnn_graph_helper.node(
                layer['layer']['name'],
                keras_graph_helper.get_node_inbounds(
                    layer['layer']['name']))
            ncnn_graph_helper.set_node_attr(
                layer['layer']['name'], {
                    'type': 'InnerProduct', 'param': ncnn_graph_attr, 'binary': [
                        bn_params['bn_kernel'], bn_params['bn_bias']]})

    def Concatenate_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):

        DIM_SEQ = [3, 2, 0, 1]

        if DIM_SEQ[layer['layer']['config']['axis']] == 0:
            print('[ERROR] Concat asix = 0 is not support. ncnn only have C/H/W Dim.')
            frameinfo = inspect.getframeinfo(inspect.currentframe())
            print('Failed to convert at %s:%d %s()' %
                  (frameinfo.filename, frameinfo.lineno, frameinfo.function))
            sys.exit(-1)

        ncnn_graph_attr = ncnn_helper.dump_args(
            'Concat', axis=DIM_SEQ[layer['layer']['config']['axis']] - 1)
        ncnn_graph_helper.node(
            layer['layer']['name'],
            keras_graph_helper.get_node_inbounds(
                layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(
            layer['layer']['name'], {
                'type': 'Concat', 'param': ncnn_graph_attr, 'binary': []})

    def UpSampling2D_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):

        RESIZE_TYPE = ['', 'nearest', 'bilinear', 'bicubic']
        if 'interpolation' in layer['layer']['config'].keys():
            resize_type = RESIZE_TYPE.index(
                layer['layer']['config']['interpolation'])
        else:
            resize_type = RESIZE_TYPE.index('bilinear')

        ncnn_graph_attr = ncnn_helper.dump_args(
            'Interp', resize_type=resize_type, height_scale=float(
                layer['layer']['config']['size'][0]), width_scale=float(
                layer['layer']['config']['size'][0]))
        ncnn_graph_helper.node(
            layer['layer']['name'],
            keras_graph_helper.get_node_inbounds(
                layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(
            layer['layer']['name'], {
                'type': 'Interp', 'param': ncnn_graph_attr, 'binary': []})

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

    def Reshape_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):
        target_shape = layer['layer']['config']['target_shape']

        if(len(target_shape) == 4):
            ncnn_graph_attr = ncnn_helper.dump_args(
                'Reshape', w=target_shape[2], h=target_shape[1], c=target_shape[3])
        else:
            if(len(target_shape) == 3):
                ncnn_graph_attr = ncnn_helper.dump_args(
                    'Reshape', w=target_shape[1], h=target_shape[2])
            else:
                if(len(target_shape) == 2):
                    ncnn_graph_attr = ncnn_helper.dump_args(
                        'Reshape', w=target_shape[1])
                else:
                    if(len(target_shape) == 1):
                        return
                    else:
                        print(
                            '[ERROR] Reshape Layer Dim %d is not supported.' %
                            len(target_shape))
                        frameinfo = inspect.getframeinfo(
                            inspect.currentframe())
                        print(
                            'Failed to convert at %s:%d %s()' %
                            (frameinfo.filename, frameinfo.lineno, frameinfo.function))
                        sys.exit(-1)

        ncnn_graph_helper.node(
            layer['layer']['name'],
            keras_graph_helper.get_node_inbounds(
                layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(
            layer['layer']['name'], {
                'type': 'Reshape', 'param': ncnn_graph_attr, 'binary': []})

    def AveragePooling2D_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):

        if 'kernel_size' in layer['layer']['config'].keys():
            kernel_w, kernel_h = layer['layer']['config']['kernel_size']
        else:
            if 'pool_size' in layer['layer']['config'].keys():
                kernel_w, kernel_h = layer['layer']['config']['pool_size']
            else:
                print('[ERROR] Invalid configuration for pooling.')
                print('=========================================')
                print(layer['layer']['config'])
                print('=========================================')
                frameinfo = inspect.getframeinfo(inspect.currentframe())
                print('Failed to convert at %s:%d %s()' %
                      (frameinfo.filename, frameinfo.lineno, frameinfo.function))
                sys.exit(-1)

        if 'dilation_rate' in layer['layer']['config'].keys():
            dilation_w, dilation_h = layer['layer']['config']['dilation_rate']
        else:
            dilation_w = 1
            dilation_h = 1

        stride_w, stride_h = layer['layer']['config']['strides']

        if layer['layer']['config']['padding'] == 'valid':
            pad_mode = 1
        elif layer['layer']['config']['padding'] == 'same':
            pad_mode = 2
        else:
            pad_mode = 0

        ncnn_graph_attr = ncnn_helper.dump_args(
            'Pooling',
            pooling_type=1,
            kernel_w=kernel_w,
            dilation_w=dilation_w,
            stride_w=stride_w,
            kernel_h=kernel_h,
            dilation_h=dilation_h,
            stride_h=stride_h,
            pad_mode=pad_mode)

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

        if 'kernel_size' in layer['layer']['config'].keys():
            kernel_w, kernel_h = layer['layer']['config']['kernel_size']
        else:
            if 'pool_size' in layer['layer']['config'].keys():
                kernel_w, kernel_h = layer['layer']['config']['pool_size']
            else:
                print('[ERROR] Invalid configuration for pooling.')
                print('=========================================')
                print(layer['layer']['config'])
                print('=========================================')
                frameinfo = inspect.getframeinfo(inspect.currentframe())
                print('Failed to convert at %s:%d %s()' %
                      (frameinfo.filename, frameinfo.lineno, frameinfo.function))
                sys.exit(-1)

        if 'dilation_rate' in layer['layer']['config'].keys():
            dilation_w, dilation_h = layer['layer']['config']['dilation_rate']
        else:
            dilation_w = 1
            dilation_h = 1

        stride_w, stride_h = layer['layer']['config']['strides']

        if layer['layer']['config']['padding'] == 'valid':
            pad_mode = 1
        elif layer['layer']['config']['padding'] == 'same':
            pad_mode = 2
        else:
            pad_mode = 0

        ncnn_graph_attr = ncnn_helper.dump_args(
            'Pooling',
            pooling_type=0,
            kernel_w=kernel_w,
            dilation_w=dilation_w,
            stride_w=stride_w,
            kernel_h=kernel_h,
            dilation_h=dilation_h,
            stride_h=stride_h,
            pad_mode=pad_mode)

        ncnn_graph_helper.node(
            layer['layer']['name'],
            keras_graph_helper.get_node_inbounds(
                layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(
            layer['layer']['name'], {
                'type': 'Pooling', 'param': ncnn_graph_attr, 'binary': []})

    def Maximum_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):
        ncnn_graph_attr = ncnn_helper.dump_args('BinaryOp', op_type=4, with_scalar=0)

        ncnn_graph_helper.node(
            layer['layer']['name'],
            keras_graph_helper.get_node_inbounds(
                layer['layer']['name']))
        ncnn_graph_helper.set_node_attr(
            layer['layer']['name'], {
                'type': 'BinaryOp', 'param': ncnn_graph_attr, 'binary': []})

    def TensorFlowOpLayer_helper(
            self,
            layer,
            keras_graph_helper,
            ncnn_graph_helper,
            ncnn_helper):
        TFOPL_SUPPORTED_OP = ['Mul']

        operator = layer['layer']['config']['node_def']['op']
        if operator not in TFOPL_SUPPORTED_OP:
            print('[ERROR] Config for TensorFlowOpLayer is not supported yet.')
            print('=========================================')
            print(layer['layer'])
            print('=========================================')
            frameinfo = inspect.getframeinfo(inspect.currentframe())
            print('Failed to convert at %s:%d %s()' %
                    (frameinfo.filename, frameinfo.lineno, frameinfo.function))
            sys.exit(-1)
        
        if operator == 'Mul':
            # Create a MemoryData for storing the constant
            constant = layer['layer']['config']['constants']
            if len(constant) != 1:
                print('[ERROR] Config for TensorFlowOpLayer is not supported yet.')
                print('=========================================')
                print(layer['layer'])
                print('=========================================')
                frameinfo = inspect.getframeinfo(inspect.currentframe())
                print('Failed to convert at %s:%d %s()' %
                        (frameinfo.filename, frameinfo.lineno, frameinfo.function))
                sys.exit(-1)
            
            # ncnn_graph_attr = ncnn_helper.dump_args('MemoryData', w=1)
            # ncnn_graph_helper.node(
            #     layer['layer']['name']+'_const', [])
            # ncnn_graph_helper.set_node_attr(
            #     layer['layer']['name']+'_const', {
            #         'type': 'MemoryData', 'param': ncnn_graph_attr, 'binary': [[constant['0']]]})

            # Insert the mul layer
            ncnn_graph_attr = ncnn_helper.dump_args('BinaryOp', op_type=2, with_scalar=1, b=constant['0'])

            ncnn_graph_helper.node(
                layer['layer']['name'],
                keras_graph_helper.get_node_inbounds(
                    layer['layer']['name']))
            ncnn_graph_helper.set_node_attr(
                layer['layer']['name'], {
                    'type': 'BinaryOp', 'param': ncnn_graph_attr, 'binary': []})


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
                layer = keras_graph_helper.get_node_attr(node_name)['layer']
                print('[ERROR] Operator %s not support.' % layer['class_name'])
                print('=========================================')
                print(layer['config'])
                print('=========================================')
                frameinfo = inspect.getframeinfo(inspect.currentframe())
                print('Failed to convert at %s:%d %s()' %
                      (frameinfo.filename, frameinfo.lineno, frameinfo.function))
                sys.exit(-1)

        ncnn_graph_helper.refresh()

        for graph_head in ncnn_graph_helper.get_graph_head():
            node_attr = ncnn_graph_helper.get_node_attr(graph_head)
            if node_attr['type'] not in ['Input', 'MemoryData']:
                ncnn_graph_attr = ncnn_helper.dump_args(
                    'Input', w=-1, h=-1, c=-1)
                ncnn_graph_helper.node(
                    graph_head + '_input', [])
                ncnn_graph_helper.set_node_attr(
                    graph_head + '_input', {
                        'type': 'Input', 'param': ncnn_graph_attr, 'binary': []})
                ncnn_graph_helper.add_node_inbounds(
                    graph_head, graph_head + '_input')

        ncnn_graph_helper.refresh()
