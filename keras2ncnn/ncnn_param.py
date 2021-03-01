import inspect
import sys


class NcnnParamDispatcher:
    operation_param_table = {
        'BatchNorm': {
            0: {'channels': 0},
            1: {'eps': 0},
        },

        'BinaryOp': {
            0: {'op_type': 0},
            1: {'with_scalar': 0},
            2: {'b': 0.},
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

            9: {'activation_type': 0},
            # 10: {'activation_params': 0},

            11: {'kernel_h': 0},
            12: {'dilation_h': 1},
            13: {'stride_h': 1},
        },

        'Deconvolution': {
            0: {'num_output': 0},
            1: {'kernel_w': 0},
            2: {'dilation_w': 1},
            3: {'stride_w': 0},
            4: {'pad_left': 0},
            5: {'bias_term': 0},
            6: {'weight_data_size': 0},

            9: {'activation_type': 0},
            # 10: {'activation_params': 0},

            11: {'kernel_h': 0},
            12: {'dilation_h': 1},
            13: {'stride_h': 1},
            14: {'pad_top': 0},
            15: {'pad_right': 0},
            16: {'pad_bottom': 0},

            18: {'output_pad_right': 0},
            19: {'output_pad_bottom': 0},

            # 20: {'output_w': 0},
            # 21: {'output_h': 0},
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

            9: {'activation_type': 0},
        },

        'Input': {
            0: {'w': 0},
            1: {'h': 0},
            2: {'c': 0},
        },

        'Interp': {
            0: {'resize_type': 0},
            1: {'height_scale': 1.0},
            2: {'width_scale': 1.0},
            3: {'output_height': 0},
            4: {'output_width': 0},
        },

        'Padding': {
            0: {'top': 0},
            1: {'bottom': 0},
            2: {'left': 0},
            3: {'right': 0},
            # 4: {'type': 0},
            # 5: {'value': 0}
        },

        'Pooling': {
            0: {'pooling_type': 0},
            1: {'kernel_w': 0},
            11: {'kernel_h': 0},
            2: {'stride_w': 1},
            12: {'stride_h': 1},
            3: {'pad_left': 0},
            4: {'global_pooling': 0},
            5: {'pad_mode': 0},
        },

        'ReLU': {
            0: {'slope': 0},
            1: {'stride': 0},
        },

        'Reshape': {
            0: {'w': -233},
            1: {'h': -233},
            2: {'c': -233},
            3: {'flag': 1}
        },

        'Sigmoid': {
        },

        'Softmax': {
            0: {'axis': 0},
        },

        'Split': {

        },

        'MemoryData': {
            0: {'w': 0},
            1: {'h': 0},
            2: {'c': 0}
        }

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
                print('[ERROR] Failed to dump arg %s with type %s.' %
                      (arg_name, type(params_arg)))
                frameinfo = inspect.getframeinfo(inspect.currentframe())
                calframe = inspect.getouterframes(inspect.currentframe(), 2)
                print(
                    'Failed to convert at %s:%d %s() called from %s()' %
                    (frameinfo.filename, frameinfo.lineno, frameinfo.function, calframe[1][3]))
                sys.exit(-1)
        return ncnn_args_phrase
