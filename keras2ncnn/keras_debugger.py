from distutils import spawn as sp
import glob
import h5py
import os
import subprocess
import shutil
import sys


class KerasDebugger:
    ncnn_prog_template = \
        '''
#include "net.h"

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>

float rand_float() { return (float)rand() / (float)RAND_MAX; }

static int rand_mat(ncnn::Mat& m){
    for(int i=0; i<3; i++){
        float *ptr = m.channel(i);
        for(int j=0; j<(m.w * m.h); j++){
            ptr[j] = rand_float();
        }
    }

    return 0;
}

static int dump_mat(const ncnn::Mat& m, const char *m_name)
{
    char filename[1000] = "";
    sprintf(filename, "%s-%d-%d-%d-layer_dump.dat", m_name, m.c, m.h, m.w);

    FILE* fp = fopen(filename, "w+");
    if(fp == NULL){
        return -1;
    }

    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        fwrite(ptr, sizeof(float), m.w * m.h, fp);
    }

    fclose(fp);

    return 0;
}

static int detect_ncnn_net()
{
    ncnn::Net ncnn_net;

    ncnn_net.load_param("$FILENAME$.param");
    ncnn_net.load_model("$FILENAME$.bin");

    ncnn::Extractor ex = ncnn_net.create_extractor();

$INSERT$

    return 0;
}

int main(int argc, char** argv)
{
    detect_ncnn_net();

    return 0;
}

'''

    input_extractor_template = '\tncnn::Mat $layer_name_rep$_blob;\n'\
        '\t$layer_name_rep$_blob.create($input_shape$, 4u);\n'\
        '\trand_mat($layer_name_rep$_blob);\n'\
        '\tex.input("$layer_name$_blob", $layer_name_rep$_blob);\n\n'

    output_extractor_template = '\tncnn::Mat $layer_name_rep$;\n'\
                                '\tex.extract("$layer_name$_blob", $layer_name_rep$);\n'\
                                '\tdump_mat($layer_name_rep$, "$layer_name$");\n\n'

    tmp_dir = os.path.join('.keras2ncnn_build')

    def init_env(self):
        if 'win32' in sys.platform:
            print('\tThe debugger currently does not support Win32.')
            return -1

        if not os.path.exists(os.path.join(self.tmp_dir, '.stamp_env_init')):

            # Test for compiler
            required_utils = [
                'git',
                'gcc',
                'g++',
                'make',
                'cmake'
            ]

            subprocess.run(['python3', '-m', 'pip', 'install',
                            '--upgrade', 'virtualenv==16.7.9'])

            for util in required_utils:
                res = sp.find_executable(util)
                if res is None:
                    print(
                        '\t%s is not inside PATH, please install it first.' %
                        res)
                    return -1

            # Setup virtualenv
            import virtualenv  # pylint: disable=import-outside-toplevel
            virtualenv.create_environment(self.tmp_dir)

            activator_filename = os.path.join(
                self.tmp_dir, 'bin', 'activate_this.py')
            exec(open(activator_filename).read(),
                 {'__file__': activator_filename})

            install_pkg_list = [
                'numpy',
                'tensorflow==1.14',
                'matplotlib',
                'scipy'
            ]

            # Install packages
            subprocess.run(
                [os.path.join(self.tmp_dir, 'bin', 'pip3'), 'install', '--upgrade', 'pip'])
            subprocess.run(
                [os.path.join(self.tmp_dir, 'bin', 'pip3'), 'install'] + install_pkg_list)

            # Pull ncnn
            subprocess.run(['git',
                            'clone',
                            'https://github.com/Tencent/ncnn.git',
                            os.path.join(self.tmp_dir,
                                         'ncnn')])

            # setup build
            os.mkdir(os.path.join(self.tmp_dir, 'ncnn', 'build'))

            open(
                os.path.join(
                    self.tmp_dir,
                    'ncnn',
                    'benchmark',
                    'CMakeLists.txt'),
                'a').write(
                'add_executable(keras2ncnn keras2ncnn.cpp)\n' +
                'target_link_libraries(keras2ncnn PRIVATE ncnn)')

            open(
                os.path.join(
                    self.tmp_dir,
                    '.stamp_env_init'),
                'w+').write('Rua!')

        else:
            activator_filename = os.path.join(
                self.tmp_dir, 'bin', 'activate_this.py')
            exec(open(activator_filename).read(),
                 {'__file__': activator_filename})

        return 0

    def emit_file(self, file_name, ncnn_graph, keras_graph, graph_seq):

        def replace_bs(x): return x.replace('/', '_')

        extractor_list = []

        c_payload = self.ncnn_prog_template

        for layer_name in graph_seq:
            layer_type = ncnn_graph.get_node_attr(layer_name)['type']
            if layer_type == 'Input':
                op_shape = keras_graph.get_node_attr(
                    layer_name)
                if op_shape == None or None in op_shape:
                    print('Input has undetermind shape in W/H/C, default to 224,224,3')
                    op_shape = [3, 224, 224]
                else:
                    op_shape = op_shape['layer']['config']['batch_input_shape'][1:]
                extractor_list.append(
                    self.input_extractor_template.replace(
                        '$layer_name$',
                        layer_name).replace(
                        '$layer_name_rep$',
                        replace_bs(layer_name)).replace(
                        '$input_shape$',
                        ', '.join(list(map(str, op_shape)))
                    ))

            extractor_list.append(
                self.output_extractor_template.replace(
                    '$layer_name$',
                    layer_name).replace(
                    '$layer_name_rep$',
                    replace_bs(layer_name)))

        c_payload = c_payload.replace('$FILENAME$', file_name)
        c_payload = c_payload.replace('$INSERT$', '\n'.join(extractor_list))

        open(
            os.path.join(
                self.tmp_dir,
                'ncnn',
                'benchmark',
                'keras2ncnn.cpp'),
            'w+').write(c_payload)

        try:
            os.mkdir(os.path.join(self.tmp_dir, 'ncnn', 'build', 'benchmark'))
        except BaseException:
            pass

        shutil.copy(
            os.path.join(
                self.tmp_dir,
                file_name + '.param'),
            os.path.join(
                self.tmp_dir,
                'ncnn',
                'build',
                'benchmark'))
        shutil.copy(
            os.path.join(
                self.tmp_dir,
                file_name + '.bin'),
            os.path.join(
                self.tmp_dir,
                'ncnn',
                'build',
                'benchmark'))

    def run_debug(self):
        for f in glob.glob(
            os.path.join(
                self.tmp_dir,
                'ncnn',
                'build',
                'benchmark',
                '*.dat')):
            try:
                os.remove(f)
            except BaseException:
                pass

        try:
            os.remove(
                os.path.join(
                    self.tmp_dir,
                    'ncnn',
                    'build',
                    'benchmark',
                    'keras2ncnn'))
        except BaseException:
            pass

        subprocess.run(
            'cd %s; cmake -DNCNN_BENCHMARK=ON ..; cd benchmark; make -j4' %
            os.path.join(
                self.tmp_dir,
                'ncnn',
                'build'),
            shell=True)
        subprocess.run(
            'cd %s; chmod +x keras2ncnn; ./keras2ncnn' %
            (os.path.join(
                self.tmp_dir,
                'ncnn',
                'build',
                'benchmark')),
            shell=True)

    def decode(self, h5_file, keras_graph, graph_seq):
        import numpy as np  # pylint: disable=import-outside-toplevel
        from tensorflow.python import keras  # pylint: disable=import-outside-toplevel
        from tensorflow.python.keras import backend as K  # pylint: disable=import-outside-toplevel
        from tensorflow.python.keras.models import model_from_json
        K.set_learning_phase(0)

        from PIL import Image  # pylint: disable=import-outside-toplevel
        from matplotlib import pyplot as plt  # pylint: disable=import-outside-toplevel
        from numpy import linalg  # pylint: disable=import-outside-toplevel
        from scipy.spatial import distance  # pylint: disable=import-outside-toplevel

        # Read results from ncnn log file
        def strip_filename(x): return (os.path.split(x)[-1])
        def is_log_file(x): return '.dat' in x

        ncnn_det_out = {}

        log_files = filter(
            is_log_file,
            os.listdir(
                os.path.join(
                    self.tmp_dir,
                    'ncnn',
                    'build',
                    'benchmark')))
        for mat_file in log_files:
            mat_sp = strip_filename(mat_file).split('-')
            ncnn_det_out[mat_sp[0]] = np.fromfile(os.path.join(self.tmp_dir, 'ncnn', 'build', 'benchmark', mat_file),
                                                  dtype='float32').reshape(*list(map(int, mat_sp[1:4])))

        # Inference using keras
        f = h5py.File(h5_file, mode='r')
        model_config_raw = f.attrs.get('model_config')

        model = model_from_json(model_config_raw)
        model.load_weights(h5_file)

        output_node_names = [[], ]
        output_nodes = [[], ]
        input_images = []
        inputs = [model.inputs]

        for layer_idx in range(len(model.layers)):

            if 'Model' in str(type(model.layers[layer_idx])):
                output_node_names.append([])
                output_nodes.append([])
                # input_images.append([])
                inputs.append(model.layers[layer_idx].inputs)

                i = 0
                while True:
                    try:
                        layer = model.layers[layer_idx].get_layer(index=i)
                        node_name = layer.name
                        if node_name in ncnn_det_out.keys():
                            if keras_graph.get_node_attr(
                                    node_name)['layer']['class_name'] == 'InputLayer':
                                input_images.append(
                                    ncnn_det_out[node_name].transpose(
                                        1, 2, 0)[
                                        np.newaxis, ...])
                            else:
                                output_node_names[-1].append(node_name)
                                output_nodes[-1].append(layer.output)

                        i = i + 1
                    except BaseException:
                        break

            else:
                node_name = model.layers[layer_idx].name
                if node_name in ncnn_det_out.keys():
                    if keras_graph.get_node_attr(
                            node_name)['layer']['class_name'] == 'InputLayer':
                        input_images.append(
                            ncnn_det_out[node_name].transpose(
                                1, 2, 0)[
                                np.newaxis, ...])
                    else:
                        output_node_names[0].append(node_name)
                        output_nodes[0].append(model.layers[layer_idx].output)

        keras_layer_dumps_list = []

        for func_idx in range(len(output_node_names)):

            functor = K.function(inputs[func_idx], output_nodes[func_idx])
            layer_outs = functor(input_images + [1, ])
            keras_layer_dumps_list.append(
                dict(zip(output_node_names[func_idx], layer_outs)))

        keras_layer_dumps = {
            k: v for d in keras_layer_dumps_list for k,
            v in d.items()}

        output_node_names = [j for i in output_node_names for j in i]
        for output_node_name in graph_seq:
            if output_node_name not in output_node_names:
                print(output_node_name)
                continue
            print('==================================')

            print(
                'Layer Name: %s, Layer Shape: keras->%s ncnn->%s' %
                (output_node_name, str(
                    keras_layer_dumps[output_node_name].shape), str(
                    ncnn_det_out[output_node_name].shape)))
            print(
                'Max: \tkeras->%.03f ncnn->%.03f \tMin: keras->%.03f ncnn->%.03f' %
                (keras_layer_dumps[output_node_name].flatten().max(),
                 ncnn_det_out[output_node_name].flatten().max(),
                 keras_layer_dumps[output_node_name].flatten().min(),
                 ncnn_det_out[output_node_name].flatten().min()))
            print(
                'Mean: \tkeras->%.03f ncnn->%.03f \tVar: keras->%.03f ncnn->%.03f' %
                (keras_layer_dumps[output_node_name].flatten().mean(),
                 ncnn_det_out[output_node_name].flatten().mean(),
                 keras_layer_dumps[output_node_name].flatten().std(),
                 ncnn_det_out[output_node_name].flatten().std()))

            if keras_layer_dumps[output_node_name][0].ndim == 3:
                if keras_layer_dumps[output_node_name].size != ncnn_det_out[output_node_name].size:
                    print('Size not matched, not able to calculate similarity.')
                else:
                    print(
                        'Cosine Similarity: %.05f' %
                        distance.cosine(
                            keras_layer_dumps[output_node_name][0].transpose(
                                (2, 0, 1)).flatten(), ncnn_det_out[output_node_name].flatten()))

                print('Keras Feature Map: \t%s' % np.array2string(
                    keras_layer_dumps[output_node_name][0][0:10, 0, 0], suppress_small=True, precision=3))
                print('Ncnn Feature Map: \t%s' % np.array2string(
                    ncnn_det_out[output_node_name][0, 0:10, 0], suppress_small=True, precision=3))

            elif keras_layer_dumps[output_node_name][0].ndim == 2:
                if keras_layer_dumps[output_node_name].size != ncnn_det_out[output_node_name].size:
                    print('Size not matched, not able to calculate similarity.')
                else:
                    print(
                        'Cosine Similarity: %.05f' %
                        distance.cosine(
                            keras_layer_dumps[output_node_name][0].transpose(
                                (1, 0)).flatten(), ncnn_det_out[output_node_name].flatten()))

                print('Keras Feature Map: \t%s' % np.array2string(
                    keras_layer_dumps[output_node_name][0][0:10, 0], suppress_small=True, precision=3))
                print('Ncnn Feature Map: \t%s' % np.array2string(
                    ncnn_det_out[output_node_name][0, 0:10], suppress_small=True, precision=3))

            elif keras_layer_dumps[output_node_name][0].ndim == 1\
                    and (ncnn_det_out[output_node_name].shape[:2] == (1, 1)
                         or ncnn_det_out[output_node_name].ndim == 1):

                print(
                    'Cosine Similarity: %.05f' %
                    distance.cosine(
                        keras_layer_dumps[output_node_name][0].flatten(),
                        ncnn_det_out[output_node_name].flatten()))

                print('Keras Feature Map: \t%s' % np.array2string(
                    keras_layer_dumps[output_node_name][0][0:10], suppress_small=True, precision=3))

                if ncnn_det_out[output_node_name].ndim == 3:
                    print('Ncnn Feature Map: \t%s' % np.array2string(
                        ncnn_det_out[output_node_name][0, 0, 0:10], suppress_small=True, precision=3))

                keras_index = keras_layer_dumps[output_node_name][0].argsort(
                )[-5:][::-1]
                keras_top_value = keras_layer_dumps[output_node_name][0][keras_index]
                keras_topk = dict(zip(keras_index, keras_top_value))

                if ncnn_det_out[output_node_name].ndim == 3:
                    ncnn_index = ncnn_det_out[output_node_name][0, 0].argsort(
                    )[-5:][::-1]
                    ncnn_top_value = ncnn_det_out[output_node_name][0,
                                                                    0][ncnn_index]
                    ncnn_topk = dict(zip(ncnn_index, ncnn_top_value))

                if os.path.exists('./ImageNetLabels.txt'):
                    labels = open('ImageNetLabels.txt').readlines()

                    keras_topk_str = ", ".join(
                        ("%s:%.03f" % (labels[i[0] + 1].strip(), i[1]) for i in keras_topk.items()))

                    ncnn_topk_str = ", ".join(
                        ("%s:%.03f" % (labels[i[0] + 1].strip(), i[1]) for i in ncnn_topk.items()))

                else:
                    keras_topk_str = ", ".join(
                        ("%d:%.03f" % i for i in keras_topk.items()))

                    ncnn_topk_str = ", ".join(
                        ("%d:%.03f" % i for i in ncnn_topk.items()))

                print(
                    'Top-k:\nKeras Top-k: \t%s\nncnn Top-k: \t%s' %
                    (keras_topk_str, ncnn_topk_str))
