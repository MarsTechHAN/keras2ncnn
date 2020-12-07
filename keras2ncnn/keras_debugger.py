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
        'Input',

        'Convolution',
        'Conv2D',

        # 'ConvolutionDepthWise',
        # 'DepthwiseConv2D',

        # 'BatchNorm',
        # 'BatchNormalization',

        'Deconvolution',
        'Conv2DTranspose',

        # 'ReLU',
        'Softmax',

        'InnerProduct',
        'Dense']

    input_extractor_template = 'ex.input("$layer_name$_blob", in);'
    output_extractor_template = '    ncnn::Mat $layer_name$; '\
                                'ex.extract("$layer_name$_blob", $layer_name$); '\
                                'dump_mat($layer_name$, "$layer_name$");'

    def dump2c(self, file_name, ncnn_graph):
        import gzip  # pylint: disable=import-outside-toplevel
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

            if layer_type in self.target_operators:
                extractor_list.append(
                    self.output_extractor_template.replace(
                        '$layer_name$', layer_name))

        c_payload = c_payload.replace('$FILENAME$', file_name)
        c_payload = c_payload.replace('$INPUT_W$', '224')
        c_payload = c_payload.replace('$INPUT_H$', '224')
        c_payload = c_payload.replace('$INSERT$', '\n'.join(extractor_list))

        open('%s.c' % file_name, 'w+').write(c_payload)

    def decode(self, h5_file, log_file):
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
        model = keras.models.load_model(h5_file)
        test_img = np.asarray(Image.open(args.input_image).resize((224, 224)))
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

            if layer_name not in keras_layer_dumps:
                continue

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
                )[-5:][::-1]
                keras_top_value = keras_layer_dumps[layer_name][0][keras_index]
                keras_topk = dict(zip(keras_index, keras_top_value))

                if ncnn_layer_dumps[ncnn_layer_name].ndim == 3:
                    ncnn_index = ncnn_layer_dumps[ncnn_layer_name][0, 0].argsort(
                    )[-5:][::-1]
                    ncnn_top_value = ncnn_layer_dumps[ncnn_layer_name][0,
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
