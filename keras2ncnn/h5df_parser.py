import json
import h5py
import sys


class H5dfParser:

    def __decode(self, payload):
        if isinstance(payload, (bytes, bytearray)):
            return payload.decode('utf-8')
        if isinstance(payload, str):
            return payload
        return str(payload)

    def __init__(self, h5_file):
        try:
            f = h5py.File(h5_file, mode='r')
            self.f = f
            model_config_raw = f.attrs.get('model_config')

        except Exception:
            print('[ERROR] Failed to read h5df file.')
            print('You are not selecting a valid keras model file.')
            print('You can check it by either opening it by Keras or Netron.')
            print('If you are very confident of your file, please repoert a bug at:')
            print('https://github.com/MarsTechHAN/keras2ncnn')
            sys.exit(-1)

        if not isinstance(model_config_raw, (str, bytes, bytearray)):
            print('[ERROR] Failed to load structure descriptor from h5df file.')
            print('You may load a weight only file.')
            print('Such issue may caused by following ways:')
            print('\t1. You are using model.save_weights instead of model.save')
            print('\t2. You are trying to load a weight file download from somewhere.')
            print('If you are very confident of your file, please repoert a bug at:')
            print('https://github.com/MarsTechHAN/keras2ncnn')
            sys.exit(-1)

        self.model_config = json.loads(self.__decode(model_config_raw))
        self.keras_version = self.get_keras_version()

        if self.keras_version != '1':
            weight_layers = self.f['model_weights']
        else:
            weight_layers = self.f

        self.weight_dict = {}
        weight_layers.visititems(self._get_weight_names)

    def get_h5df_file(self):
        return self.f

    def get_model_config(self):
        return self.model_config

    def get_keras_version(self):
        if 'keras_version' in self.f['model_weights'].attrs:
            original_keras_version = self.__decode(self.f['model_weights']\
                .attrs['keras_version'])
            return original_keras_version
        else:
            return '1'

    def get_backend_version(self):
        if 'backend' in self.f['model_weights'].attrs:
            original_backend = self.__decode(self.f['model_weights']\
                .attrs['backend'])
            return original_backend
        else:
            return None

    def _get_weight_names(self, name, obj):
        for key, val in obj.attrs.items():
            if key == 'weight_names':
                weight_names = list(
                    map(lambda x: self.__decode(x), val.tolist()))
                if len(weight_names) > 0:
                    wegith_group = '/'.join(weight_names[0].split('/')[0:-1])
                    self.weight_dict[name] = obj[wegith_group]
                    for weight_name in weight_names:
                        wegith_group = '/'.join(weight_name.split('/')[0:-1])
                        self.weight_dict[weight_name.split(
                            '/')[-2]] = obj[wegith_group]
                        self.weight_dict[wegith_group] = obj[wegith_group]

    def find_weights_root(self, layer_name):
        if layer_name in self.weight_dict.keys():
            return self.weight_dict[layer_name]
        else:
            return None

    def get_if_sequential(self):
        if self.model_config['class_name'] == 'Sequential' or \
                self.model_config['class_name'] == 'Model':
            return True
        else:
            return False

    def join_inbound_nodes(self, layer):
        inbound_nodes = []

        def get_inbound_nodes(inbound_list, inbound_nodes):
            for entry in inbound_list:
                if isinstance(entry, list):
                    get_inbound_nodes(entry, inbound_nodes)
                else:
                    if isinstance(entry, str):
                        inbound_nodes.append(entry)

        if 'inbound_nodes' in layer.keys():
            get_inbound_nodes(layer['inbound_nodes'], inbound_nodes)

        return inbound_nodes

    def parse_graph(self, graph_helper):
        self.joined_layers = []
        for layers in self.model_config['config']['layers']:
            if layers['class_name'] == 'Model':
                self.parse_model_graph(
                    layers['config']['layers'], graph_helper)
            else:
                if layers['class_name'] == 'TensorFlowOpLayer':
                    layer_name = layers['name']
                else:
                    layer_name = layers['config']['name']
                    layers['name'] = layers['config']['name']

                inbound_nodes = self.join_inbound_nodes(layers)
                if len(inbound_nodes) == 0:
                    inbound_nodes = graph_helper.get_graph_tail()

                graph_helper.node(layer_name, inbound_nodes)
                graph_helper.set_node_attr(
                    layer_name, {
                        'layer': layers, 'weight': self.find_weights_root(
                            layer_name)})

    def parse_model_graph(self, model_layers, graph_helper):
        for layer in model_layers:
            inbound_nodes = self.join_inbound_nodes(layer)

            graph_helper.node(layer['name'], inbound_nodes)
            graph_helper.set_node_attr(
                layer['name'], {
                    'layer': layer, 'weight': self.find_weights_root(
                        layer['name'])})
