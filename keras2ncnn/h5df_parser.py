import json
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
            weight_layers = self.f['model_weights']
        else:
            weight_layers = self.f

        for weight_layer_name in [''] + list(weight_layers.keys()):
            if weight_layer_name == '':
                layer = weight_layers
            else:
                layer = weight_layers[weight_layer_name]

            while True:
                if layer_name not in layer:
                    break
                layer = layer[layer_name]
                if (not hasattr(layer, "keys")) or len(layer.keys()) > 1:
                    return layer
                layer_keys = list(layer.keys())
                if len(layer_keys) < 1:
                    break
                else:
                    layer_name = list(layer.keys())[0]

        return None

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
                tails = graph_helper.get_graph_tail()
                layers['name'] = layers['config']['name']

                graph_helper.node(layers['config']['name'], tails)
                graph_helper.set_node_attr(
                    layers['config']['name'], {
                        'layer': layers, 'weight': self.find_weights_root(
                            layers['config']['name'])})

    def parse_model_graph(self, model_layers, graph_helper):
        for layer in model_layers:
            inbound_nodes = self.join_inbound_nodes(layer)

            graph_helper.node(layer['name'], inbound_nodes)
            graph_helper.set_node_attr(
                layer['name'], {
                    'layer': layer, 'weight': self.find_weights_root(
                        layer['name'])})
