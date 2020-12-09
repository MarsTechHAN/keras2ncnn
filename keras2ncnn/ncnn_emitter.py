import numpy as np


class NcnnEmitter:

    def __init__(self, ncnn_graph):
        self.MAGGGGGIC = 7767517
        self.ncnn_graph = ncnn_graph

    def get_graph_seq(self):

        graph_head = self.ncnn_graph.get_graph_head()

        # Thanks to Blckknght for the topological sort alg
        seen = set()
        stack = []
        order = []
        q = [graph_head[0]]

        for head in graph_head:
            q = [head]
            while q:
                v = q.pop()
                if v not in seen:
                    seen.add(v)
                    q.extend(self.ncnn_graph.get_node_outbounds(v))

                    while stack and v not in self.ncnn_graph.get_node_outbounds(
                            stack[-1]):
                        order.append(stack.pop())
                    stack.append(v)

        return stack + order[::-1]

    def emit_param(self, file_name, seq):
        ncnn_param_file = open(file_name, 'w+')

        ncnn_param_file.write('%d\n' % self.MAGGGGGIC)

        param_contect = ''
        blob_count = 0

        for layer_name in seq:
            layer_type = self.ncnn_graph.get_node_attr(layer_name)['type']
            input_count = len(self.ncnn_graph.get_node_inbounds(layer_name))

            output_count = len(self.ncnn_graph.get_node_outbounds(layer_name))
            output_count = 1 if output_count == 0 else output_count

            input_blobs = []
            inbound_nodes = self.ncnn_graph.get_node_inbounds(layer_name)
            for in_node in inbound_nodes:
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

            blob_count += len(output_blobs)

            param_contect += (
                ('%s' + (
                    25 - len(layer_type)) * ' ' + '%s' + (
                    40 - len(layer_name)) * ' ' + '%d %d %s %s %s\n') % (layer_type,
                                                                         layer_name,
                                                                         input_count,
                                                                         output_count,
                                                                         ' '.join(input_blobs),
                                                                         ' '.join(output_blobs),
                                                                         self.ncnn_graph.get_node_attr(layer_name)['param']))

        layer_count = len(self.ncnn_graph.get_graph())
        ncnn_param_file.write('%d %d\n' % (layer_count, blob_count))
        ncnn_param_file.write(param_contect)

        ncnn_param_file.close()

    def emit_binary(self, file_name, seq):
        f = open(file_name, 'w+b')
        for layer_name in seq:
            for weight in self.ncnn_graph.get_node_attr(layer_name)['binary']:
                f.write(np.asarray(weight, dtype=np.float32).tobytes())
        f.close()
