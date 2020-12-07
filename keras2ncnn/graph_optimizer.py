class GraphOptimization:

    @staticmethod
    def removing_unused_nodes(graph):
        UNUSED_NODES = ['Dropout', 'Lambda']
        nodes_to_remove = []

        for target_node_name in graph.get_graph().keys():
            if graph.get_node_attr(target_node_name)[
                    'layer']['class_name'] in UNUSED_NODES:
                for layer_name in graph.get_graph().keys():
                    if target_node_name in graph.get_graph()[
                            layer_name]['inbounds']:
                        graph.remove_node_inbounds(
                            layer_name, target_node_name)
                        graph.add_node_inbounds(
                            layer_name, graph.get_graph()[target_node_name]['inbounds'][0])
                nodes_to_remove.append(target_node_name)

        for removed_nodes_name in nodes_to_remove:
            graph.remove_node(removed_nodes_name)
