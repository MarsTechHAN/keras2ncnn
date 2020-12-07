import os

from pathlib import Path
import argparse

from keras2ncnn.graph_tool import Grapher
from keras2ncnn.graph_optimizer import GraphOptimization
from keras2ncnn.h5df_parser import H5dfParser
from keras2ncnn.keras_converter import KerasConverter
from keras2ncnn.keras_debugger import KerasDebugger
from keras2ncnn.ncnn_emitter import NcnnEmitter
from keras2ncnn.ncnn_param import NcnnParamDispatcher


def main():
    parser = argparse.ArgumentParser()

    if parser.prog == '__main__.py':
        parser.prog = 'python3 -m keras2ncnn'

    parser.add_argument(
        '-i',
        '--input_file',
        type=str,
        help='Input h5df file',
        required=True)

    parser.add_argument(
        '-o',
        '--output_dir',
        type=str,
        help='Output file dir',
        default='')

    parser.add_argument(
        '-p',
        '--plot_graph',
        action='store_true',
        help='Virtualize graph.')

    parser.add_argument(
        '-d',
        '--debug',
        action='store_true',
        help='Output debug C file.')

    parser.add_argument(
        '-l',
        '--load_debug_log',
        type=str,
        default='',
        help='Load debug log for comparing.')

    parser.add_argument(
        '-m',
        '--input_image',
        type=str,
        default='',
        help='Input image for comparing')
    args = parser.parse_args()

    # Create a source graph and a dest graph
    keras_graph = Grapher()
    ncnn_graph = Grapher()

    # Read and parse keras file to graph
    print('Reading and parsing keras h5df file...')
    H5dfParser(args.input_file).parse_graph(keras_graph)

    # Graph Optimization
    print('Start graph optimizing pass...')
    print('\tRemoving unused nodes...')
    GraphOptimization.removing_unused_nodes(keras_graph)

    print('\tRefreshing graph...')
    keras_graph.refresh()

    # Convert keras to ncnn representations
    print('Converting keras graph to ncnn graph...')
    KerasConverter().parse_keras_graph(keras_graph, ncnn_graph, NcnnParamDispatcher())

    if args.plot_graph:
        print('Rendering graph plots...')
        keras_graph.plot_graphs(Path(args.input_file).stem + '_keras')
        ncnn_graph.plot_graphs(Path(args.input_file).stem + '_ncnn')

    # Emit the graph to params and bin

    if args.output_dir != '':
        print('Start emitting to ncnn files.')
        emitter = NcnnEmitter(ncnn_graph)
        graph_seq = emitter.get_graph_seq()

        print('\tEmitting param...')
        emitter.emit_param(
            os.path.join(
                args.output_dir,
                Path(
                    args.input_file).stem +
                '.param'), graph_seq)

        print('\tEmitting binary...')
        emitter.emit_binary(
            os.path.join(
                args.output_dir,
                Path(
                    args.input_file).stem +
                '.bin'), graph_seq)

    if args.debug:
        print('Generating ncnn dump helper file...')
        KerasDebugger().dump2c(Path(args.input_file).stem, ncnn_graph)

    if args.load_debug_log != '':
        print('Start loading debug log...')
        KerasDebugger().decode(args.input_file, args.load_debug_log)

    print('Done!')