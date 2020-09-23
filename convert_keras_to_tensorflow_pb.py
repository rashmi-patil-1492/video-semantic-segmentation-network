import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.framework.graph_util import remove_training_nodes

tf.disable_eager_execution()

# save model to pb ====================
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.
    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = remove_training_nodes(graph.as_graph_def())
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)

        return frozen_graph


# model_config = {
#     "model_class": "resnet50_segnet",
#     "n_classes": 12, "input_height": 416,
#     "input_width": 608, "output_height": 208,
#     "output_width": 304
# }

# save keras model as tf pb files ===============
import os
import json
from model.segnet import resnet50_segnet
wkdir = '.'
pb_filename = 'resnet50_segnet_model/resnet50_segnet.pb'
session = K.get_session()
K.set_learning_phase(0)

hd5_model = './resnet50_segnet_model/resnet50_segnet.h5'
config_file = './resnet50_segnet_model/resnet50_segnet_config.json'
assert (os.path.isfile(config_file)), "config file not found."
model_config = json.loads(open(config_file, "r").read())
print('loading model from config:', model_config)
model = resnet50_segnet(n_classes=model_config['n_classes'], input_height=model_config['input_height'], input_width=model_config['input_width'])
model.load_weights(hd5_model)
model.summary()
frozen_graph = freeze_session(session, output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, wkdir, pb_filename, as_text=False)