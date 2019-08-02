from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
import os

from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

# model_dir = "trainOutput/"
model_dir = "LeNet/trainNew/"
model_file_name = "lenet"
ckpt_file_name = model_dir + model_file_name + ".ckpt"
checkpoint_path = os.path.join(ckpt_file_name)
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

# for key in var_to_shape_map:
#     print("tensor_name: ", key,reader.get_tensor(key).shape)
#     # print(reader.get_tensor(key)) #print values

# #Step 1
# #import the model metagraph
saver = tf.train.import_meta_graph(model_dir + 'lenet.ckpt.meta', clear_devices=True)
# #make that as the default graph
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
#now restore the variables
saver.restore(sess, model_dir + "lenet.ckpt")

#Step 2
# Find the output name
graph = tf.get_default_graph()
for op in graph.get_operations():
  print (op.name)

#Step 3
output_node_names = ["Lenet/fc9_1/Relu"]
output_graph_def = graph_util.convert_variables_to_constants(
        sess, # The session
        input_graph_def, # input_graph_def is useful for retrieving the nodes
        output_node_names  )

#Step 4
#output folder
output_fld ='./'
#output pb file name
output_model_file = model_dir + 'lenet.pb'
#write the graph
graph_io.write_graph(output_graph_def, output_fld, output_model_file, as_text=False)
graph_io.write_graph(output_graph_def, output_fld, output_model_file+"txt", as_text=True)


# meta_path = model_dir + 'lenet.ckpt.meta' # Your .meta file
# output_node_names = ['output']    # Output nodes
# graph = tf.get_default_graph()
# input_graph_def = graph.as_graph_def()
# with tf.Session() as sess:
#     # Restore the graph
#     saver = tf.train.import_meta_graph(meta_path)

#     # Load weights
#     # saver.restore(sess,tf.train.latest_checkpoint(model_dir + 'lenet.ckpt'))
#     saver.restore(sess, model_dir + "lenet.ckpt")

#     init=tf.global_variables_initializer()
#     sess.run(init)
#     # Freeze the graph
#     frozen_graph_def = tf.graph_util.convert_variables_to_constants(
#         sess,
#         input_graph_def,
#         output_node_names)

#     # Save the frozen graph
#     with open(model_dir + 'output_graph.pb', 'wb') as f:
#       f.write(frozen_graph_def.SerializeToString())

#     output_fld ='./'
#     #output pb file name
#     output_model_file = model_dir + 'lenetNew.pb'
#     #write the graph
#     graph_io.write_graph(frozen_graph_def, output_fld, output_model_file, as_text=False)
#     graph_io.write_graph(frozen_graph_def, output_fld, output_model_file+"txt", as_text=True)