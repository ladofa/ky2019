import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
# from matplotlib import pyplot as plt
from PIL import Image

import cv2
import time
import sys

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def preprocess(src,height,width):
    img = cv2.resize(src, (width, height))
    img = np.expand_dims(img[:,:,0],-1)
    np_image_data = np.asarray(img,dtype=np.float32)
    image_np_expanded = np.expand_dims(np_image_data, axis=0)  #1,28,28,1: NHWC
    return image_np_expanded

if __name__ == "__main__":

    modelFileName = "Lenet/trainNew/lenet.pb"
    height,width = 28, 28
    
    class_names = [0,1,2,3,4,5,6,7,8,9]
    imageFileName = "../images/im_1.png"

    img = cv2.imread(imageFileName)
    cv2.imshow("input",img)
    cv2.waitKey(0)
    image_np_expanded = preprocess(img,height,width)
    config = tf.ConfigProto()
    config.inter_op_parallelism_threads = 1
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(modelFileName, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
            # print("node names:")
            # for node in output_graph_def.node:
            #     print(node.op,"\t: ",node.name)

            print(("%d ops in the final graph." % len(output_graph_def.node)))
            len([x for x in tf.get_default_graph().get_operations() if x.name.startswith('name/')])

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            inputNodes = ["data"]
            outputNodes = ["Lenet/fc9_1/Relu"]
            image_tensor = sess.graph.get_tensor_by_name(inputNodes[0] + ":0")
            predictions = sess.graph.get_tensor_by_name(outputNodes[0] + ":0")
            print("input tensor shape:",image_np_expanded.shape)
            predictions = sess.run([predictions],feed_dict={image_tensor:image_np_expanded})
            maxIndex = np.argmax(predictions)
            print("prediction result:")
            print ("input image is \"%s\" in tensorflow lenet model"%class_names[maxIndex])
