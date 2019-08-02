
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import config as cfg
import os
import lenet
from lenet import Lenet
import numpy as np

def main():
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    sess = tf.Session()
    batch_size = cfg.BATCH_SIZE
    parameter_path = cfg.PARAMETER_FILE
    lenet = Lenet()
    max_iter = cfg.MAX_ITER
    batchSize = cfg.BATCH_SIZE
    saver = tf.train.Saver()
    if os.path.exists(parameter_path):
        saver.restore(parameter_path)
    else:
        sess.run(tf.initialize_all_variables())

    for i in range(max_iter):
        batch = mnist.train.next_batch(batchSize)
        # print (batch[0].shape)
        inputBatch0 = np.reshape(batch[0],[batchSize,28,28,1])
        inputBatch1 = batch[1]
        # print (inputBatch0.shape)
        if i % 100 == 0:
            train_accuracy = sess.run(lenet.train_accuracy,feed_dict={lenet.input_images: inputBatch0,lenet.raw_input_label: inputBatch1})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        sess.run(lenet.train_op,feed_dict={lenet.input_images: inputBatch0,lenet.raw_input_label: inputBatch1})
    save_path = saver.save(sess, parameter_path)

if __name__ == '__main__':
    main()


