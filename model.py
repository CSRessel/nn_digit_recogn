from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import numpy as np
import dataset

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

feature_columns = [tf.feature_column.numeric_column("x", shape=[784])]
classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[64, 64],
        n_classes=10,
        model_dir="./tmp/model",
)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(mnist.train.images)},
      y=np.array(mnist.train.labels).astype(np.int32),
      num_epochs=None,
      shuffle=True)

#def train_input_fn():
    #ds = dataset.train("data")
    #ds = ds.cache().shuffle(buffer_size=50000).batch(100)
    #ds = ds.repeat(40)
    #return ds

#def eval_input_fn():
#    return dataset.test("data").batch(100).make_one_shot_iterator().get_next()

test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(mnist.test.images)},
      y=np.array(mnist.test.labels).astype(np.int32),
      num_epochs=1,
      shuffle=False)


classifier.train(input_fn=train_input_fn)
score = classifier.evaluate(input_fn=test_input_fn)['accuracy']
print("\nTest Accuracy: {0:f}\n".format(score))
