from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#import dataset

import logging
logging.getLogger().setLevel(logging.INFO)

from constants import *

print("\nImporting MNIST dataset...\n")

mnist = input_data.read_data_sets('MNIST_data')

config = tf.estimator.RunConfig(log_step_count_steps=1000)
feature_columns = [tf.feature_column.numeric_column("x", shape=[784])]
classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=([HIDDEN_UNIT_WIDTH] * HIDDEN_UNIT_DEPTH),
        n_classes=10,
        config=config,
        model_dir="./tmp/model",
)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(mnist.train.images)},
      y=np.array(mnist.train.labels).astype(np.int32),
      num_epochs=None,
	  batch_size=50,
      shuffle=True)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(mnist.test.images)},
      y=np.array(mnist.test.labels).astype(np.int32),
      num_epochs=1,
      shuffle=False)

print("\nTraining classifier...\n")

start = time.clock()
classifier.train(input_fn=train_input_fn, steps=100000)
finish = time.clock()

wall_time = finish - start

score = classifier.evaluate(input_fn=test_input_fn)['accuracy']
print("\nTest Accuracy: {0:f}\n".format(score))
print("\nTraining Time: {0:f}\n".format(wall_time))
