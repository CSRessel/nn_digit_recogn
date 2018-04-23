from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import random

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#import dataset

import logging
logging.getLogger().setLevel(logging.INFO)

from constants import *

percentage_noisy_labels = float(sys.argv[1])
test_sample_noise_factor = float(sys.argv[2])

print("\nImporting MNIST dataset...\n")

mnist = input_data.read_data_sets('MNIST_data')

config = tf.estimator.RunConfig(log_step_count_steps=LOG_STEP_COUNT_STEPS)
feature_columns = [tf.feature_column.numeric_column("x", shape=[784])]
classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=HIDDEN_UNITS,
        n_classes=10,
        config=config,
        model_dir="./tmp/model",
)

# add noise to labels in training set
labels = np.array(mnist.train.labels).astype(np.int32)
num_samples = labels.shape[0]
num_noisy_labels = num_samples * percentage_noisy_labels
garbled_labels = random.sample(range(num_samples), int(num_noisy_labels))
for i in garbled_labels:
	inc_labels = [np.int32(i) for i in range(10)]
	inc_labels.remove(labels[i])

	labels[i] = random.choice(inc_labels)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(mnist.train.images)},
      y=labels,
      num_epochs=None,
	  batch_size=BATCH_SIZE,
      shuffle=True)

test_samples = np.array(mnist.test.images)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_samples},
      y=np.array(mnist.test.labels).astype(np.int32),
      num_epochs=1,
      shuffle=False)

def noisify(x, weight):
	rand = np.random.rand(*x.shape)
	return x * (1. - weight) + rand * weight

noisy_test_samples = np.zeros(test_samples.shape)
for i in range(test_samples.shape[0]):
	noisy_test_samples[i] = noisify(test_samples[i], test_sample_noise_factor)

noisy_test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": noisy_test_samples},
      y=np.array(mnist.test.labels).astype(np.int32),
      num_epochs=1,
      shuffle=False)

print("\nTraining classifier...\n")

start = time.clock()
classifier.train(input_fn=train_input_fn, steps=STEPS)
finish = time.clock()

wall_time = finish - start

score = classifier.evaluate(input_fn=test_input_fn)['accuracy']
print("\nTest Accuracy: {0:f}\n".format(score))
score = classifier.evaluate(input_fn=noisy_test_input_fn)['accuracy']
print("\nNoisy Test Accuracy: {0:f}\n".format(score))
print("\nTraining Time: {0:f}\n".format(wall_time))
