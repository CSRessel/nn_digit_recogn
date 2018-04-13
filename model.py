from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import dataset

feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]
classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[64, 64],
        model_dir="./tmp/model",
        n_classes=10
)

def train_input_fn():
    ds = dataset.train("data")
    ds = ds.cache().shuffle(buffer_size=50000).batch(100)
    ds = ds.repeat(40)
    return ds
def eval_input_fn():
    return dataset.test("data").batch(100).make_one_shot_iterator().get_next()

classifier.train(input_fn=train_input_fn)
eval_results = classifier.evaluate(input_fn=eval_input_fn)
print("\nEvaluation results:\n\t%s\n" % eval_results)
