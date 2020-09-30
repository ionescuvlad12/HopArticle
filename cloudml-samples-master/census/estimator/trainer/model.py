# Copyright 2016 Google Inc. All Rights Reserved. Licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
"""Defines a Wide + Deep model for classification on structured data.

Tutorial on wide and deep: https://www.tensorflow.org/tutorials/wide_and_deep/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# Define the format of your input data including unused columns
CSV_COLUMNS = [
    'age', 'gender', 'weight', 'height', 'tJump',
    'hJump', 'type', 'labels'
]
CSV_COLUMN_DEFAULTS = [[0], [''], [0], [0], [0], [0], [''], ['']]
LABEL_COLUMN = 'labels'
LABELS = ['p<=2','p>2']

# Define the initial ingestion of each feature used by your model.
# Additionally, provide metadata about the feature.
INPUT_COLUMNS = [
    # Categorical base columns

    # For categorical columns with known values we can provide lists
    # of values ahead of time.
    tf.feature_column.categorical_column_with_vocabulary_list(
        'gender', [' F', ' M']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'type', [
                 ' sarituri ca mingea',' sarituri ca mingea cu greutate', ' sarituri duble greutate',
                 ' sarituri duble', ' sarituri simple', ' sarituri simple cu greutate'
        ]),

    # Continuous base columns.
    tf.feature_column.numeric_column('age'),
    tf.feature_column.numeric_column('weight'),
    tf.feature_column.numeric_column('height'),
    tf.feature_column.numeric_column('tJump'),
    tf.feature_column.numeric_column('hJump'),
]

UNUSED_COLUMNS = set(CSV_COLUMNS) - {col.name for col in INPUT_COLUMNS} - \
    {LABEL_COLUMN}


def build_estimator(config, embedding_size=8, hidden_units=None):
    (gender, type, age, weight, height, tJump, hJump) = INPUT_COLUMNS
  # Reused Transformations.
  # Continuous columns can be converted to categorical via bucketization
    age_buckets = tf.feature_column.bucketized_column(
      age, boundaries=[6, 8, 10, 12, 14, 16, 18, 20, 22, 24])

  # Wide columns and deep columns.
    wide_columns = [
      # Interactions between different categorical features can also
      # be added as new virtual features.
#      tf.feature_column.crossed_column([hJump, tJump],
#                                       hash_bucket_size=int(1e4)),
#      tf.feature_column.crossed_column([age_buckets, 'type', 'gender'],
#                                       hash_bucket_size=int(1e6)),
                    tf.feature_column.crossed_column(['gender', 'type'],
                                      hash_bucket_size=int(1e4)),
      gender,
      weight,
      height,
      tJump,
      hJump,
      type,
      age_buckets,
  ]
    deep_columns = [
      # Use indicator columns for low dimensional vocabularies
      tf.feature_column.indicator_column(gender),
      tf.feature_column.indicator_column(type),

  ]
    return tf.estimator.DNNLinearCombinedClassifier(
      config=config,
      linear_feature_columns=wide_columns,
      dnn_feature_columns=deep_columns,
      dnn_hidden_units=hidden_units or [100, 70, 50, 25])


def parse_label_column(label_string_tensor):
    table = tf.contrib.lookup.index_table_from_tensor(tf.constant(LABELS))
    return table.lookup(label_string_tensor)


# ************************************************************************
# YOU NEED NOT MODIFY ANYTHING BELOW HERE TO ADAPT THIS MODEL TO YOUR DATA
# ************************************************************************


def csv_serving_input_fn():
    csv_row = tf.placeholder(shape=[None], dtype=tf.string)
    features = _decode_csv(csv_row)
    features.pop(LABEL_COLUMN)
    return tf.estimator.export.ServingInputReceiver(features,
                                                  {'csv_row': csv_row})


def example_serving_input_fn():
    example_bytestring = tf.placeholder(
      shape=[None],
      dtype=tf.string,
  )
    features = tf.parse_example(
      example_bytestring,
      tf.feature_column.make_parse_example_spec(INPUT_COLUMNS))
    return tf.estimator.export.ServingInputReceiver(
      features, {'example_proto': example_bytestring})


# [START serving-function]
def json_serving_input_fn():
    inputs = {}
    for feat in INPUT_COLUMNS:
        inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)

    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


# [END serving-function]

SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn,
    'EXAMPLE': example_serving_input_fn,
    'CSV': csv_serving_input_fn
}


def _decode_csv(line):
    row_columns = tf.expand_dims(line, -1)
    columns = tf.decode_csv(row_columns, record_defaults=CSV_COLUMN_DEFAULTS)
    features = dict(zip(CSV_COLUMNS, columns))

  # Remove unused columns
    for col in UNUSED_COLUMNS:
      features.pop(col)
    return features


def input_fn(filenames,
             num_epochs=None,
             shuffle=True,
             skip_header_lines=0,
             batch_size=200):

    dataset = tf.data.TextLineDataset(filenames).skip(skip_header_lines).map(
                                                                             _decode_csv)

    if shuffle:
      dataset = dataset.shuffle(buffer_size=batch_size * 10)
    iterator = dataset.repeat(num_epochs).batch(
        batch_size).make_one_shot_iterator()
    features = iterator.get_next()
    return features, parse_label_column(features.pop(LABEL_COLUMN))
