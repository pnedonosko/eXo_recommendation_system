from typing import List, Text

import absl
import tensorflow as tf
import tensorflow_transform as tft

from tfx.components.trainer.executor import TrainerFnArgs

_DENSE_FLOAT_FEATURE_KEYS = ['number_of_comments', 'owner_influence']

_BUCKET_FEATURE_KEYS = [
    'number_of_likes'
]

_LABEL_KEY = 'rank'

_TRAIN_BATCH_SIZE = 20
_EVAL_BATCH_SIZE = 10


# Helper functions
# =======================================================================

def _transformed_name(key):
  return key + '_xf'


def _transformed_names(keys):
  return [_transformed_name(key) for key in keys]


def _gzip_reader_fn(filenames):
  """
    Small utility returning a record reader that can read gzip'ed files.
    CsvExampleGen stores examples in gzip'ed files after reading.
  """
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')



def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example."""

  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    transformed_features = model.tft_layer(parsed_features)

    return model(transformed_features)

  return serve_tf_examples_fn



def _input_fn(file_pattern: List[Text],
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
    """Generates features and label for tuning/training.
    Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch
    Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=_transformed_name(_LABEL_KEY))


    return dataset

# One way to build a model,
# taken from https://github.com/tensorflow/tfx/blob/master/tfx/examples/iris/iris_utils_native_keras.py

# def _build_keras_model() -> tf.keras.Model:
#       """Creates a DNN Keras model for classifying taxi data.
#       Args:
#         hidden_units: [int], the layer sizes of the DNN (input layer first).
#       Returns:
#         A keras Model.
#       """
#
#       inputs = [
#           keras.layers.Input(shape=(1,), name=_transformed_name(f))
#           for f in _DENSE_FLOAT_FEATURE_KEYS
#       ]
#
#       d = keras.layers.concatenate(inputs)
#
#       for _ in range(int(4)):
#           d = keras.layers.Dense(8, activation='relu')(d)
#           print(type(d))
#       output = keras.layers.Dense(1)(d)
#
#
#       model = keras.Model(inputs=inputs, outputs=output)
#
#       model.compile(loss='mean_absolute_error',
#                     optimizer='adam',
#                     metrics=['mean_absolute_error'])
#       model.summary(print_fn=absl.logging.info)
#
#       return model


# Another way to build a model,
# taken from https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_utils_native_keras.py
def _build_keras_model(hidden_units: List[int] = None) -> tf.keras.Model:
    """Creates a DNN Keras model for classifying taxi data.

    Args:
    hidden_units: [int], the layer sizes of the DNN (input layer first).

    Returns:
    A keras Model.
    """
    real_valued_columns = [
        tf.feature_column.numeric_column(key, shape=())
        for key in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)
    ]

    categorical_columns = [
        tf.feature_column.categorical_column_with_identity(
            key, num_buckets=5, default_value=0)
        for key in _transformed_names(_BUCKET_FEATURE_KEYS)
    ]

    indicator_column = [
        tf.feature_column.indicator_column(categorical_column)
        for categorical_column in categorical_columns
    ]

    model = _wide_and_deep_classifier(
        wide_columns=indicator_column,
        deep_columns=real_valued_columns,
        dnn_hidden_units=hidden_units or [100, 70, 50, 25])
    return model

def _wide_and_deep_classifier(wide_columns, deep_columns, dnn_hidden_units):
    """Build a simple keras wide and deep model.

    Args:
    wide_columns: Feature columns wrapped in indicator_column for wide (linear)
      part of the model.
    deep_columns: Feature columns for deep part of the model.
    dnn_hidden_units: [int], the layer sizes of the hidden DNN.

    Returns:
    A Wide and Deep Keras model
    """
    # Following values are hard coded for simplicity in this example,
    # However prefarably they should be passsed in as hparams.

    # Keras needs the feature definitions at compile time.
    input_layers = {
      colname: tf.keras.layers.Input(name=colname, shape=(), dtype=tf.float32)
      for colname in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)
    }

    input_layers.update({
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
        for colname in _transformed_names(_BUCKET_FEATURE_KEYS)
    })

    
    deep = tf.keras.layers.DenseFeatures(deep_columns)(input_layers)
    for numnodes in dnn_hidden_units:
        deep = tf.keras.layers.Dense(numnodes)(deep)
    wide = tf.keras.layers.DenseFeatures(wide_columns)(input_layers)

    output = tf.keras.layers.Dense(
      1, activation='relu')(
          tf.keras.layers.concatenate([deep, wide]))
    output = tf.squeeze(output, -1)

    model = tf.keras.Model(input_layers, output)
    model.compile(loss='mean_absolute_error',
                  optimizer='adam',
                  metrics=['mean_absolute_error'])
    model.summary(print_fn=absl.logging.info)
    return model

# =======================================================================


# TFX Transform will call this function.
def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
      inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
      Map from string feature key to transformed feature operations.
    """
    outputs = {}

    for key in _BUCKET_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.bucketize(inputs[key], 5)

    for key in _DENSE_FLOAT_FEATURE_KEYS:
        outputs[_transformed_name(key)] = inputs[key]

    outputs[_transformed_name(_LABEL_KEY)] = inputs[_LABEL_KEY]
    return outputs


# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
    """Train the model based on given args.
    Args:
    fn_args: Holds args used to train the model as name/value pairs.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output,
                            batch_size=_TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output,
                           batch_size=_EVAL_BATCH_SIZE)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = _build_keras_model()


    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps)

    signatures = {
        'serving_default':
            _get_serve_tf_examples_fn(model,
                                      tf_transform_output).get_concrete_function(
                tf.TensorSpec(
                    shape=[None],
                    dtype=tf.string,
                    name='examples')),
    }

    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)



