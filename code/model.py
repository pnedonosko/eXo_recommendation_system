import tensorflow as tf
from tensorflow.keras import layers

import warnings
warnings.filterwarnings('ignore')



class MyModel:

    def __init__(self):
        self.model = tf.keras.Sequential()


    def build(self, feature_columns):
        # create layer from column features
        feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

        self.model.add(feature_layer)
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(1))


        self.model.compile(loss='mean_absolute_error',
                      optimizer='adam',
                      metrics=['mean_absolute_error'])


    def train(self, train_ds, val_ds):
        self.model.fit(train_ds,
                  validation_data=val_ds,
                  epochs=100)



    def evaluate(self, test_ds):
        results = self.model.evaluate(test_ds, batch_size=32)
        print("Test loss: {}\nTest accuracy: {}".format(results[0], results[1]))


