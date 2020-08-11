import tensorflow as tf

class MyModel:

    def __init__(self):
        self.model = None


    def build(self, inputs, linear_feature_columns, dnn_feature_columns, dnn_hidden_units):
        deep = tf.keras.layers.DenseFeatures(dnn_feature_columns)(inputs)
        for numnodes in dnn_hidden_units:
            deep = tf.keras.layers.Dense(numnodes, activation='relu')(deep)
        wide = tf.keras.layers.DenseFeatures(linear_feature_columns)(inputs)
        both = tf.keras.layers.concatenate([deep, wide])
        output = tf.keras.layers.Dense(1)(both)
        model = tf.keras.Model(inputs, output)

        model.compile(loss='mean_absolute_error',
                      optimizer='adam',
                      metrics=['mean_absolute_error'])

        self.model = model

    def train(self, train_ds, val_ds):
        self.model.fit(train_ds,
                  validation_data=val_ds,
                  epochs=50)



    def evaluate(self, test_ds):
        results = self.model.evaluate(test_ds, batch_size=32)
        print("Test loss: {}\nTest accuracy: {}".format(results[0], results[1]))

