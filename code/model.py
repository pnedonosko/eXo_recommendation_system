from tensorflow.keras import callbacks
from tensorflow.keras import models
from tensorflow.keras import layers
import os

import warnings
warnings.filterwarnings('ignore')



class MyModel:

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = models.Sequential()

        # create folder for model weights
        if not os.path.exists("weights"):
            os.mkdir("weights")

        #initialize checkpoints
        checkpoint_name = 'weights/Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
        checkpoint = callbacks.ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        self.callbacks_list = [checkpoint]



    def build(self):
        # The Input Layer :
        self.model.add(layers.Dense(128, kernel_initializer='normal', input_dim=self.input_shape, activation='relu'))

        # The Hidden Layers :
        self.model.add(layers.Dense(256, kernel_initializer='normal', activation='relu'))
        self.model.add(layers.Dense(256, kernel_initializer='normal', activation='relu'))
        self.model.add(layers.Dense(256, kernel_initializer='normal', activation='relu'))

        # The Output Layer :
        self.model.add(layers.Dense(1, kernel_initializer='normal', activation='linear'))

        # Compile the network :
        self.model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.2, callbacks=self.callbacks_list)


    def evaluate(self, x_test, y_test):
        results = self.model.evaluate(x_test, y_test, batch_size=32)
        print("Test loss: {}\nTest accuracy: {}".format(results[0], results[1]))


