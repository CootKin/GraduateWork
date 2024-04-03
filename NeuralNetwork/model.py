import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from keras.models import Model, load_model
from keras.layers import Input, Dense

class TestModel:
    def __init__(self):
        self.__model = Model()

    @property
    def model(self):
        return self.__model

    def compile_model(self, _loss, _optimizer, _metrics):
        input_layer = Input(shape=(1,))
        hidden_layer = Dense(1, activation='relu')(input_layer)
        output_layer = Dense(1, activation='sigmoid')(hidden_layer)
        self.__model = Model(inputs=input_layer, outputs=output_layer)
        self.__model.compile(loss=_loss, optimizer=_optimizer, metrics=_metrics)

    def fit_model(self, x_train, y_train, _epochs, _verbose):
        self.__model.fit(x=x_train, y=y_train, epochs=_epochs, verbose=_verbose)

    def save_model(self, name):
        self.__model.save(f"ModelSaves/{name}")

    def load_model(self, name):
        self.__model = load_model(f"ModelSaves/{name}")

