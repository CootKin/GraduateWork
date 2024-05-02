import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import time
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint
from utils import DataConverter, DataLoader
from model import MuseGAN


#################
### Параметры ###
#################

COUNT_SONGS = 200
COUNT_BARS = 9
COUNT_BEATS = 144
COUNT_TRACKS = 4
COUNT_STEPS_PER_BAR = 16
COUNT_NOTES = 75

BATCH_SIZE = 64
NOISE_LENGTH = 32
DISCRIMINATOR_STEPS = 3
GRADIENT_PENALTY_WEIGHT = 10
DISCRIMINATOR_LEARNING_RATE = 0.001
GENERATOR_LEARNING_RATE = 0.001
ADAM_BETA_1 = 0.5
ADAM_BETA_2 = 0.9
EPOCHS = 5000

LOAD_MODEL = False
REFACTOR_DATA = False
DATA_FILENAME = "dataset"
RAW_DATA_DIR_NAME = "LakhMidi"
TAGS_FILE_PATH = os.path.join("./datasets/RefactorData/data/dataset.txt")



#######################
####### Функции #######
#######################

# Предобработка
def reshape_dataset(data, COUNT_SONGS, COUNT_BARS, COUNT_STEPS_PER_BAR, COUNT_TRACKS, COUNT_NOTES):
    reshape_data = data.reshape([COUNT_SONGS, COUNT_BARS, COUNT_STEPS_PER_BAR, COUNT_TRACKS])
    reshape_data = np.eye(COUNT_NOTES)[reshape_data]
    reshape_data[reshape_data == 0] = -1
    reshape_data = reshape_data.transpose([0, 1, 2, 4, 3])
    return reshape_data

# Построение графиков потерь
def plot_graphs(discriminator_loss, generator_loss):
    fig, ax =  plt.subplots(figsize=(16, 9),ncols=1, nrows=2)
    ax[0].plot(discriminator_loss, 'b')
    ax[0].plot(range(len(discriminator_loss)), [0]*len(discriminator_loss), color='black', linestyle='dashed')
    ax[0].set_ylabel("discriminator_loss", fontsize=14)
    ax[1].plot(generator_loss, 'r')
    ax[1].plot(range(len(generator_loss)), [0]*len(generator_loss), color='black', linestyle='dashed')
    ax[1].set_ylabel("generator_loss", fontsize=14)
    plt.xlabel("epochs", fontsize=14)
    fig.savefig("graphs/loss_graphs")



#######################
### Загрузка данных ###
#######################

if (REFACTOR_DATA):
    data_converter = DataConverter(RAW_DATA_DIR_NAME)
    data_converter.extract_files()
    time.sleep(30)
    data_converter.delete_percussion()
    time.sleep(30)
    data_converter.preprocess(32, 4)
    data_converter.remove_files()
    time.sleep(30)


data, tags = DataLoader(DATA_FILENAME).get_dataset()
data = reshape_dataset(data, COUNT_SONGS, COUNT_BARS, COUNT_STEPS_PER_BAR, COUNT_TRACKS, COUNT_NOTES)



#########################
### Модель и обучение ###
#########################

callback = ModelCheckpoint(
    filepath="./checkpoint/checkpoint_{epoch:02d}/checkpoint.weights.h5",
    save_weights_only=True,
    save_freq=100,
    verbose=0,
)


model = MuseGAN(NOISE_LENGTH,
                DISCRIMINATOR_STEPS,
                GRADIENT_PENALTY_WEIGHT,
                COUNT_BARS,
                COUNT_TRACKS,
                COUNT_STEPS_PER_BAR,
                COUNT_NOTES,
                True)

model.compile(DISCRIMINATOR_LEARNING_RATE,
              GENERATOR_LEARNING_RATE,
              ADAM_BETA_1,
              ADAM_BETA_2)

history = model.fit(
    data,
    epochs=101,
    callbacks=[callback]
)

plot_graphs(history.history['discriminator_loss'], history.history['generator_loss'])