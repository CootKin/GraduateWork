import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
#import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, initializers

from utils import DataConverter


#################
### Параметры ###
#################
"""
BATCH_SIZE = 16
N_BARS = 4
N_STEPS_PER_BAR = 16
MAX_PITCH = 83
N_PITCHES = MAX_PITCH + 1
Z_DIM = 32

CRITIC_STEPS = 5
GP_WEIGHT = 10
DISCRIMINATOR_LR = 0.001
GENERATOR_LE = 0.001
BETA_1 = 0.5
BETA_2 = 0.9
EPOCHS = 1000
LOAD_MODEL = False
"""
REFACTOR_DATA = True
RAW_DATA_DIR_NAME = "LakhMidi"


#######################
### Загрузка данных ###
#######################

if (REFACTOR_DATA):
    data_converter = DataConverter(RAW_DATA_DIR_NAME)
    data_converter.extract_files()
    data_converter.delete_percussion()
    data_converter.preprocess(4)
    data_converter.remove_files()