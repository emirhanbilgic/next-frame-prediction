#3D CNN core structure:

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense

model_3d_cnn = Sequential([
    Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(10, 64, 64, 3)),  # 10 frames of 64x64 RGB images
    MaxPool3D(pool_size=(2, 2, 2)),
    Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
    MaxPool3D(pool_size=(2, 2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')  # binary classification for demonstration
])

model_3d_cnn.summary()

#ConvLSTM2D core structure:

from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization

model_convLSTM2D = Sequential([
    ConvLSTM2D(filters=40, kernel_size=(3, 3), input_shape=(10, 64, 64, 3), padding='same', return_sequences=True),
    BatchNormalization(),
    ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True),
    BatchNormalization(),
    ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True),
    BatchNormalization(),
    ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True),
    BatchNormalization(),
    Flatten(),
    Dense(1, activation='sigmoid')  # binary classification for demonstration
])

model_convLSTM2D.summary()

#PredRNN core structure:

from tensorflow.keras.layers import TimeDistributed, LSTM

model_predRNN = Sequential([
    TimeDistributed(Conv3D(32, kernel_size=(3, 3, 3), activation='relu'), input_shape=(10, 5, 64, 64, 3)),  # 10 timesteps of 5 frames
    TimeDistributed(MaxPool3D(pool_size=(2, 2, 2))),
    TimeDistributed(Conv3D(64, kernel_size=(3, 3, 3), activation='relu')),
    TimeDistributed(MaxPool3D(pool_size=(2, 2, 2))),
    TimeDistributed(Flatten()),
    LSTM(50, return_sequences=True),
    LSTM(50),
    Dense(1, activation='sigmoid')  # binary classification for demonstration
])

model_predRNN.summary()

