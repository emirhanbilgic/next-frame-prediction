from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import os

# check available gpus
print(tf.config.list_physical_devices('GPU'))

# connect to Google Drive
from google.colab import drive
drive.mount('/content/drive')

# get list of NPZ files from Drive folder
npz_file_names = os.listdir('/content/drive/MyDrive/all_npz_files/')

# create a dataframe with file names and their paths
file_info_df = pd.DataFrame({
    'Date': [x.split('.')[0] for x in npz_file_names],
    'NPZ_Path': ['/content/drive/MyDrive/all_npz_files/' + x for x in npz_file_names]
})

# convert Date column to string
file_info_df['Date'] = file_info_df['Date'].astype(str)

# split data into train and validation datasets
train_df = file_info_df[:1500]
val_df = file_info_df[1500:1750]

# function to get X and y from given dataframe
def extract_features_labels(df):
    X_image, X_statistics = [], []
    labels = []

    for file_path in df['NPZ_Path']:
        loaded_npz = np.load(file_path)

        image = loaded_npz['pic']
        X_image.append(image)

        statistics = loaded_npz['stats']
        X_statistics.append(statistics)

        labels.append(loaded_npz['Lagged_Production'])

    X_image, X_statistics = np.array(X_image), np.array(X_statistics)
    labels = np.array(labels)

    return (X_image, X_statistics), labels

(X_train_image, X_train_statistics), y_train = extract_features_labels(train_df)
(X_val_image, X_val_statistics), y_val = extract_features_labels(val_df)

# defining the parallel model
def create_parallel_model():
    image_input = keras.Input((49, 350, 350, 3)) # depth, height, width, channels

    x_image = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(image_input)
    x_image = layers.MaxPool3D(pool_size=2)(x_image)
    x_image = layers.BatchNormalization()(x_image)

    x_image = layers.Conv3D(filters=1, kernel_size=3, activation="relu")(x_image)
    x_image = layers.MaxPool3D(pool_size=2)(x_image)
    x_image = layers.BatchNormalization()(x_image)

    x_image = layers.GlobalAveragePooling3D()(x_image)

    statistics_input = keras.Input(shape=(25,))
    x_statistics = layers.Dense(units=100, activation="relu")(statistics_input)
    x_statistics = layers.Dense(50, activation='relu')(x_statistics)

    merged_layers = concatenate([x_image, x_statistics])
    dense_layer = Dense(8, activation='relu', name='dense_layer1')(merged_layers)
    output_layer = Dense(1, activation='linear', name='dense_layer2')(dense_layer) # linear for regression

    model = keras.Model(inputs=[image_input, statistics_input], outputs=output_layer)
    return model

model = create_parallel_model()
model.summary()

optimizer = Adam(learning_rate=0.0001)
model.compile(loss='mse', optimizer=optimizer, metrics=['mean_absolute_error'])

checkpoint_callback = ModelCheckpoint('model/', save_best_only=True)
model.fit(x=[X_train_image, X_train_statistics], y=y_train, 
          validation_data=([X_val_image, X_val_statistics], y_val), 
          batch_size=1, epochs=50, callbacks=[checkpoint_callback])

loaded_model = load_model('model/')

# testing phase
comparison_df = pd.DataFrame()
start_index = 0
step = 3

while start_index <= 717: #717 is for the ram insufficiency 
    test_df = file_info_df[2000+start_index:2000+start_index+step]
    (X_test_image, X_test_statistics), y_test = extract_features_labels(test_df)
    y_test_series = pd.Series(y_test)
    test_df.reset_index(drop=True, inplace=True)
    test_df['Lagged_Production'] = y_test_series
    predictions = loaded_model.predict([X_test_image, X_test_statistics]).flatten()
    predicted_series = pd.Series(predictions)
    test_df.reset_index(drop=True, inplace=True)
    test_df['Predicted_Lagged_Production'] = predicted_series
    comparison_df = comparison_df.append(test_df)
    start_index += step

mape = 100 * abs(comparison_df["Lagged_Production"] - comparison_df["Predicted_Lagged_Production"]).sum() / comparison_df["Lagged_Production"].sum()
mape
