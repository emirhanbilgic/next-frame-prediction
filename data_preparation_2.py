from google.colab import drive
import feather
import pandas as pd
import numpy as np
import os

#mount Google Drive
drive.mount('/content/drive')

#load data from a feather file
data = pd.read_feather("/content/drive/MyDrive/aydem_feather/aydem_full.feather")

#filter data to only include entries from 2022 onwards
data_filtered = data.iloc[8757:]

#remove columns 'Kurulu Güç' and 'Production_scaled'
data_cleaned = data_filtered.loc[:, ~data_filtered.columns.isin(['Kurulu Güç', 'Production_scaled'])]
data_no_na = data_cleaned.dropna()

#define z-score standardization function
def z_score_standardization(series):
    return (series - series.mean()) / series.std()

#select columns to apply z-score standardization
columns_to_standardize = data_no_na.columns[1:26]  
date_column = data_no_na.columns[0]

#apply z-score standardization to the columns
for col in columns_to_standardize:
    data_no_na[col] = z_score_standardization(data_no_na[col])
data_standardized = data_no_na

#generate a lagged production column
data_standardized["Lagged_Production"] = data_standardized["Production"].shift(-40)

#select only the 'Date' column
date_only_column = data_standardized.loc[:, data_standardized.columns.isin(['Date'])]
production_only_column = data_standardized.loc[:, data_standardized.columns.isin(['Production'])]

#remove the 'Production' column
data_without_production = data_standardized.loc[:, ~data_standardized.columns.isin(['Production'])]

#get filenames from a directory in Google Drive
file_names = os.listdir('/content/drive/MyDrive/buyuk_resim_dosyasi/')

#convert file names to DataFrame
file_names_df = pd.DataFrame({
    'Date': [x.split('.')[0] for x in file_names],
    'Path': ['/content/drive/MyDrive/buyuk_resim_dosyasi/' + x for x in file_names]
})

#convert 'Date' columns to string type for merging
file_names_df['Date'] = file_names_df['Date'].astype(str)
data_without_production['Date'] = data_without_production['Date'].astype(str)

#merge data on 'Date'
merged_data = data_without_production.merge(file_names_df, on='Date', how='inner')

#initialize empty list for paths
npz_file_paths = []

#iterate through rows and save data as .npz files
for i, row in merged_data.iterrows():
    image_path = row['Path']
    npz_path = image_path.split('.')[0] + '.npz'
    npz_file_paths.append(npz_path)
    img_data = np.load(image_path)
    stats = np.array([row[col] for col in columns_to_standardize])
    lagged_production = row['Lagged_Production']

    #save compressed data as .npz file
    np.savez_compressed(npz_path, pic=img_data, stats=stats, Lagged_Production=lagged_production)

#add paths of .npz files to the DataFrame
merged_data['NPZ_Path'] = pd.Series(npz_file_paths)

merged_data.head()
