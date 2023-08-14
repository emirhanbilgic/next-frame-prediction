import glob
import pandas as pd
from datetime import timedelta
import os
import shutil

#define the path and retrieve all PNG files from the given directory
input_directory = "2022_all/"
files = glob.glob(input_directory + "*.png")
files.sort()

#extract dates from the filenames
extracted_dates = [i.split("/")[1].split("-")[5].split(".")[0] for i in files]

#define the base format of the file name
base_filename_format = "MSG3-SEVI-MSGCLMK-0100-0100-{}.000000000Z-NA.png"

available_dates = []  #to store dates when the files are available
missing_dates = []  #to store dates when the files are missing

output_directory = "2022_processed_data/"
os.mkdir(output_directory)

#iteration over the date range and check for file availability
for date in pd.date_range("2022-01-01 00:00:00", "2022-08-31 23:00:00", freq="h"):
    print(date)
    start = date - timedelta(hours=24)
    end = date
    
    #create a range of dates in 30min intervals
    half_hour_range = pd.date_range(start, end, freq="30min")
    formatted_dates = [str(i).replace(":", "").replace("-", "").replace(" ", "") for i in half_hour_range]
    
    #check if all files in the date range are available
    if set(formatted_dates).issubset(set(extracted_dates)):
        available_dates.append(date)
        os.mkdir(f"{output_directory}/{date}")
        for formatted_date in formatted_dates:
            filename = base_filename_format.format(formatted_date)
            shutil.copyfile(f"{input_directory}{filename}", f"{output_directory}/{date}/{filename}")
    else:
        missing_dates.append(date)

#format and clean the lists for available and missing dates
def format_date_list(date_list):
    rounded_dates = [date.round("h") for date in date_list]
    string_dates = [str(date) for date in rounded_dates]
    unique_dates = list(set(string_dates))
    unique_dates.sort()
    return [i.replace(":", "").replace("-", "").replace(" ", "") for i in unique_dates]

available_dates = format_date_list(available_dates)
missing_dates = format_date_list(missing_dates)

#save the available dates into a CSV file
df = pd.DataFrame({"Date": available_dates})
df.to_csv("available_hours.csv")

