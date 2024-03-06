import os
from typing import Dict
import numpy as np
import pandas as pd
import scipy.io as sio
import re

os.chdir(r"C:\Users\doron\OneDrive\Desktop\thesis\TSC\yuri_data")

DIRECTORY = "Israel_South_2004-2014_EX_HRFI_mat"
EXCEL_HRFI = "GII_4NETA_Israel_2004-2014_SOUTH_EX_HRFI.xlsx"
EXCEL_HRFI_HARTUV = "GII_4NETA_Israel_2004-2014_SOUTH_EX_HRFI_HarTuv.xlsx"
EXCEL_HRFI_MITZPERAMON = "GII_4NETA_Israel_2004-2014_SOUTH_EX_HRFI_MitzpeRamon.xlsx"
EXCEL_HRFI_ORON = "GII_4NETA_Israel_2004-2014_SOUTH_EX_HRFI_Oron.xlsx"
EXCEL_HRFI_ROTEM = "GII_4NETA_Israel_2004-2014_SOUTH_EX_HRFI_Rotem.xlsx"

df_hartuv = pd.read_excel(EXCEL_HRFI_HARTUV)
df_mitzperamon = pd.read_excel(EXCEL_HRFI_MITZPERAMON)
df_oron = pd.read_excel(EXCEL_HRFI_ORON)
df_rotem = pd.read_excel(EXCEL_HRFI_ROTEM)

df_hrfi = pd.read_excel(EXCEL_HRFI)

helper = {"hartuv": df_hartuv, "mitzperamon": df_mitzperamon, "oron": df_oron, "rotem": df_rotem}

# Get list of all files in the directory
files = os.listdir(DIRECTORY)

data = [file for file in files]
string = "HRFI.BHE.200406071506"
# data = [string]

# Define regular expressions to match the components
pattern = r'([A-Z]+)\.([A-Z]+)\.(\d{12})'

# Use re.match to search for the pattern in the string
match = re.match(pattern, string)

def parsing_file(string) -> Dict:
    parts = string.split('.')
    timestamp = parts[2]
    mat = sio.loadmat(f'{DIRECTORY}/{string}')
    vec = np.hstack(mat["W"])
    station = find_station(timestamp)
    metadata = df_hrfi[df_hrfi["YYYYMMDDHHMiMi"] == timestamp]
    existing_df = pd.DataFrame({'Station': station,
                                'Axis': parts[1],
                                'Timestamp': timestamp,
                                'Vector': vec})
    resulted_df = pd.concat([existing_df, metadata], axis = 1)

    return resulted_df

def find_station(timestamp):
    for station in helper:
        tempdf = helper[station]
        if tempdf["YYYYMMDDHHMiMi"] == int(timestamp):
            return station
    return "HRFI"

parsed_data = [parsing_file(string) for string in data]

df = pd.DataFrame(parsed_data)

# Display the DataFrame
print(df)