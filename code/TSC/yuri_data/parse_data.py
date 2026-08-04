import os
from typing import Dict
import numpy as np
import pandas as pd
import scipy.io as sio
import re
import paths  # noqa: F401  # TSC path config

os.chdir(paths.HAND_MOVEMENT_DIR)  # legacy yuri_data output; override TSC_DATA_DIR if needed

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
    metadata = pd.DataFrame()
    parts = string.split('.')
    timestamp = parts[2]
    mat = sio.loadmat(f'{DIRECTORY}/{string}')
    vec = np.hstack(mat["W"]) # BHE - east, BHN - north, BHZ - , 
    station = find_station(timestamp)
    metadata = df_hrfi[df_hrfi["YYYYMMDDHHMiMi"] == int(timestamp)]
    metadata = metadata.reset_index(drop=True)
    return {'Station': station,
            'Axis': parts[1],
            'Timestamp': timestamp,
            'Vector': [vec],
            'Second': metadata.at[0, "second"],
            'Orid': metadata.at[0, "Orid"],
            'EtimeB': metadata.at[0, "EtimeB"],
            'LatB': metadata.at[0 ,"LatB"],	
            'LonB': metadata.at[0, "LonB"],
            'DepthB': metadata.at[0, "DepthB"],
            'Md': metadata.at[0, "Md"],
            'TypeB': metadata.at[0, "TypeB"],
            'HRFI_Ponset-OriginTime': metadata.at[0, "HRFI_Ponset-OriginTime"]
            }

def find_station(timestamp):
    for station in helper:
        tempdf = helper[station]
        if int(timestamp) in tempdf["YYYYMMDDHHMiMi"].values:
            return station
    return "HRFI"

parsed_data = [parsing_file(string) for string in data]

# parsed_data = parsing_file(string)

df = pd.DataFrame(parsed_data)

# Display the DataFrame
print(df)