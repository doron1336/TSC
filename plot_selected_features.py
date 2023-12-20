import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

# Load the Data
os.chdir(r"C:\Users\doron\OneDrive\Desktop\thesis\TSC\handMovement\Database\osuleaf")

# miniRocket section
# filename_train = os.path.abspath(".") + "\\osuleaf_train"
# filename_test = os.path.abspath(".") + "\\osuleaf_test"

with open("mrmr_selected", 'rb') as file:
    mrmr_selected = pickle.load(file)
with open("fisher_selected", 'rb') as file:
    fisher_selected = pickle.load(file)
with open("dm_selected", 'rb') as file:
    dm_selected = pickle.load(file)
with open("random_selected", 'rb') as file:
    random_selected = pickle.load(file)
with open("relief_selected", 'rb') as file:
    relief_selected = pickle.load(file)

with open("dm_coordinates", 'rb') as file:
    dm_coordinates = pickle.load(file)
with open("avg_jm", 'rb') as file:
    avg_jm = pickle.load(file)

with open("mrmr", 'rb') as file:
    mrmr = pickle.load(file)
with open("fisher_ga", 'rb') as file:
    fisher = pickle.load(file)

with open("dm_ga", 'rb') as file:
    dm = pickle.load(file)
with open("random_ga", 'rb') as file:
    random = pickle.load(file)
with open("relief_ga", 'rb') as file:
    relief = pickle.load(file)

with open("dm_coordinates", 'rb') as file:
    dm_coordinates = pickle.load(file)
with open("avg_jm", 'rb') as file:
    avg_jm = pickle.load(file)

x1 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,
      120, 130, 140, 150, 160, 170, 180, 190, 200]

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)

# print(sorted_indices)
# Calculate the index corresponding to the q percentile
# sequence_containing_x_vals = dm_coordinates[:, 0]
# sequence_containing_y_vals = dm_coordinates[:, 1]

# sc = ax.scatter(sequence_containing_x_vals,
#                 sequence_containing_y_vals, c=avg_jm, cmap='viridis')
# plt.colorbar(sc)
# ax.scatter(sequence_containing_x_vals[relief_selected],
#            sequence_containing_y_vals[relief_selected], c='red', label='Selected Indices')
# plt.title('OSULEAF : relief selected features')
# plt.legend()
# plt.show()

fig, ax = plt.subplots()
plt.plot(x1, mrmr, label='mrmr')
plt.plot(x1, fisher, label='fisher')
plt.plot(x1, relief, label='relief')
plt.plot(x1, random, label='random')
plt.plot(x1, dm, label='dm')
plt.title("handmovement ga before all")

plt.legend()
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Multiple Datasets Scatter Plot')
plt.show()
# print(dm)
# print(mrmr)
