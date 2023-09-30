import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

# Load the Data
os.chdir(r"C:\Users\doron\OneDrive\Desktop\thesis\TSC\handMovement\Database\osuleaf")

# miniRocket section
filename_train = os.path.abspath(".") + "\\osuleaf_train"
filename_test = os.path.abspath(".") + "\\osuleaf_test"

with open("mrmr", 'rb') as file:
    mrmr = pickle.load(file)
with open("fisher", 'rb') as file:
    fisher = pickle.load(file)
with open("dm", 'rb') as file:
    dm = pickle.load(file)
with open("random", 'rb') as file:
    random = pickle.load(file)
with open("relief", 'rb') as file:
    relief = pickle.load(file)

x1 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,
      120, 130, 140, 150, 160, 170, 180, 190, 200]
# fig, ax = plt.subplots()
plt.plot(x1, mrmr, label='mrmr')
plt.plot(x1, fisher, label='fisher')
plt.plot(x1, relief, label='relief')
plt.plot(x1, random, label='random')
plt.plot(x1, dm, label='dm')

plt.legend()
# plt.set_xlabel('X-axis')
# plt.set_ylabel('Y-axis')
# plt.set_title('Multiple Datasets Scatter Plot')
plt.show()
# print(dm)
# print(mrmr)
