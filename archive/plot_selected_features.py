import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

baseDir = "handMovement/Database"
os.chdir(baseDir)
base_directory = os.getcwd()


def average_of_list(array) -> float:
    list_to_use = np.array(array)

    return np.sum(list_to_use, 0) / len(list_to_use)


mrmr, fisher, dm, dm_datafold, random, relief = [], [], [], [], [], []
for directory in range(1, 2):
    print("directory ", directory)
    # Load the Data
    os.chdir(os.path.join(base_directory, "handmovement8", str(directory)))
    with open("mrmr_ga_before", 'rb') as file:
        mrmr.append(pickle.load(file))
    with open("fisher_ga_before", 'rb') as file:
        fisher.append(pickle.load(file))
    with open("dm_ga_before", 'rb') as file:
        dm.append(pickle.load(file))
    with open("dm_datafold_ga_before", 'rb') as file:
        dm_datafold.append(pickle.load(file))
    with open("random_ga_before", 'rb') as file:
        random.append(pickle.load(file))
    with open("relief_ga_before", 'rb') as file:
        relief.append(pickle.load(file))

mrmr_avg = average_of_list(mrmr)
fisher_avg = average_of_list(fisher)
dm_avg = average_of_list(dm)
dm_datafold_avg = average_of_list(dm_datafold)
random_avg = average_of_list(random)
relief_avg = average_of_list(relief)

# miniRocket section
# filename_train = os.path.abspath(".") + "\\osuleaf_train"
# filename_test = os.path.abspath(".") + "\\osuleaf_test"

# with open("mrmr_selected", 'rb') as file:
#     mrmr_selected = pickle.load(file)
# with open("fisher_selected", 'rb') as file:
#     fisher_selected = pickle.load(file)
# with open("dm_selected", 'rb') as file:
#     dm_selected = pickle.load(file)
# with open("random_selected", 'rb') as file:
#     random_selected = pickle.load(file)
# with open("relief_selected", 'rb') as file:
#     relief_selected = pickle.load(file)

# with open("dm_coordinates", 'rb') as file:
#     dm_coordinates = pickle.load(file)
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

# ploting
# fig, ax = plt.subplots()
# plt.plot(x1, mrmr, label='mrmr')
# plt.plot(x1, fisher, label='fisher')
# plt.plot(x1, relief, label='relief')
# plt.plot(x1, random, label='random')
# plt.plot(x1, dm, label='dm')
# plt.title("handmovement ga before all")

# plt.legend()
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_title('Multiple Datasets Scatter Plot')
# plt.show()

fig, ax = plt.subplots()
plt.plot(x1, mrmr_avg, label='mrmr')
plt.plot(x1, fisher_avg, label='fisher')
plt.plot(x1, relief_avg, label='relief')
plt.plot(x1, random_avg, label='random')
plt.plot(x1, dm_avg, label='dm')
plt.plot(x1, dm_datafold_avg, label='dm_datafold')
ax.set_title("handmovement ga before all")

plt.legend()
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
# ax.set_title('Multiple Datasets Scatter Plot')
plt.show()
