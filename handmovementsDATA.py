import pickle
import scipy.io as sio
import os
from os.path import dirname, join as pjoin
import numpy as np
import pandas as pd


def GetHandMovementDATA():
    os.chdir(r"C:\Users\doron\OneDrive\Desktop\thesis\TSC\handMovement\Database")

    mat = sio.loadmat('male_1.mat')
    mat_to_pd = pd.Series(mat)
    mat_to_pd = pd.DataFrame(
        {'label': mat_to_pd.index, 'list': mat_to_pd.values}).iloc[3:,].reset_index(drop=True)

    # The data is a dictionary, each key is a channel of hand movement and represents a matrix of 30X3000.

    label_dictionary = {1: "cyl", 2: "hook",
                        3: "tip", 4: "palm", 5: "spher", 6: "lat"}

    label = np.repeat([1], 30)
    cyl_ch1_df = pd.DataFrame(mat['cyl_ch1'])
    cyl_ch2_df = pd.DataFrame(mat['cyl_ch2'])
    cyl_final = pd.concat([cyl_ch1_df, cyl_ch2_df], axis=1)
    cyl_final['label'] = label
    # cyl_ch1_df['label'] = label
    train_cyl = cyl_final.iloc[:25]
    test_cyl = cyl_final.iloc[25:]

    label = np.repeat([2], 30)
    hook_ch1_df = pd.DataFrame(mat['hook_ch1'])
    hook_ch2_df = pd.DataFrame(mat['hook_ch2'])
    hook_final = pd.concat([hook_ch1_df, hook_ch2_df], axis=1)
    hook_final['label'] = label
    # hook_ch1_df['label'] = label
    train_hook = hook_final.iloc[:25]
    test_hook = hook_final.iloc[25:]

    label = np.repeat([3], 30)
    tip_ch1_df = pd.DataFrame(mat['tip_ch1'])
    tip_ch2_df = pd.DataFrame(mat['tip_ch2'])
    tip_final = pd.concat([tip_ch1_df, tip_ch2_df], axis=1)
    tip_final['label'] = label
    # tip_ch1_df['label'] = label
    train_tip = tip_final.iloc[:25]
    test_tip = tip_final.iloc[25:]

    label = np.repeat([4], 30)
    palm_ch1_df = pd.DataFrame(mat['palm_ch1'])
    palm_ch2_df = pd.DataFrame(mat['palm_ch2'])
    palm_final = pd.concat([palm_ch1_df, palm_ch2_df], axis=1)
    palm_final['label'] = label
    # palm_ch1_df['label'] = label
    train_palm = palm_final.iloc[:25]
    test_palm = palm_final.iloc[25:]

    label = np.repeat([5], 30)
    spher_ch1_df = pd.DataFrame(mat['spher_ch1'])
    spher_ch2_df = pd.DataFrame(mat['spher_ch2'])
    spher_final = pd.concat([spher_ch1_df, spher_ch2_df], axis=1)
    spher_final['label'] = label
    # spher_ch1_df['label'] = label
    train_spher = spher_final.iloc[:25]
    test_spher = spher_final.iloc[25:]

    label = np.repeat([6], 30)
    lat_ch1_df = pd.DataFrame(mat['lat_ch1'])
    lat_ch2_df = pd.DataFrame(mat['lat_ch2'])
    lat_final = pd.concat([lat_ch1_df, lat_ch2_df], axis=1)
    lat_final['label'] = label
    # lat_ch1_df['label'] = label
    train_lat = lat_final.iloc[:25]
    test_lat = lat_final.iloc[25:]

    train_set = pd.concat([train_cyl, train_hook, train_tip,
                           train_palm, train_spher, train_lat], axis=0)

    test_set = pd.concat([test_cyl, test_hook, test_tip,
                          test_palm, test_spher, test_lat], axis=0)

    train_set = train_set.sample(frac=1)  # shuffles the rows
    # print(train_set)
    x_train = train_set.iloc[:, :3000].to_numpy().astype(np.float32)
    y_train = train_set["label"].to_numpy()

    x_test = test_set.iloc[:, :3000].to_numpy().astype(np.float32)
    y_test = test_set["label"].to_numpy()
    # print(type(x_train))
    # print(x_train.shape)
    # print(type(y_train))
    # print(y_train.shape)
    y_test = [int(i) for i in y_test]
    if np.min(y_test) != 0:
        y_test = [str(int(i)-1) for i in y_test]
        y_train = [str(int(i)-1) for i in y_train]

    return x_train, y_train, x_test, y_test
