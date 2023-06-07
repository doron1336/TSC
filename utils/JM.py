import numpy as np

dictionatyJM = {}


def computeJM(array1, array2, column):
    a = np.asarray(array1)
    b = np.asarray(array2)
    if np.array_equal(a, b):
        JM = 0
    else:
        firstArgument = 0.125*np.power((np.mean(a) - np.mean(b)), 2)
        secondArgument = weird_division(2, (np.var(a) + np.var(b)))
        if weird_division((np.var(a) + np.var(b)), (2*np.std(a)*np.std(b))) == 0:
            thirdArgument = 0
        else:
            thirdArgument = 0.5 * \
                (np.log((np.var(a) + np.var(b))/(2*np.std(a)*np.std(b))))

        B = firstArgument*secondArgument + thirdArgument
        JM = 2*(1 - np.exp(-B))

    return JM


def weird_division(n, d):
    return n / d if d else 0


def JM_matrix(feature, gt, i):
    column = i
    numOfClasses = len(np.unique(gt))
    matrix = np.zeros([numOfClasses, numOfClasses])
    array = []
    gt = np.asarray(gt)
    for i in range(0, numOfClasses, 1):
        loc = np.where(gt == str(i))
        if i > 0:
            dict(id=i-1, array=array)
        array = []
        for j in range(0, numOfClasses, 1):
            loc2 = np.where(gt == str(j))
            if ((i == j) or (i >= j)):
                continue
            else:
                a = np.asarray(feature)[loc]
                b = np.asarray(feature)[loc2]
                matrix[i][j] = computeJM(a, b, column)
                matrix[j][i] = matrix[i][j]
                array.append(matrix[i][j])

    return matrix


def generateJMVector(X_train_transform, gt):
    meanVec, maxVec = [], []
    for cls in range(len(np.unique(gt))):
        dictionatyJM[cls] = []
    for i in range(len(X_train_transform[0, :])):
        feature = np.asarray(X_train_transform[:, i])
        mat = JM_matrix(feature, gt, i)
        for j in range(mat.shape[0]):
            dictionatyJM[int(j)] = dictionatyJM[int(j)] + list(mat[j, :])
        meanVec.append(np.mean(mat))
        maxVec.append(np.max(mat))
        # there are 6 classes hence we have matrix of 6X6 that shows the separation
    return np.asarray(meanVec), np.asarray(maxVec), dictionatyJM


def JM_flat(X_train_transform, gt):
    flatArr = []
    meanArr = []
    for cls in range(len(np.unique(gt))):
        dictionatyJM[cls] = []
    for i in range(len(X_train_transform[0, :])):
        feature = np.asarray(X_train_transform[:, i])
        mat = JM_matrix(feature, gt, i)
        iu1 = np.triu_indices(mat.shape[0], k=1)
        flatten_upper_triangle = mat[iu1]
        flatArr.append(flatten_upper_triangle)
        meanArr.append(np.mean(flatten_upper_triangle))
    return np.asarray(flatArr), np.asarray(meanArr)
