from cure import *
import sys,time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd

# The number of representative points
numRepPoints = 5
# Shrink factor
alpha = 0.1
# Desired cluster number
numDesCluster = 4

start = time.process_time()
data_set = pd.read_csv('campaign2.csv')
data = data_set.iloc[:,0:2].values
Label_true = data_set.iloc[:,2].values
print("Please wait for CURE clustering to accomplete...")
Label_pre = runCURE(data, numRepPoints, alpha, numDesCluster)
print("The CURE clustering is accompleted!!\n")
end = time.process_time()
print("The time of CURE algorithm is", end - start, "\n")
# Compute the NMI
nmi = metrics.v_measure_score(Label_true, Label_pre)
print("NMI =", nmi)

# Plot the result
plt.subplot(121)
plt.scatter(data_set.iloc[:, 0].values, data_set.iloc[:, 1].values, marker='.')
plt.text(0, 0, "origin")
plt.subplot(122)
scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown', 'cyan', 'brown',
                 'chocolate', 'darkgreen', 'darkblue', 'azure', 'bisque']
for i in range(data_set.shape[0]):
    color = scatterColors[Label_pre[i]]
    plt.scatter(data_set.iloc[i, 0], data_set.iloc[i, 1], marker='o', c=color)
plt.text(0, 0, "clusterResult")
plt.show()