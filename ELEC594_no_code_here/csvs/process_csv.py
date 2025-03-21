import pandas as pd
import glob
import numpy as np
# Get a sorted list of CSV files matching the pattern
n100 = np.loadtxt("capstone/parallelrm/csvs/times_7_100_ms.csv", delimiter=",")
n1000 = np.loadtxt("capstone/parallelrm/csvs/times_7_1000_ms.csv", delimiter=",")
n10000 = np.loadtxt("capstone/parallelrm/csvs/times_7_10000_ms.csv", delimiter=",")
n50000 = np.loadtxt("capstone/parallelrm/csvs/times_7_50000_ms.csv", delimiter=",")
n100000 = np.loadtxt("capstone/parallelrm/csvs/times_7_100000_ms.csv", delimiter=",")

combined = np.array([n100, n1000, n10000, n50000, n100000]).T

print(combined.shape)

print(combined)
csv = pd.DataFrame(combined, columns=["n=100", "n=1000", "n=10000", "n=50000", "n=100000"])
csv.to_csv("capstone/parallelrm/csvs/times_7_combined.csv", index=False)