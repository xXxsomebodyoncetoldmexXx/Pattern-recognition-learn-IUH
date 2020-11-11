import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import scipy.stats
from sklearn.model_selection import train_test_split

# Load data
columnsName = ['Feature1','Feature2']
classA = pd.read_csv("Data/classA.txt", sep="\t", names=columnsName)
classB = pd.read_csv("Data/classB.txt", sep="\t", names=columnsName)

# Merge
classAB = pd.concat([classA, classB], keys=['A', 'B']).reset_index().drop("level_1", axis=1).rename(columns = {'level_0': 'Class'})

train_set, test_set = train_test_split(classAB, train_size=0.7)
