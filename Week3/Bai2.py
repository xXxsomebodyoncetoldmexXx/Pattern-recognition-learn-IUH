import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import scipy.stats
import scipy.io
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load data
columnsName = ['Feature1','Feature2']
classA = pd.DataFrame(scipy.io.loadmat('./Data/classA.mat')['classA'],columns = columnsName)
classB = pd.DataFrame(scipy.io.loadmat('./Data/classB.mat')['classB'],columns = columnsName)

# Plot data
classAB = pd.concat([classA,classB], keys=['A', 'B']).reset_index().drop('level_1', axis=1).rename(columns = {'level_0': 'Class'})
sns.set()
sns.FacetGrid(classAB, hue="Class", height=7).map(plt.scatter,"Feature1","Feature2",).add_legend()
plt.title('Scatter plot')
# plt.show()

print("Sỗ mẫu của Class A là:", len(classA))
print("Sỗ mẫu của Class B là:", len(classB))
print()

# Split train data set and test set
train, test = train_test_split(classAB, train_size=0.8)
print("Số lượng tập train là:", len(train))
print("Số lượng tập test là:", len(test))
print()

mu_list = [np.ravel(x) for x in np.split(train.groupby('Class').mean().values,[1])]
cov_list = np.split(train.groupby('Class').cov().values,[2])
pi_list = train.iloc[:,0].value_counts().values / len(train)

print("Mean của từng đặc trưng trong Class A là:", mu_list[0])
print("Mean của từng đặc trưng trong Class B là:", mu_list[1])
print()

def df(X, mu_list, cov_list, pi_list):
  scores_list = list()
  classes = len(mu_list)
  for p in range(classes):
    Wi = (-1/2)*np.linalg.inv(cov_list[p])
    wi = np.linalg.inv(cov_list[p])@mu_list[p]
    wi0 = (-1/2)*np.transpose(mu_list[p])@np.linalg.inv(cov_list[p])@mu_list[p] + (-1/2)*np.log(np.linalg.norm(cov_list[p])) + np.log(pi_list[p])
    score = np.transpose(X)@Wi@X + np.transpose(wi)@X + wi0
    scores_list.append(score)
  return np.argmax(scores_list)

prediction = ["A" if df(np.array([x,y]).reshape(-1,1), mu_list, cov_list, pi_list)==0 else "B" for x, y in test[["Feature1","Feature2"]].values]
label = list(test["Class"].values)
print(pd.DataFrame(confusion_matrix(label, prediction), index=['Class A', 'Class B'], columns=['Class A Predict', 'Class B Predict']))

N = 100
X = np.linspace(-5, 5, N)
Y = np.linspace(-5, 5, N)
X, Y = np.meshgrid(X, Y)

color_list = ['Blues','Reds']
g = sns.FacetGrid(test, hue="Class", height=10, palette = 'colorblind', hue_order=["A","B"]).map(plt.scatter,"Feature1","Feature2",).add_legend()
my_ax = g.ax

zz = np.array( [df(np.array([xx,yy]).reshape(-1,1), mu_list, cov_list, pi_list) for xx, yy in zip(np.ravel(X), np.ravel(Y)) ] )

Z = zz.reshape(X.shape)

my_ax.contourf( X, Y, Z, 1, alpha = .1, colors = ('blue','red'))
my_ax.contour( X, Y, Z, 1, alpha = 1, colors = ('blue','red'))

my_ax.set_xlabel('Feature1')
my_ax.set_ylabel('Feature2')
my_ax.set_title('Biên phân lớp dựa trên phân phối Gauss')
plt.show()
