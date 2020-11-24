import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Load data
columnsName = ['Feature1','Feature2']
classA = pd.read_csv("Data/classA.txt", sep="\t", names=columnsName)
classB = pd.read_csv("Data/classB.txt", sep="\t", names=columnsName)

# Merge
classAB = pd.concat([classA, classB], keys=['A', 'B']).reset_index().drop("level_1", axis=1).rename(columns = {'level_0': 'Class'})

# Plot dataset on Oxy planed
sns.set()
sns.FacetGrid(classAB, hue="Class", height=7).map(plt.scatter,"Feature1","Feature2",).add_legend()
plt.title('Đồ thị biểu diễn dữ liệu')
plt.savefig("2a.png")
plt.show()

# Split train test
train_set, test_set = train_test_split(classAB, train_size=0.7)
print("Số lượng tập train:", len(train_set))
print("Số lượng tập test :", len(test_set))

# def dist(x, y):
#   return np.sqrt(np.sum(np.power(x-y, 2)))

# def phi(x, y, h):
#   if(dist(x, y)/h > 0.5):
#     return False
#   return True

def phi(x, y, h):
  return np.exp((-np.transpose(x-y)@(x-y))/(2*np.power(h, 2)))

def pw(X, data_set, h):
  score_list = list()
  for p in data_set.groupby("Class"):
    k = 0
    n = p[1].shape[0]   # Get number of feature
    d = len(p[1].shape) # Get dimenstino
    for x in p[1][['Feature1','Feature2']].to_numpy():
      k += phi(X, x, h)
    score_list.append(k/(n * (h**d)))
  return np.argmax(score_list)

# diameter of the hypercube
h = 1

# Evaluate
# confusion matrix to precision + recall
def cm2pr_binary(cm):
  p = cm[0,0]/np.sum(cm[:,0])
  r = cm[0,0]/np.sum(cm[0])
  return (p, r)

predict = ["A" if pw(np.array([x, y]), train_set, h) == 0 else "B" for x, y in test_set[["Feature1", "Feature2"]].values]
label = list(test_set["Class"].values)
cf_mtx = confusion_matrix(label, predict)
print(pd.DataFrame(cf_mtx, index=["Class A", "Class B"], columns=["Class A predict", "Class B predict"]))
p, r = cm2pr_binary(cf_mtx)
print("Precition = {0:.2f}, Recall = {1:.2f}".format(p, r))
# exit(0)
# Plot result
#Plot with boundary contours
N = 100
X = np.linspace(-8, 10, N)
Y = np.linspace(-1, 10, N)
X, Y = np.meshgrid(X, Y)

#Configure plot
color_list = ['Blues','Reds']
g = sns.FacetGrid(test_set, hue="Class", height=10, palette = 'colorblind', hue_order=["A","B"]).map(plt.scatter,"Feature1","Feature2",).add_legend()
my_ax = g.ax

#Computing the predicted class function for each value on the grid
zz = np.array( [pw(np.array([xx,yy]), train_set, h) for xx, yy in zip(np.ravel(X), np.ravel(Y)) ] )

#Reshaping the predicted class into the meshgrid shape
Z = zz.reshape(X.shape)

#Plot the filled and boundary contours
my_ax.contourf( X, Y, Z, 1, alpha = .1, colors = ('blue','red'))
my_ax.contour( X, Y, Z, 1, alpha = 1, colors = ('blue','red'))

# Addd axis and title
my_ax.set_xlabel('Feature1')
my_ax.set_ylabel('Feature2')
my_ax.set_title('Biên phân lớp dựa trên Parzen window')
plt.savefig("2b.png")
plt.show()
