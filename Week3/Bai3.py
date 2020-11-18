import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import scipy.io
from sklearn.metrics import confusion_matrix

# Load data
columnsName = ['Feature1','Feature2']
train1 = pd.DataFrame(scipy.io.loadmat('./Week3/Data/class1_train.mat')['class1_train'],columns = columnsName)
test1 = pd.DataFrame(scipy.io.loadmat('./Week3/Data/class1_test.mat')['class1_test'],columns = columnsName)
train2 = pd.DataFrame(scipy.io.loadmat('./Week3/Data/class2_train.mat')['class2_train'],columns = columnsName)
test2 = pd.DataFrame(scipy.io.loadmat('./Week3/Data/class2_test.mat')['class2_test'],columns = columnsName)

# Merge data frame
train_set = pd.concat([train1, train2], keys=['1', '2']).reset_index().drop('level_1', axis=1).rename(columns = {'level_0': 'Class'})
test_set = pd.concat([test1, test2], keys=['1', '2']).reset_index().drop('level_1', axis=1).rename(columns = {'level_0': 'Class'})

# Concat into one large dataset to plot
data_set = pd.concat([train_set, test_set]).reset_index().drop("index", axis=1)

# Plot data
sns.set()
sns.FacetGrid(data_set, hue="Class", height=7).map(plt.scatter,"Feature1","Feature2",).add_legend()
plt.title('Scatter plot')
plt.savefig("Week3/3a.png")
plt.show()

print("Sỗ mẫu của Class 1 là:", len(train1) + len(test1))
print("Sỗ mẫu của Class 2 là:", len(train2) + len(test2))
print()
print("Số lượng tập train là:", len(train_set))
print("Số lượng tập test là:", len(test_set))
print()

# Use train_set to calculate parameter
mu_list = [np.ravel(x) for x in np.split(train_set.groupby('Class').mean().values,[1])]
cov_list = np.split(train_set.groupby('Class').cov().values,[2])
pi_list = train_set.iloc[:,0].value_counts().values / len(train_set)

print("Mean của từng đặc trưng trong Class 1 là:", mu_list[0])
print("Mean của từng đặc trưng trong Class 2 là:", mu_list[1])
print()
print("Cov của Class 1 là:")
print(cov_list[0])
print("Cov của Class 2 là:")
print(cov_list[1])
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

# confusion matrix to precision + recall
def cm2pr_binary(cm):
  p = cm[0,0]/np.sum(cm[:,0])
  r = cm[0,0]/np.sum(cm[0])
  return (p, r)

predict = ['1' if df(np.array([xx, yy]).reshape(-1, 1), mu_list, cov_list, pi_list)==0 else '2' for xx, yy in train_set[["Feature1","Feature2"]].values]
label = list(train_set["Class"].values)

cf_mtx = confusion_matrix(label, predict)
print(pd.DataFrame(cf_mtx, index=["Class 1", "Class 2"], columns=["Class 1 predict", "Class 2 predict"]))
p, r = cm2pr_binary(cf_mtx)
print("Precition = {0:.2f}, Recall = {1:.2f}".format(p, r))

N = 100
X = np.linspace(-3, 9, N)
Y = np.linspace(-9, 7, N)
X, Y = np.meshgrid(X, Y)

color_list = ['Blues','Reds']
g = sns.FacetGrid(test_set, hue="Class", height=10, palette = 'colorblind', hue_order=["1","2"]).map(plt.scatter,"Feature1","Feature2",).add_legend()
my_ax = g.ax

zz = np.array([df(np.array([xx,yy]).reshape(-1,1), mu_list, cov_list, pi_list) for xx, yy in zip(np.ravel(X), np.ravel(Y))])

Z = zz.reshape(X.shape)
my_ax.contourf( X, Y, Z, 1, alpha = .1, colors = ('blue','red'))
my_ax.contour( X, Y, Z, 1, alpha = 1, colors = ('blue','red'))

my_ax.set_xlabel('Feature1')
my_ax.set_ylabel('Feature2')
my_ax.set_title('Biên phân lớp dựa trên biệt hàm')
plt.savefig("Week3/3b.png")
plt.show()
