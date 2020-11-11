import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import scipy.stats

sns.set()
iris = sns.load_dataset("iris")
iris = iris.rename(index = str, columns = {'sepal_length':'1_sepal_length','sepal_width':'2_sepal_width','petal_length':'3_petal_length', 'petal_width':'4_petal_width'})
print(iris)
sns.FacetGrid(iris, hue="species", height=7).map(plt.scatter,"1_sepal_length","2_sepal_width", ).add_legend()
plt.title('Scatter plot')
# plt.show()

def predict_NB_gaussian_class(X, mu_list, std_list, pi_list):
  scores_list = list()
  classes = len(mu_list)
  for p in range(classes):
    score = (scipy.stats.norm.pdf(x = X[0], loc = mu_list[p][0], scale = std_list[p][0] ) * scipy.stats.norm.pdf(x = X[1], loc = mu_list[p][1], scale = std_list[p][1] ) * pi_list[p])
    scores_list.append(score)
  return np.argmax(scores_list)

def predict_Bayes_class(X,mu_list,sigma_list):
  scores_list = []
  classes = len(mu_list)
  for p in range(classes):
    score = scipy.stats.multivariate_normal.pdf(X, mean=mu_list[p],cov=sigma_list[p])
    scores_list.append(score)
  return np.argmax(scores_list)

df1 = iris[["1_sepal_length", "2_sepal_width",'species']]

mu_list = [np.ravel(x) for x in np.split(df1.groupby('species').mean().values,[1,2])]
std_list = [np.ravel(x) for x in np.split(df1.groupby('species').std().values,[1,2])]
pi_list = df1.iloc[:,2].value_counts().values / len(df1)

N = 100
X = np.linspace(4, 8, N)
Y = np.linspace(1.5, 5, N)
X, Y = np.meshgrid(X, Y)

color_list = ['Blues','Greens','Reds']
g = sns.FacetGrid(iris, hue="species", height=10, palette = 'colorblind').map(plt.scatter, "1_sepal_length", "2_sepal_width",).add_legend()
my_ax = g.ax

zz_NB = np.array( [predict_NB_gaussian_class( np.array([xx,yy]).reshape(-1,1), mu_list, std_list, pi_list) for xx, yy in zip(np.ravel(X), np.ravel(Y)) ] )
zz_Bayes = np.array( [predict_Bayes_class( np.array([xx,yy]).reshape(-1,1), mu_list, std_list) for xx, yy in zip(np.ravel(X), np.ravel(Y)) ] )

Z1 = zz_NB.reshape(X.shape)
Z2 = zz_Bayes.reshape(X.shape)

#Plot the filled and boundary contours
my_ax.contourf( X, Y, Z1, 2, alpha = .1, colors = ('blue','green','red'))
my_ax.contour( X, Y, Z1, 2, alpha = 1, colors = ('blue','green','red'))
# ---
# my_ax.contourf( X, Y, Z2, 2, alpha = .1, colors = ('blue','green','red'))
# my_ax.contour( X, Y, Z2, 2, alpha = 1, colors = ('blue','green','red'))
# ---
my_ax.set_xlabel('Sepal length')
my_ax.set_ylabel('Sepal width')
my_ax.set_title('Gaussian Naive Bayes decision boundaries')
plt.show()
