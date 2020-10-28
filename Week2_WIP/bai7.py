import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.spatial import distance

mean, var, size = np.array([1, 3]), np.array([2, 2]), 250
x = np.random.normal(loc=mean[0], scale=sqrt(var[0]), size=size)
y = np.random.normal(loc=mean[1], scale=sqrt(var[1]), size=size)
plt.plot(x, y, "b.")

# x, y = np.random.multivariate_normal(mean, cov, size).T
# plt.plot(x, y, "r.")

# cov of this dataset
data_cov = np.cov(x, y)

# target
a = np.array([0, 0])
b = np.array([3, 4])
c = np.array([1, 2])

plt.plot(*a, "r.")
plt.plot(*b, "r.")
plt.plot(*c, "r.")

data_dist = list(zip(x, y))
inverse_data_cov = np.linalg.inv(data_cov)

mahalanobis_dist = distance.cdist(data_dist, np.array([a,b,c]), 'mahalanobis', VI=inverse_data_cov)

print("Mahalanobis distance:")
print(mahalanobis_dist)

# Make a grid [-10 10] x [-10 10] 
plt.plot(-10, -10, ",")
plt.plot(10, 10, ",")

plt.grid()
plt.show()
