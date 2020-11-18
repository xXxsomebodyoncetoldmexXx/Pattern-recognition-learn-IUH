import numpy as np
import matplotlib.pyplot as plt

def gauss(x, mean, var):
  return (1 / (var * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x-mean)/var)**2))

def gauss2d(x, y, mean, var):
  return gauss(x, mean[0], var[0]) * gauss(y, mean[1], var[1])

def mahalanobis_dist(x, mean, cov):
  return np.sqrt(np.transpose(x - mean)@np.linalg.inv(cov)@(x - mean))

mean = np.array([1, 3])
var = np.array([2, 2])

x = np.linspace(-10, 10)
y = np.linspace(-10, 10)
x, y = np.meshgrid(x, y)
z = gauss2d(x, y, mean, var)
plt.contourf(x, y, z)
plt.show()

# x = np.array([0, 0])
# print(np.linalg.inv(np.cov(z)))
# print(mahalanobis_dist(x, mean, np.cov(mean, var)))
