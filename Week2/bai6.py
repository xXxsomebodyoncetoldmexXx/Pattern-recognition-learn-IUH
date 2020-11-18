import matplotlib.pyplot as plt
import numpy as np
import math

def gauss(x, mean, var):
  return (1 / (var * math.sqrt(2*math.pi)) * math.exp(-0.5 * ((x-mean)/var)**2))

mean = 2
var = 1.5
x = np.arange(0, 10, 0.01)
y = [gauss(i, mean, var) for i in x]
plt.plot(x, y)
plt.title(f"Gaussian ($\mu$= {mean}), $\sigma$= {var}")
plt.savefig("Week2/6.png")
plt.show()
