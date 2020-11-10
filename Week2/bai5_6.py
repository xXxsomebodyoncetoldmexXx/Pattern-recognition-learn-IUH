import matplotlib.pyplot as plt
import numpy as np
from math import sqrt


mean, sigma, size = 5, sqrt(3), int(1e6)
x = np.random.normal(loc=mean, scale=sigma, size=size)
plt.hist(x, density=True, bins=100, color="red", alpha=0.75, label="Cau 5")

mean, sigma, size = 2, sqrt(1.5), int(1e6)
x = np.random.normal(loc=mean, scale=sigma, size=size)
plt.hist(x, density=True, bins=100, color="green", alpha=0.75, label="Cau 6")
plt.ylabel = "Probability"
plt.xlabel = "Data"
plt.legend(loc="best")
plt.show()
