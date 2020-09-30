import numpy as np
import matplotlib.pyplot as plt 
from collections import Counter

# A group
x = np.random.normal(-5, 5, size=100)
y = np.random.normal(5, 5, size=100)
plt.scatter(x, y, c='g', marker='^', label="A group")
groupA = np.array([(i, j) for i, j in zip(x, y)])

# B group
x = np.random.normal(5, 5, size=100)
y = np.random.normal(-5, 5, size=100)
plt.scatter(x, y, c='r', marker='v', label="B group")
groupB = np.array([(i, j) for i, j in zip(x, y)])

# Guess line
x = np.linspace(-10, 10)
y = 2*x + 3
plt.plot(x, y)

plt.legend(loc='best')
plt.show()
