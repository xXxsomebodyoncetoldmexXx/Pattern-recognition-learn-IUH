import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from math import sqrt

mean, var, size = 5, sqrt(3), 1000000
ar = Counter(np.random.normal(mean, var, size).round())
print(ar)
plt.bar(*zip(*ar.items()),1, color="blue", alpha=0.5, label="Cau 5")

mean, var, size = 2, sqrt(1.5), 1000000
ar = Counter(np.random.normal(mean, var, size).round())
print(ar)
plt.bar(*zip(*ar.items()), 0.9, color="red", alpha=0.5, label="Cau 6")
plt.legend(loc="best")
plt.show()
