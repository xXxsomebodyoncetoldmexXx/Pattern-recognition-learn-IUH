import numpy as np
import matplotlib.pyplot as plt
import random

def gen_data(size):
  # y = 2x + 3
  x = np.linspace(-10, 10, size)
  y = list()
  for val in x:
    y.append(2*val + 3 + random.randrange(-3, 3))
  y = np.array(y)
  return (x, y)

x, y = gen_data(100)

# plot data
plt.scatter(x, y)
plt.show()

