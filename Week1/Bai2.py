import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import random

tf.disable_v2_behavior()

def generate_dataset(size):
  # y = 3x + 5
  x_batch = np.linspace(0, 2, size)
  y_batch = 3 * x_batch + 5 +  np.random.randn(*x_batch.shape) * 0.2
  return (x_batch, y_batch)

def f(x):
  # Linear function
  return 3*x + 5

def linear_regression():
  # Input/Output node
  x = tf.placeholder(tf.float32, shape=(None, ), name='x')
  y = tf.placeholder(tf.float32, shape=(None, ), name='y')
  # Hidden node
  w = tf.Variable(np.random.normal(), name='W')
  b = tf.Variable(np.random.normal(), name='b')
  # loss function
  pred_y = tf.add(tf.multiply(w, x), b)
  loss = tf.reduce_mean(tf.square(pred_y - y))
  return (x, y, pred_y, loss)


data_x, data_y = generate_dataset(100)
x, y, pred_y, loss = linear_regression()
epochs = 50 # train times

# Cause our loss function calculate the distance between two result, we need to minimize it
optimizer = tf.train.GradientDescentOptimizer(0.1)
train_op = optimizer.minimize(loss)

with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  feed_dict = {x: data_x, y: data_y}
  for steps in range(epochs):
    _ = session.run(train_op, feed_dict)
    print(steps+1, "Loss val:", loss.eval(feed_dict))
  pred_y_final = session.run(pred_y, {x: data_x})

  # Show some result
  x_inp = np.array([1, 7, 3.5, 3.141592])
  y_out_correct = np.array([f(i) for i in x_inp])
  y_out_pred = session.run(pred_y, {x: x_inp})
  print("Input value:".ljust(18, " "), x_inp)
  print("Expected result:".ljust(18, " "), y_out_correct)
  print("Predict result:".ljust(18, " "), y_out_pred)

# plot data
print("Ploting the result")
y_correct = [f(i) for i in data_x]
plt.plot(data_x, pred_y_final, color="red", linewidth=4)
plt.plot(data_x, y_correct, color="green", linewidth=4)
plt.scatter(data_x, data_y)
plt.savefig("2.png")


