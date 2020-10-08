import numpy as np

def mean(arr):
    if len(arr.shape) == 1:
        return sum(arr)/len(arr)
    return sum(mean(xi) for xi in arr)/len(arr)

def var(arr):
    m = mean(arr)
    return sum((xi - m)**2 for xi in arr)/len(arr)
# Cau 3
a = np.array([1, 2, 4, 6, 9, 10, 20, 7])
print("mean(a):", mean(a), mean(a)==a.mean())
print("var(a) :", var(a), var(a)==a.var())
print("---------------------")

b = np.arange(0, 101, 2)
print("mean(b):", mean(b), mean(b)==b.mean())
print("var(b) :", var(b), var(b)==b.var())
print("---------------------")

c = np.square(np.arange(1, 100, 2))
print("mean(c):", mean(c), mean(c)==c.mean())
print("var(c) :", var(c), var(c)==c.var())
print("---------------------")

d = np.array([[2,4],[3,7],[4,6],[5,5],[2,3]])
print("mean(d):", mean(d), mean(d)==d.mean())
print("var(d) :", d.var(), "Unknow")