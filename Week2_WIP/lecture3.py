import matplotlib.pyplot as plt 
import numpy as np 

in_time = [(0,27), (1,25), (2,16), (3,19), (4,26), (5,20), (6,19), (7,17), (8,10), (9,5), (10,4), (11,4), (12,2)]
cls_late = [(5,3), (6,5), (7,8), (8,15), (9,17), (10,18), (11,19), (12,16), (13,9), (14,8), (15,8)]

X,Y = zip(*in_time)
X2,Y2 = zip(*cls_late)

bar_width = 0.9

plt.bar(X,Y,bar_width,color="blue",alpha=0.75,label="Dung gio")

bar_width = 0.8

plt.bar(X2,Y2,bar_width,color="red",alpha=0.75,label="Tre")

plt.legend(loc='upper right')

plt.show()

in_time_dict = dict(in_time)
too_late_dict = dict(cls_late)

def catch_the_train(min):
    s = in_time_dict.get(min,0)
    if s == 0:
        return 0
    else:
        m = too_late_dict.get(min,0)
        return s/(s+m)
for minutes in range(0,15):
    print(minutes,catch_the_train(minutes))

