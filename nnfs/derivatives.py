import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 2*x**2

x = np.array(range(5))
y = f(x)

print((y[1]-y[0]) / (x[1]-x[0]))
print((y[3]-y[2]) / (x[3]-x[2]))
