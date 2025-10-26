import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 2*x**2

x = np.arange(0, 5, 0.001)
y = f(x)

plt.plot(x, y)

p2_delta = 0.0001
x1 = 2
x2 = x1 + p2_delta # add delta

y1 = f(x1) # result at the derivation point
y2 = f(x2) # result at the derivation point

print((x1, y1), (x2, y2))

# Derivative approximation and y-intercept for the tangent line
approximate_derivative = (y2-y1)/(x2-x1)
b = y2 - approximate_derivative*x2
