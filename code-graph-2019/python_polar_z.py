import numpy as np
import matplotlib.pyplot as plt

with open("y_80.txt") as f:
    content = f.readlines()
# x = open("x.txt", "r").read().split()
# print(content)
content = [x.strip() for x in content]
# print(content, len(content), content[-1])
x = content[-1].split()
print(x, len(x))
x = np.array([int(float(i)) for i in x])
n = len(x)
m = int(sum(x))
angle = np.linspace(2 * np.pi / n, 2 * np.pi, n)
vec = np.multiply(angle, x)
theta = vec[np.nonzero(vec)]
radii = np.ones(m)
width = 2 * np.pi / n * np.ones(m)
ax = plt.subplot(111, projection = 'polar')
bars = ax.bar(theta, radii, width = width, bottom = 0.0, color = '0')

plt.show()
