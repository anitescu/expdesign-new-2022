import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import re

with open("y_sqp_5_bfgs.txt") as f:
    content = f.readlines()
content = [x.strip() for x in content]
# print(content, len(content), content[-1])
xtmp = [part.strip() for part in re.split('[\[\]]', content[-1]) if part.strip()]
x = xtmp[-1].split(',')
# x = [s.strip() for s in xtmp[-1].split(',')]
# x = content[-1].split()
# x = open("y_80.txt", "r").read().split()
# print(x, len(x))
x = np.array([float(i) for i in x])
n = len(x)
print(x)
angle = np.linspace(2 * np.pi / n, 2 * np.pi, n)
radii = np.ones(n)
width = 2 * np.pi / n * np.ones(n)
cmap = plt.cm.rainbow
norm = matplotlib.colors.Normalize(vmin = - 100, vmax = 100)
ax = plt.subplot(111, projection = 'polar')
# bars = ax.bar(angle, radii, width = width, bottom = 0.0, color = cmap(norm(x)))
bars = ax.bar(angle, radii, width = width, bottom = 0.0)
for r, bar in zip(x, bars):
    # print(r)
    bar.set_facecolor(str(1 - r))
    # bar.set_alpha(0)

plt.show()
