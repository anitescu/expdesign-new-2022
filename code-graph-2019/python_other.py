import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

# with open("data/p_3vr_0.01/maxVio.txt") as f:
#     content = f.readlines()
# maxVio = [x.strip().split() for x in content]
# maxVio = [[float(x) for x in y] for y in maxVio]
# label = ["c=2", "c=4", "c=8", "c=16"]
# for i in range(4):
#     plt.plot(np.arange(5,32,5), maxVio[i], marker = "o", label = label[i])
# plt.legend()
# plt.xlabel("problem size")
# plt.ylabel("KKT condition violation")
# plt.grid(linestyle = '--')
# plt.savefig("maxVio.png")

# with open("data/p_3vr_0.01/dval.txt") as f:
#     content = f.readlines()
# dval = [x.strip().split() for x in content]
# dval = [[float(x) for x in y] for y in dval]
# label = ["c=2", "c=4", "c=8", "c=16"]
# for i in range(4):
#     plt.plot(np.arange(5,32,5), dval[i], marker = "o", label = label[i])
# plt.legend()
# plt.xlabel("problem size")
# plt.ylabel("approximation error in objective value")
# plt.grid(linestyle = '--')
# plt.savefig("dval.png")

# t = open("data/p_3vr_0.01/t.txt", "r").read().split()
# t = [float(x) for x in t]
# # t_sqp = open("data/p_3vr_0.01/y_tsqp.txt", "r").read().split()
# # t_sqp = [float(x) for x in t_sqp]
# with open("data/p_3vr_0.01/y_tsqp.txt") as f:
#     content = f.readlines()
# content = [x.strip().split() for x in content]
# t_sqp = [[float(x) for x in y] for y in content]
# print(t_sqp)
# label = ["SQP with c = 2", "SQP with c = 4", "SQP with c = 8", "SQP with c = 16"]
# plt.semilogy(np.arange(5,32,5), t, marker = "o", label = "Exact method with Ipopt solver")
# for i in range(4):
#     plt.semilogy(np.arange(5,32,5), t_sqp[i + 1], marker = "o", label = label[i])
# plt.xlabel("problem size")
# plt.ylabel("computation time (s)")
# plt.legend()
# plt.grid(linestyle = '--')
# plt.savefig("time.png")

t = open("/Users/jingyu621/Downloads/gravity_tr/t.txt", "r").read().split()
t = [float(x) for x in t]
with open("/Users/jingyu621/Downloads/gravity_tr/t_sqp.txt") as f:
    content = f.readlines()
content = [x.strip() for x in content]
t_sqp = [float(x) for x in content]
print(np.arange(5, 12), t[4:], t_sqp[4:])
plt.semilogy(np.arange(5, 12), t[4:], marker = "o", label = "Exact method with Ipopt solver")
plt.semilogy(np.arange(5, 12), t_sqp[4:], marker = "o", label = "SQP")
plt.xlabel(r"$\#$ cadidate loations = $2^x$")
plt.ylabel("computation time (s)")
plt.legend()
plt.grid(linestyle = '--')
plt.savefig("time.png")
