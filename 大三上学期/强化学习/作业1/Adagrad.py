import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


x_data = [338.,333.,328.,207.,226.,25.,179.,60.,208.,606.]
y_data = [640.,633.,619.,393.,428.,27.,193.,66.,226.,1591]

b = -150
w = 0
lr = 1
iteration = 10000
lr_b = 0
lr_w = 0
e = 1e-6 
s_w = 0
s_b = 0

batch_size = 10
batch_num = math.ceil(len(x_data)/batch_size)
data = []
batch_data = []

for i in range(len(x_data)):
    tmp = [x_data[i],y_data[i]]
    batch_data.append(tmp)
    if len(batch_data) == batch_size:
        data.append(batch_data)
        batch_data = []

if len(batch_data) != batch_size and len(batch_data) != 0:
    data.append(batch_data)
    batch_data = []

w_list = [float]*iteration
b_list = [float]*iteration


for i in range(iteration):
    
    for j in data:
        w_grad = 0
        b_grad = 0
        for k in j:
            y = k[0]*w + b
            w_grad += 2*(y - k[1])*k[0]
            b_grad += 2*(y - k[1])
        w_grad = w_grad/len(j)
        b_grad = b_grad/len(j)

        s_w = s_w + w_grad*w_grad
        lr_w = lr/(math.sqrt(s_w + e))
        s_b = s_b + b_grad*b_grad
        lr_b = lr/(math.sqrt(s_b + e))
        
        w = w - w_grad*lr_w
        b = b - b_grad*lr_b

    w_list[i] = w
    b_list[i] = b

# fig= plt.figure(dpi=500)
print(w)
print(b)
fig= plt.figure( )
plt.xlim(-200,-80)
plt.ylim(-4,4)

xmin, xmax = xlim = -200,-80
ymin, ymax = ylim = -4,4
ax = fig.add_subplot(111, xlim=xlim, ylim=ylim,
                     autoscale_on=False)
X = [ [4, 4],[4, 4],[4, 4],[1, 1]]
ax.imshow(X, interpolation='bicubic', cmap=cm.Spectral,
          extent=(xmin, xmax, ymin, ymax), alpha=1)
ax.set_aspect('auto')

plt.scatter(b_list,w_list,s=2,c='black',label=(lr,iteration))
plt.legend()
plt.show()