
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import SubplotZero

with open('C:\\temp\Hough\data1.txt','r') as myDataFile:
#myDataFile = open('C:\\temp\Hough\data1.txt','r')
    f=myDataFile.readlines()

accumulator = []

for i in range(0,len(f)):   
    f[i] = f[i].replace(',','.').strip("\n")
    f[i] = f[i].split('\t')
    accumulator.append((np.radians(float(f[i][0])),float(f[i][1])))   

a=[]
b=[]
theta=[]
rho=[]

N=len(accumulator)
for i in range(0,N):
    theta.append(accumulator[i][0])
    rho.append(accumulator[i][1])

fig = plt.figure(1)
ax = SubplotZero(fig, 111)
#ax.set_facecolor((0, 0, 0))
fig.add_subplot(ax)
#ax.axhline(color="yellow")
#ax.axvline(color="yellow")
#ax.set_xticks([1])
#ax.set_yticks([1])
#ax.set_xticklabels(['x'])
#ax.set_yticklabels(['y'])
#ax.axis("equal")

for direction in ["xzero", "yzero"]:
    ax.axis[direction].set_axisline_style("-|>")
    ax.axis[direction].set_visible(True)
for direction in ["left", "right", "bottom", "top"]:
    ax.axis[direction].set_visible(False)

x = np.linspace(-60,60)

for i in range(0,60):
    ax.plot(x,-x*np.cos(theta[i])/np.sin(theta[i])+rho[i]/np.sin(theta[i]), linewidth=0.5, color='black')

plt.show()