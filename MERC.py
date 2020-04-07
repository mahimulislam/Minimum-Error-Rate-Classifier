import sympy as sym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

df=pd.read_csv('test.txt',sep=",",header=None,dtype='Float64')
df_arr=df.values

xtrain=df_arr[:,0]
ytrain=df_arr[:,1]


fig = plt.figure()
ax=fig.add_subplot(projection="3d")

u1=np.asmatrix(([0,0]))
u2=np.asmatrix(([2,2]))
sig1=np.asmatrix(([[.25,.3],[.3,1]]))
sig2=np.asmatrix(([[.5,0],[0,.5]]))
sig1inv=(sig1**(-1))
sig2inv=(sig2**(-1))
u1t=np.transpose(u1)
u2t=np.transpose(u2)
flagc1=0
flagc2=0
dimval=2
classtrain=[]
for i in range(0,len(xtrain)):

    xtval=np.transpose(np.asmatrix(([xtrain[i], ytrain[i]])))
    xval=np.asmatrix(([xtrain[i], ytrain[i]]))

    part1=np.dot(np.dot(xval,sig1inv),xtval)
    part2=np.dot(np.dot(u1,sig1inv),u1t)
    part3=2*np.dot(np.dot(u1,sig1inv),xtval)

    ndf1 = ((-0.5) * (float(part1)+float(part2)-float(part3)))-(dimval/2*np.log(2*3.1416))-(0.5*np.log(np.linalg.det(sig1)))

    part4 = np.dot(np.dot(xval, sig2inv), xtval)
    part5 = np.dot(np.dot(u2, sig2inv), u2t)
    part6 = 2 * np.dot(np.dot(u2, sig2inv), xtval)

    ndf2 = ((-0.5) * (float(part4) + float(part5) - float(part6))) - (dimval / 2 * np.log(2 * 3.1416)) - (
                0.5 * np.log(np.linalg.det(sig2)))

    if((ndf1)>(ndf2)):
        classtrain.append(1)
        if flagc1 == 0:
            ax.scatter(xtrain[i], ytrain[i], s=20, c='r', marker='o', label='class1 train')
            flagc1 = 1
        else:
            ax.scatter(xtrain[i], ytrain[i], s=20, c='r', marker='o')
    else:
        classtrain.append(2)
        if flagc2 == 0:
            ax.scatter(xtrain[i], ytrain[i], s=20, c='g', marker='x', label='class2 train')
            flagc2 = 1
        else:
            ax.scatter(xtrain[i], ytrain[i], s=20, c='g', marker='x')


print(classtrain)
x1 = np.arange(-7,7,0.2)
x2 = np.arange(-7,7,0.2)

[X1,X2]=np.meshgrid(x1,x2)

mu1 = np.asmatrix([0,0]);
Sigma1 = np.asmatrix([[.25,.3],[.3,1]])

F1=multivariate_normal([0,0],[[.25,.3],[.3,1]]);
print(F1)


pos = np.empty(X1.shape + (2,))
pos[:, :, 0] = X1
pos[:, :, 1] = X2

F1=F1.pdf(pos)
F1=np.reshape(F1,(len(X1),len(X2)))

mu2 = np.asmatrix([2,2]);
Sigma2 = np.asmatrix([[.5,0],[0,0.5]])


F2=multivariate_normal([2,2],[[.5,0],[0,.5]]);

pos = np.empty(X1.shape + (2,))
pos[:, :, 0] = X1
pos[:, :, 1] = X2

F2=F2.pdf(pos)

F2=np.reshape(F2,(len(X1),len(X2)))


#
ax.plot_surface(X1, X2, F1, cmap="Accent", lw=0.5, rstride=1, cstride=1, alpha=0.5)
ax.contour(X1, X2, F1, 10,cmap="autumn_r", linestyles="solid", offset=-1)

ax.plot_surface(X1, X2, F2, cmap="Accent", lw=0.5, rstride=1, cstride=1, alpha=0.5)
ax.contour(X1, X2, F2, 10, cmap="autumn_r", linestyles="solid", offset=-1)


r1=1/(((2*3.1416)**2*np.linalg.det(sig1))**(1/2))
r2=1/(((2*3.1416)**2*np.linalg.det(sig2))**(1/2))
lnresult=np.log(r2/r1)
print(lnresult)

y1=np.zeros_like(x1)
for i in range(0,len(x1)):
    xv=np.asmatrix([x1[i],x2[i]])
    g1=0.5*np.dot(np.dot((xv-mu1),sig1),np.transpose(xv-mu1))
    g2=0.5*np.dot(np.dot((xv-mu2),sig2),np.transpose(xv-mu2))
    res=g2-g1-lnresult
    y1[i]=res

ax.set_xlabel('X1')
ax.set_xlim(-7, 7)
ax.set_ylabel('X2')
ax.set_ylim(-7, 7)
ax.set_zlabel('Probability Density')
ax.set_zlim(-1, 0.6)

ax.plot(x1, y1,-1, color='black')


plt.show()

