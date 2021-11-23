# script to run the Plummer sphere benchmark from the JOSS paper: time how long it takes to run acceleration and potential solves for tree and bruteforce methods, and plot the results

from pytreegrav import *
import numpy as np
from time import time
from matplotlib import pyplot as plt
import palettable

parallel = True
theta = 0.7
soft = 0.
N = 2**np.arange(6,28)
t1 = []
t2 = []
t3 = []
t4 = []
force_error = []
phi_error = []
x = np.random.rand(10**1,3)
m = np.random.rand(10**1)
Accel(x,m,np.repeat(soft,len(m)), parallel=parallel,theta=theta,method='tree')
Accel(x,m,np.repeat(soft,len(m)), parallel=parallel,theta=theta,method='bruteforce')
#BruteForceAccel(x,m,np.repeat(soft,len(m)))
Potential(x,m, np.repeat(soft,len(m)),parallel=parallel,theta=theta,method='tree')
Potential(x,m, np.repeat(soft,len(m)),parallel=parallel,theta=theta,method='bruteforce')

fig, ax = plt.subplots(figsize=(4,4))
ax.set_prop_cycle('color', palettable.colorbrewer.qualitative.Dark2_4.mpl_colors)
for n in N:
    print(n)
    x = np.random.rand(n)
    r = np.sqrt( x**(2./3) * (1+x**(2./3) + x**(4./3))/(1-x**2))
    phi_exact = -(1+r**2)**-0.5
    x = np.random.normal(size=(n,3))
    x = (x.T * r/np.sum(x**2,axis=1)**0.5).T
    m = np.repeat(1./n,n)
    h = np.ones_like(m) * soft
    t = time()
    phitree = Potential(x, m, h, parallel=parallel,theta=theta,method='tree')
    t = time() - t 
    t1.append(t)
    t = time()
    atree = Accel(x, m, h, parallel=parallel,theta=theta,method='tree')
    print(atree)
    t = time() - t
    t2.append(t)
    if n < 64**3:
        t = time()
        phibrute = Potential(x, m, h, parallel=parallel,theta=theta,method='bruteforce')
        t = time() - t
        t3.append(t)
        phi_error.append(np.std((phitree-phibrute)/phibrute))
        t = time()
        abrute = Accel(x, m, h, parallel=parallel,theta=theta,method='bruteforce')
        t = time() - t
        t4.append(t)
        amag = ((np.sum(abrute**2,axis=1) + np.sum(atree**2,axis=1))/2)
        aerror = np.sum((abrute-atree)**2,axis=1)
        force_error.append((aerror/amag).mean()**0.5)
        print(force_error[-1])
    else:
        t4.append(np.nan)
        t3.append(np.nan)
        force_error.append(np.nan)
        phi_error.append(np.nan)


ax.loglog(N, np.array(t1)/N,label="Potential (Tree)")
ax.loglog(N, np.array(t2)/N,label="Acceleration (Tree)")
ax.loglog(N, np.array(t3)/N,label="Potential (Brute Force)")
ax.loglog(N, np.array(t4)/N, label="Acceleration (Brute Force)")
ax.legend(labelspacing=0.1,frameon=True)
ax.set_ylabel("Time per particle (s)")
ax.set_xlabel("Number of particles")
plt.savefig("CPU_Time.png",bbox_inches='tight')
plt.clf()
plt.loglog(N, phi_error, label="Potential error")
plt.loglog(N, force_error, label="Acceleration error")
print(force_error, phi_error)
plt.legend()
plt.savefig("Errors.png",bbox_inches='tight')

