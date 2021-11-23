# simple test program for the tree solver: computes acceleration and potential and checks that it is as accurate as expected

import numpy as np
from pytreegrav import Accel, Potential
from time import time

def test_answer():
    # generate points
    np.random.seed(42)
    N = 10**4
    x = np.random.rand(N,3)
    m = np.ones(N)/N
    h = np.repeat(0.01,N)

    accel_tree = Accel(x,m,h,method='tree')
    accel_bruteforce = Accel(x,m,h,method='bruteforce')
    phi_tree = Potential(x,m,h,method='tree')
    phi_bruteforce = Potential(x,m,h,method='bruteforce')

    acc_error = np.sqrt(np.mean(np.sum((accel_tree-accel_bruteforce)**2,axis=1))) # RMS force error
    print("RMS force error: ", acc_error)
    phi_error = np.std(phi_tree - phi_bruteforce)
    print("RMS potential error: ", phi_error)

    assert acc_error < 0.02
    assert phi_error < 0.02
