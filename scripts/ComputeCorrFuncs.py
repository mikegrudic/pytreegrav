#!/usr/bin/env python
# Computes correlation functions for gas in STARFORGE snapshots, stored in correlation_functions/corrfunc_N.dat
from sys import argv
import h5py
from os.path import isdir,abspath
from os import mkdir
from pytreegrav import *

for f in argv[1:]:
    with h5py.File(f,"r") as F:
        corrpath = abspath(f).split("snapshot_")[0] + "/correlation_functions"
        snapnum = f.split("snapshot_")[1].split(".hdf5")[0]
        if not isdir(corrpath): mkdir(corrpath)
        
        rho = np.array(F["PartType0"]["Density"])
        n = rho*29.9
        cut = (n > 10)
        rho = rho[cut]
        x=np.array(F["PartType0"]["Coordinates"])[cut]
        m = np.array(F["PartType0"]["Masses"])[cut]
        v = np.array(F["PartType0"]["Velocities"])[cut]/1e3
        B = np.array(F["PartType0"]["MagneticField"])[cut]*1e4
        vol = m/rho
        boxsize = F["Header"].attrs["BoxSize"]

        rbins = np.logspace(-3,2,61)
        rbins, mbin = DensityCorrFunc(x,m,rbins,theta=0.5,parallel=True,boxsize=boxsize)
        rhocorr = mbin / np.diff(4*np.pi*rbins**3 / 3) # convert from mass in bins to average density profile around a point
        rbins, Sv = VelocityStructFunc(x,vol,v,rbins,theta=0.5,parallel=True,boxsize=boxsize)
        rbins, SB = VelocityStructFunc(x,vol,B,rbins,theta=0.5,parallel=True,boxsize=boxsize)
        rbins, vcorr = VelocityCorrFunc(x,vol,v,rbins,theta=0.5,parallel=True,boxsize=boxsize)
        rbins, Bcorr = VelocityCorrFunc(x,vol,B,rbins,theta=0.5,parallel=True,boxsize=boxsize)

        r_avg = (rbins[1:]*rbins[:-1])**0.5
        np.savetxt(corrpath + "/corrfunc_" + snapnum + ".dat", np.c_[r_avg, rhocorr, vcorr, Bcorr, Sv, SB], header="#COLUMNS: (0) bin effective radius (1) density autocorrelation (2) velocity autocorrelation (3) Magnetic field autocorrelation (4) Velocity structure function (5) Magnetic field structure function" )
        
