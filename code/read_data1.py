#!/usr/bin/env python3.7

import numpy as np
import sys
import os
import obspy
from obspy.clients.fdsn import Client
import glob

# ------------------------------------------------------------------------------

def read_template(D,path_T):

    ## Extract stations from data
    staD = sta_from_data(D)
    ## Extract template corresponding to data
    T = sta_from_temp(staD,path_T)

    return T

# ------------------------------------------------------------------------------

def sta_from_data(st):
    sta = list()
    for tr in st:
        sta.append(tr.stats.network+'.'+tr.stats.station)

    stat = list(set(sta))
    return stat

# ------------------------------------------------------------------------------

def sta_from_temp(staD,path_temp):
    staT = list()
    for sta in staD:
        #print(path_temp + sta + '/*' + id_T + '*')
        #print(glob.glob(path_temp + '/' + sta + '*.sac'))
        if glob.glob(path_temp + '/' + sta + '*.sac'):
            t = obspy.read(path_temp + '/' + sta + '*.sac')
            try:
                t0 = t[0].stats.sac.t0
                staT.append(sta)
            except:
                pass
    #print(staT)
    T = obspy.Stream()
    for sta in staT:
        if glob.glob(path_temp + '/' + sta + '*.sac'):
            st = obspy.read(path_temp + '/' + sta + '*.sac')
            T += st

    return T

# ------------------------------------------------------------------------------
