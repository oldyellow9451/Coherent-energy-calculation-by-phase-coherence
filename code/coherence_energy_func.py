#!/usr/bin/env python
# coding: utf-8
#%% Import modules
import os
import numpy as np
import obspy
from obspy import Stream
from obspy.geodetics import gps2dist_azimuth
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Import internals
import PhaseCoherence as PC
from read_data1 import read_template
#%% earthquake catalog
usgs_ev=pd.read_csv("../data/query.csv") 
evlat=usgs_ev.latitude
evlon=usgs_ev.longitude
evdep=usgs_ev.depth
ot=usgs_ev.time
#%% parameters
# frequency range
blim=[2,8]

# window for template,
wintemp = [-0.25,1.75]

# we'll buffer by buftemp on either side of the template
# the template tapers to zero with a cosine taper within the buffer
buftemp = 0.0
#abound=2.

# times of window centers: every 6 seconds
# relative to reflook + shtemp + shtry
wlenlook=4 # Windows where PC is computed are wlenlook seconds long
dtl=2 # Windows are separated by dtl seconds

trange=[-200,dtl]
#shift = []
tlook = np.arange(trange[0],trange[1],dtl)

# need to highpass to avoid aliasing
hpfilt=np.minimum(np.diff(wintemp)[0],wlenlook)
hpfilt=3/hpfilt

#constrain stations used
chal_limit=15
min_dist=0
max_dist=50

isdatasection=True
isstamap=True
plot_idxr=[0,10]

wdir='../data/'
path_temp=wdir+'template_event/'
path_data=wdir+'rawdata_event_sanjacinto/'

odata_dir='./results/result_event_within50km_4sec/'

if not os.path.exists(odata_dir):
    os.makedirs(odata_dir)
#%% calculation
for ev_id in [38]:
    print('event: '+str(ev_id))
    path_T = path_temp+'event_'+str(ev_id)+'_temp/'
    path_D = path_data+'event_'+str(ev_id)+'/'
    
    stalist=pd.read_csv(path_temp+'allstalist')
    alldata=[]
    dist=[]
    azi=[]
    for k in range(len(stalist)):
        dis_m,azi_m,_ = gps2dist_azimuth(evlat[ev_id], evlon[ev_id], stalist['lat.'][k], stalist['lon.'][k])
        dist.append(dis_m/1000)
        azi.append(azi_m)
    stalist['dist'] = dist
    stalist['azi'] = azi
    stalist.sort_values(by=['dist'],inplace=True)
    #%%
    # Get data waveform
    D = obspy.read(path_D + '/*.sac')
    D.merge()
    #%%
    # Get template
    T = read_template(D,path_T)
    T.merge()
    #T.plot()
    #%%
    # Get time shifts based on arrival time differences
    shifts = {}
    lini = 99999999.
    for c,tr in enumerate(T):
        nid = tr.stats.network+'.'+tr.stats.station+'.'+tr.stats.channel
        shifts[nid] = tr.stats.sac['t0']
        if tr.stats.sac['t0']<lini:
            lini=tr.stats.sac['t0']
        tr.stats.t0=tr.stats.sac['t0']
        
        std=D.select(network=tr.stats.network,station=tr.stats.station,channel=tr.stats.channel)
        #print(std)
        for tri in std:
            tri.stats.t0=tr.stats.starttime+tr.stats.t0-tri.stats.starttime
    
    D_tmp=obspy.Stream()
    for tr in D:
        if hasattr(tr.stats,'t0'):
            D_tmp.append(tr)
    D=D_tmp
    #%%
    # Filter that
    T.filter('bandpass',freqmin=hpfilt,freqmax=10)
    D.filter('bandpass',freqmin=hpfilt,freqmax=10)
    #%% test
    for tr in D:
        tmp=stalist.loc[(stalist['station']==tr.stats.network+'.'+tr.stats.station) & (stalist['chl.']==tr.stats.channel[:-1])]['dist']
        tr.stats.dist=tmp.iloc[0]
        tmp=stalist.loc[(stalist['station']==tr.stats.network+'.'+tr.stats.station) & (stalist['chl.']==tr.stats.channel[:-1])]['azi']
        tr.stats.azi=tmp.iloc[0]
    for tr in T:
        tmp=stalist.loc[(stalist['station']==tr.stats.network+'.'+tr.stats.station) & (stalist['chl.']==tr.stats.channel[:-1])]['dist']
        tr.stats.dist=tmp.iloc[0]
        tmp=stalist.loc[(stalist['station']==tr.stats.network+'.'+tr.stats.station) & (stalist['chl.']==tr.stats.channel[:-1])]['azi']
        tr.stats.azi=tmp.iloc[0]
    D=D.sort(['dist','azi'])
    T=T.sort(['dist','azi'])
    
    T = Stream([tr for tr in T if min_dist <= tr.stats.dist <= max_dist])
    D = Stream([tr for tr in D if min_dist <= tr.stats.dist <= max_dist])
    #%% remove certain stations
    if ev_id==38:
        for tr in D.select(network='PB',station="B086"):
            D.remove(tr)
    #%%
    print(T.__str__(extended=True))
    print(D.__str__(extended=True))
    if len(T) >= chal_limit:
        #%% preprocess the continuous data
        D0=D.copy()
        for tr in D:
            ind=tr.times()>tr.stats.t0+wintemp[1]+buftemp
            tr.data[ind]=0
        #%%
        # Initialise Phase coherence object
        P = PC.PhaseCoherence('test',template=D,data=D)
        
        # Prepare data for computation
        P.PrepareData()
        #%% Define few extra parameters
        normtype='template-sumfreq-bounded'
        #%%
        print('1: Set parameters')
        # Set parameters in objects
        P.setParams(shtemp='t0',wintemp=wintemp,buftemp=buftemp,shlook='t0',wlenlook=wlenlook,tlook=tlook,blim=blim,normtype=normtype)
        #%%
        print('2: x-c')
        # Make cross-correlation
        P.crosscorrDT()
        #%%
        print('3: taper')
        # Taper first cross-correlation
        P.taperCrosscorr()
        #%%
        print('4: Cp')
        # Compute phase coherence
        # You'll probably get some divide by zero errors here.  That's fine
        P.computeCp()
        
        t = P.params['tlook']
        print('Done')
        #%%plot coherence
        x_stat=P.Cp['Cpstat']
        x_comp=P.Cp['Cpcomp']
        
        P1=P.data.copy()
        Cp=P.Cp.copy()
        P.templatePower()
        #%%
        fig,axis=plt.subplots()
        # with uncertainty
        tp='Cpstat'
        fct=1./P.temppow[tp]
        x_stat_rescaled=x_stat*fct
        x_stat_std_rescaled=Cp[tp+'_std']*fct
        plt.plot(t,x_stat_rescaled,marker='o',mfc='red',color='black',ms=8,linestyle='None',linewidth=1.5)
        plt.plot(t,-x_stat_rescaled,marker='^',mfc='gray',color='gray',ms=8,linestyle='None',linewidth=1.5)
        
        x=np.append(Cp['tlook'],np.flipud(Cp['tlook']))
        y=np.append(x_stat_rescaled-x_stat_std_rescaled,
                    np.flipud(x_stat_rescaled+x_stat_std_rescaled))
        
        ply = Polygon(np.vstack([x,y]).transpose())
        ply.set_edgecolor('none')
        ply.set_color('lightblue')
        ply.set_alpha(0.7)
        #axis.add_patch(ply)
        plt.yscale('log')
        plt.xlim([-200,0])
        plt.xlabel('Time since mainshock (s)',size=15)
        plt.ylabel('Coherent power',size=15)
        plt.savefig(odata_dir+str(ev_id)+'_energy_intersta.pdf')
        #%%
        tmp=np.empty((len(t),3))
        tmp[:]=np.nan
        tmp[:,0]=t
        tmp[:,1]=x_stat_rescaled
        tmp[:,2]=x_stat_std_rescaled
        np.save(odata_dir+str(ev_id)+'_energy_intersta.npy', tmp)