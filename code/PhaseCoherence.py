'''
A base class to perform Phase Coherence analysis

Original code from J. Hawthorne (Hawthorne and Ampuero 2017)
Transformed into class by B. Gombert (May 2018)

'''

# Import classic external stuff
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy
import datetime
import code

# Import seismo external stuff
import obspy
import spectrum

# Import J. personnal modules
#import general # NOW UNUSED
import graphical
import seisproc

# for plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import gridspec

class PhaseCoherence(object):

    '''
    Class implementing the phase coherence method

    Args required:
        * name      : Instance Name
        * template  : Obspy stream for templates

    Args optionnal:
        * data      : Obspy stream of data set to search (default: st1)
        * lat0      : Origin latitude (for plot, default=0)
        * lon0      : Origin longitude (for plot, default=0)
    '''

    # -------------------------------------------------------------------------------
    # Initialize class #
    def __init__(self,name, template, data=None, lon0=None, lat0=None):

        '''
        Initialize main variables of the class
        '''

        # Save name
        assert(type(name) is str), 'name argument must be a string'
        self.name = name

        # Lon/Lat of reference.
        if lon0 is None:
            self.lon0 = 0.
        else:
            self.lon0 = lon0

        if lat0 is None:
            self.lat0 = 0.
        else:
            self.lat0 = lat0

        # Copy data and template with copuy.deepcopy() to avoid changing original data
        # Save template
        assert(type(template) is obspy.core.stream.Stream), 'template argument must be an obspy stream'
        self.template = template.copy()

        # default data
        if data is None:
            self.data = template.copy()
        else:
            assert(type(template) is obspy.core.stream.Stream), 'data argument must be an obspy stream'
            self.data = data.copy()

    # -------------------------------------------------------------------------------
    def PrepareData(self,verbose=False):
        '''
        This function just check streams, remove unused traces, resample, etc...
        '''

        # 1/ Get infos on stations, compo, and common traces
        self._getCommonTraces()

        # 2/ Keep only traces which are in common in template and data
        self._removeUnusedTraces(verbose=verbose)

        # 3/ Resample data if necessary
        self._resampleTraces(verbose=verbose)

        # 4/ Merge and mask data
        self.merge()

        return

    # -------------------------------------------------------------------------------
    def ComputePhaseCoherence(self,reftemp=None, shtemp=None, wintemp=None,\
                              buftemp=None, reflook=None, shlook=None, wlenlook=None,\
                              tlook=None,blim=None, shtry=None, shgrid=None,\
                              taper=True, cptype='both',\
                              normtype=None,abound=None,verbose=True,
                              returnjack=None):
        '''
        This function assumes you already prepare the data, makes te rest
         * reftemp   : reference time for templates (default: often ignored,
                      but first start time of template[0])
        * shtemp    : Time shift for templates (default: 't0'). Either
                        (1) a string for a marker (then reftemp is ignored)
                        (2) time shifts since reftemp in seconds
        * wintemp   : Window since reference for template (default: [0,3.])
        * buftemp   : A buffer on either side of the template window (default: diff(wintemp)/6)
        * reflook   : Reference times for search data (default: often ignored,
                      but first start time of st2[0])
        * shlook    : Time shift for data (default: 't0'). Either
                        (1) a string for a marker---in which case reflook is ignored
                        (2) time shifts since reflook in seconds
        * wlenlook  : Window length for search (default: 5.)
        * tlook     : Times to search (default: all available)
        * blim      : Bandlimit to consider (default: [1,10.])
        * shtry     : A dictionary of time shifts to try for each component
        * shgrid    : Time spacing to use for gridding (default: wlen/4)
        * cptype    : Which phase coherence to compute?
                            - 'stat' for inter-station PC
                            - 'comp' for inter-component PC
                            - 'both' (default) for both
        * normtype  : Which normalization to use
        * abound    : a bound on the amplitude to use with normalization
        * returnjack: return the jackknifing results?

        * verbose   : Let the code whisper sweet words into your ears...
        '''

        # 1/ Set parameters
        print('1: Set parameters')
        self.setParams(reftemp, shtemp, wintemp,buftemp, reflook,\
                       shlook, wlenlook, tlook, blim, shtry, shgrid,
                       normtype=normtype,returnjack=returnjack)
        import timeit
        t1=timeit.time.time()

        # 2/ Make template-data cross-correlation
        print('2: x-c')
        self.crosscorrDT()
        t2=timeit.time.time()

        # 3/ Taper crosscor if wanted
        print('3: taper')
        self.params['taper']=taper
        if taper:
            self.taperCrosscorr()
        t3=timeit.time.time()

        # 4/ Compute Cp
        print('4: Cp')
        self.computeCp(cptype=cptype)
        t4=timeit.time.time()

        print('Cp time / xc time: {:0.2f}'.format((t4-t3)/(t2-t1)))

        print('Done')
        # All done
        return


    # -------------------------------------------------------------------------------
    def ComputeAll(self,reftemp=None, shtemp='t0', wintemp=None,\
                   buftemp=None, reflook=None, shlook='t0', wlenlook=5., tlook=None,\
                   blim=None, shtry=None, shgrid=None, taper=True, cptype='both',\
                   normtype=None,abound=None,verbose=True,
                   returnjack=None):

        '''
        Main function of this class. It will call a bunch of others to compute the
        phase coherence. Args to write

        * reftemp   : reference time for templates (default: often ignored,
                      but first start time of template[0])
        * shtemp    : Time shift for templates (default: 't0'). Either
                        (1) a string for a marker (then reftemp is ignored)
                        (2) time shifts since reftemp in seconds
        * wintemp   : Window since reference for template (default: [0,3.])
        * buftemp   : A buffer on either side of the template window (default: diff(wintemp)/6)
        * reflook   : Reference times for search data (default: often ignored,
                      but first start time of st2[0])
        * shlook    : Time shift for data (default: 't0'). Either
                        (1) a string for a marker---in which case reflook is ignored
                        (2) time shifts since reflook in seconds
        * wlenlook  : Window length for search (default: 5.)
        * tlook     : Times to search (default: all available)
        * blim      : Bandlimit to consider (default: [1,10.])
        * shtry     : A dictionary of time shifts to try for each component
        * shgrid    : Time spacing to use for gridding (default: wlen/4)
        * cptype    : Which phase coherence to compute?
                            - 'stat' for inter-station PC
                            - 'comp' for inter-component PC
                            - 'both' (default) for both
        * normtype  : Which normalization to use
        * abound    : A bound on the amplitude after normalization
        * returnjack: return the jackknifing results?

        * verbose   : Let the code whisper sweet words into your ears...
        '''

        # To be implemented
        if taper is False:
            taper=True
            print('taper was performed anyway, not implemented otherwise (yet)')


        # 1/ Get infos on stations, compo, and common traces
        self._getCommonTraces()

        # 2/ Keep only traces which are in common in template and data
        self._removeUnusedTraces(verbose=verbose)

        # 3/ Resample data if necessary
        self._resampleTraces(verbose=verbose)

        # 4/ Merge and mask data
        self.merge()

        # 5/ Set parameters
        self.setParams(reftemp, shtemp, wintemp,buftemp, reflook,\
                       shlook, wlenlook, tlook, blim, shtry, shgrid,
                       normtype=normtype,abound=abound,
                       returnjack=returnjack)

        print('6: x-c')
        # 6/ Make template-data cross-correlation
        self.crosscorrDT()

        print('7: tapering')
        # 7/ Taper crosscor if wanted
        if taper:
            self.taperCrosscorr()

        print('8: Cp calculation')
        # 8/ Compute Cp
        self.computeCp(cptype=cptype)

        print('done')
        # All done
        return

    # ------------------------------------------------------------------------------

    def templatePower(self):
        """
        compute the power in the template using the same approach as the data are treated
        """

        # reset the data to the template
        st=self.template.copy()
        data=self.data
        self.data=st

        for tr in st:
            # trim the template
            tsec,tdat1 = self.resolvepick(tr,pk=self.params['shtemp'],
                                          mkset=None,reftemp=self.params['reftemp'])
            wintemp,buftemp=self.params['wintemp'],self.params['buftemp']
            tr.trim(starttime=tdat1+wintemp[0]-buftemp,
                    endtime=tdat1+wintemp[1]+buftemp,pad=True)

            # taper the buffered template
            tr.taper(type='cosine',max_percentage=None,
                     max_length=buftemp,side='both')

            # pad with zeros
            nzer=int(np.ceil(self.params['wlenlook']/tr.stats.delta))+1
            tr.data=np.hstack([np.zeros(nzer),tr.data,np.zeros(nzer)])
            tr.stats.starttime=tr.stats.starttime-nzer*tr.stats.delta

            # and move pick
            if isinstance(self.params['shtemp'],str):
                tr.stats[self.params['shtemp']]=tdat1-tr.stats.starttime

        # and reset the reference times
        params=self.params
        reflook,shlook=params['reflook'],params['shlook']
        params['reflook']=params['reftemp']
        params['shlook']=params['shtemp']

        # consider time with no shift
        tlook=params['tlook']
        params['tlook']=np.array([0.])

        # 6/ Make template-data cross-correlation
        self.crosscorrDT()


        # 7/ Taper crosscor if wanted
        self.taperCrosscorr()

        # 8/ Compute Cp
        self.computeCp()

        # save these numbers
        self.temppow={'Cpstat':self.Cp['Cpstat'][0],
                      'Cpcomp':self.Cp['Cpcomp'][0]}


        # reset the data to the original
        self.data=data
        params['reflook']=reflook
        params['shlook']=shlook
        params['tlook']=tlook

    def errorDist(self,typ='chi2',bns=None):
        """
        return a  probability distribution
        :param         typ: which distribution to obtain
        :param         bns: the bin edges
        :return       Nper: the probability density function
        :return        bns: bin edges, as a fraction of the standard deviation
        :return      stdev: the standard deviation
        """

        # binning
        if bns is None:
            bns=np.arange(-6,6.01,0.1)

        if typ == 'chi2':
            # number of variables to average over
            Npair,Nind,Nave = self.naveraged()
            Nave=int(np.round(np.mean(Nave)))
            Nind=np.median(Nind[Nind>0])


            # make random distributions
            vl=[np.mean(np.random.chisquare(1,Nave))-1.
                for k in range(0,20000)]
            vl=vl/Nind/(Nind-1)

            # note standard deviation
            Nind=np.median(Nind[Nind>0])
            stdev=np.std(vl)

        # bin them
        Nper,trash=np.histogram(vl,bins=bns*stdev)

        return Nper,bns,stdev


    # -------------------------------------------------------------------------------
    def setParams(self,reftemp=None, shtemp=None, wintemp=None,buftemp=None,\
                  reflook=None, shlook=None, wlenlook=None, tlook=None,\
                  blim=None, shtry=None, shgrid=None,normtype=None,abound=None,
                  returnjack=None,uncweights=None):

        '''
        Set the different parameters used for the phase coherence analysis in a dictionnary
        Args:
        * reftemp   : reference time for templates (default: often ignored,
                      but first start time of template[0])
        * shtemp    : Time shift for templates (default: 't0'). Either
                        (1) a string for a marker (then reftemp is ignored)
                        (2) time shifts since reftemp in seconds
        * wintemp   : Window since reference for template (default: [0,3.])
        * buftemp   : A buffer on either side of the template window (default: diff(wintemp)/6)
        * reflook   : Reference times for search data (default: often ignored,
                      but first start time of st2[0])
        * shlook    : Time shift for data (default: 't0'). Either
                        (1) a string for a marker---in which case reflook is ignored
                        (2) time shifts since reflook in seconds
        * wlenlook  : Window length for search (default: 5.)
        * tlook     : Times to search (default: all available)
        * blim      : Bandlimit to consider (default: [1,10.])
        * shtry     : A dictionary of time shifts to try for each component
        * shgrid    : Time spacing to use for gridding (default: wlen/4)
        * normtype  : which type of normalization to use (default: 'full')
        * abound    : a bounding on the amplitude after normalization (default: 3)
        * returnjack: Return the jackknifing results
        * uncweights: choose weighting based on taper uncertainty
        '''

        # Make empty dictionnary if needed
        # moved before setting parameters to allow internal references
        # 24-Sep-2020 by JCH
        if not 'params' in self.__dict__.keys():
            self.params = {}
        param=self.params


        # default reference times
        if reftemp is None:
            param['reftemp'] = self.template[0].stats.starttime
        elif not 'reftemp' in param.keys():
            param['reftemp'] = reftemp

        if reflook is None:
            param['reflook'] = self.data[0].stats.starttime
        elif not 'reflook' in param.keys():
            param['reflook'] = reflook

        # template info
        if wintemp is not None:
            assert(len(wintemp)==2), 'wintemp must have 2 values'
            param['wintemp'] = np.array(wintemp)

        if buftemp is None:
            if not 'buftemp' in param.keys():
                param['buftemp'] = np.diff(self.params['wintemp'])[0]/6.
        else:
            param['buftemp'] = buftemp

        # return the jackknifing results?
        if returnjack is None:
            if not 'returnjack' in param.keys():
                param['returnjack'] = False
        else:
            param['returnjack'] = returnjack


        # Time shift for template and data
        if shlook is not None:
            param['shlook'] = shlook
        elif not 'shlook' in param.keys():
            param['shlook']='t0'

        if shtemp is not None:
            param['shtemp'] = shtemp
        elif not 'shtemp' in param.keys():
            param['shtemp']='t0'

        # Length of looking window
        if wlenlook is not None:
            param['wlenlook'] = wlenlook

        # frequency range
        if blim is not None:
            assert(len(blim)==2), 'dlim must have 2 values'
            param['blim'] = blim

        # default time shift gridding
        if tlook is not None:
            # added if statement 24-sep-2020, JCH
            param['tlook'] = tlook

        if shgrid is None:
            shgrid = param['wlenlook']/4.
            param['shgrid'] = np.maximum(shgrid,self.dtim)
        else:
            param['shgrid'] = shgrid

        if shtry is None:
            param['shtry'] = dict((nsci,np.array([0.])) for nsci in self.stations['CommonTr'])
        else:
            assert(type(shtry) is dict), 'shtry must be a dictionnary'
            param['shtry'] = shtry

        # for normalized x-c
        if normtype is not None:
            param['normtype']=normtype
        elif not 'normtype' in param.keys():
            param['normtype']='full'
        elif param['normtype'] is None:
            param['normtype']='full'

        if abound is not None:
            param['abound']=abound
        elif 'abound' not in param.keys():
            param['abound']=3.
        elif param['abound'] is None:
            param['abound']=3.

        # for the uncertainty-based weighting
        if uncweights is not None:
            param['uncweights']=uncweights
        elif 'uncweights' not in param.keys():
            param['uncweights']=False
        elif param['uncweights'] is None:
            param['uncweights']=False



        # All done
        return

    # ------------------------------------------------------------------------------

    def naveraged(self,mode='station'):
        """
        returns the number of averages in each phase coherence calculation
        :param        mode: which type of coherence is calculated ('station' or 'component')
        :return      Npair: the number of station/component pairs in each calculation
        :return       Nind: the number of individual inputs to the coherence calculation
        :return       Nave: the number of station/components/tapers/frequencies in the
                               averaged coherence value
        """

        if mode.lower()=='station':
            Nind=self.Cp['Nstat']
            Nind=np.median(Nind,axis=1)
            Npair=np.multiply(Nind,Nind-1)/2

            Nave=np.sum(self.Cp['Nstat']>0,axis=1)*len(self.freq)*self.Ntap

        elif mode.lower()=='component':
            Nind=self.Cp['Ncomp']
            Nind=np.median(Nind,axis=1)
            Npair=np.multiply(Nind,Nind-1)/2

            Nave=np.sum(self.Cp['Ncomp']>0,axis=1)*len(self.freq)*self.Ntap

        # to account for frequency spacing
        Nave = (Nave/self.fspc).astype(int)

        return Npair,Nind,Nave

    # -------------------------------------------------------------------------------
    def merge(self):
        '''
        Merge traces and add a mask if necessary
        '''
        self.template = self.template.merge()
        self.data     = self.data.merge()
        for tr in self.template+self.data:
            if not isinstance(tr.data,np.ma.masked_array):
                tr.data=np.ma.masked_array(tr.data,mask=np.isnan(tr.data))

        # All done
        return
    # -------------------------------------------------------------------------------
    def resolvepick(self,tr,pk='t0',mkset=None,reftemp=None):
        '''
         INPUT

         tr          trace
         pk          reference time for tr
                        either a string for a marker, a time in seconds from the beginning,
                        or a time
         mkset       string of a marker to set (default: None, not set)
         reftemp     reference time (default: tr.stats.starttime)

         OUTPUT

         tsec        pick time in seconds relative to the start time
         tdat        pick time as a date
        '''


        if reftemp is None:
            reftemp=tr.stats.starttime

        if isinstance(pk,str):
            # time from marker
            tsec = tr.stats[pk]
            tdat = tr.stats.starttime+tsec
        elif isinstance(pk,float):
            # if it's a time since reference
            tdat = reftemp+pk
            tsec = reftemp-tr.stats.starttime
        elif isinstance(pk,datetime.datetime):
            # if it's just a time
            tdat = obspy.UTCDateTime(pk)
            tsec = tdat - tr.stats.starttime
        elif isinstance(pk,obspy.UTCDateTime):
            # if it's just a time
            tdat = pk
            tsec = tdat - tr.stats.starttime

        if mkset:
            # set marker
            tr.stats[mkset]=tsec

        return tsec,tdat

    # -------------------------------------------------------------------------------
    def crosscorrDT(self):
        '''
        Make the cross-correlation between the data and the template
        in every window
        '''

        assert(hasattr(self,'params')),'You need to set the parameters with setParams() first'

        # Get params
        wlenlook = self.params['wlenlook']
        wintemp  = self.params['wintemp']
        reftemp  = self.params['reftemp']
        reflook  = self.params['reflook']
        buftemp  = self.params['buftemp']
        tlook    = self.params['tlook']
        shlook   = self.params['shlook']
        shtemp   = self.params['shtemp']
        dtim     = self.dtim
        nsc      = self.stations['CommonTr']

        # Get time shift params
        shcalc, ishfs = self._makeGrid()
        Nst      = self.Nst

        # a time to allow buffering of the data
        buftime = wlenlook+buftemp*2+np.diff(wintemp)[0]
        buftime = buftime + 3*dtim

        # length of template
        M=int(np.round((np.diff(wintemp)[0]+2*buftemp)/dtim))

        # length of window
        Mw=int(np.round(wlenlook/dtim))

        # number of indices
        Nt=len(tlook)

        # number of stations
        Ns=len(nsc)

        # cross-correlation
        xc = np.ndarray([Mw,Nt,Nst],dtype=float)
        # commented this line out; I don't know why it's here
        #self.tmp = np.ndarray([73401,Nst],dtype=float)
        dok = np.ndarray([Nt,Nst],dtype=bool)

        # also initialize the cross-correlation for the template
        xct = np.ndarray([Mw,Nst],dtype=float)

        # relative time limits to extract
        tget = [wintemp[0]-buftemp+np.min(tlook)-wlenlook-buftime,
                wintemp[1]+buftemp+np.max(tlook)+wlenlook+buftime+4*dtim]

        # potential length of data to search
        N=int((tget[1]-tget[0])/dtim)

        # need to pick times
        ix=np.arange(0,N,1)*dtim
        ix=np.searchsorted(ix,tlook-wlenlook/2.+wintemp[0]-buftemp-tget[0])
        ix=np.minimum(ix,N-1)
        ix=np.maximum(ix,0)

        # also identify the portion to extract for the template x-c with itself
        jx=((np.diff(wintemp)[0]+buftemp*2)/2-wlenlook/2)/dtim
        ntemp=int((np.diff(wintemp)[0]+buftemp*2)/dtim)
        jx=int(jx)+np.arange(0,Mw)
        jx=jx[np.logical_and(jx>=0,jx<ntemp)]
        nbf=np.maximum(0,int((Mw-jx.size)/2))
        naf=np.maximum(Mw-nbf-jx.size,0)

        # how to do the cross-correlation
        xcapproach=2

        # grid to extract
        if xcapproach==1:
            i1,i2=np.meshgrid(ix,np.arange(0,Mw,1))
        elif xcapproach==2:
            i1,i2=np.meshgrid(ix,np.arange(-M,Mw+M,1))
        i1=i1+i2
        i1=np.minimum(i1,N-1)
        i1=np.maximum(i1,0)

        # and to extract for data percentage
        ix2=ix+Mw+M
        ix2=np.minimum(ix2,N-1)

        # computation starts
        d = obspy.Stream()
        t = obspy.Stream()

        # for each trace
        for nsci in nsc:
            # template extraction just once per station/channel
            # related data
            vls=nsci.split('.')
            tr1 = self.template.select(network=vls[0],station=vls[1],
                                       channel='*'+vls[2])[0].copy()

            # arrival time
            tsec,tdat1 = self.resolvepick(tr1,pk=shtemp,mkset=None,reftemp=reftemp)

            # grab the template (note this distorts the picks)
            tr1=tr1.trim(starttime=tdat1+wintemp[0]-buftemp,
                         endtime=tdat1+wintemp[1]+buftemp+3*dtim,pad=True)
            tr1.data=tr1.data[0:M]

            # taper
            tr1.taper(type='cosine',max_percentage=None,
                      max_length=buftemp,side='both')

            # extract data and set missing data to zero
            data1=tr1.data.data
            data1[tr1.data.mask]=0.

            # cross-correlate the template with itself,
            # as we'll want the template power later
            xcti=scipy.ndimage.filters.correlate1d(data1,data1,mode='constant',cval=0.)
            xcti=np.hstack([np.zeros(nbf,dtype=float),xcti[jx],np.zeros(naf,dtype=float)])


            for mm in range(0,len(ishfs[nsci])):
                # which time shifts to consider here
                tshf = shcalc[nsci][mm]
                m = ishfs[nsci][mm]

                # save the template x-c
                # note this may be duplicated if there are multiple shifts per station
                xct[:,m]=xcti

                # related data
                tr2 = self.data.select(network=vls[0],station=vls[1],
                                       channel='*'+vls[2])[0].copy()

                # arrival time
                tsec,tdat2 = self.resolvepick(tr2,pk=shlook,mkset=None,reftemp=reflook)

                # grab the limited data of interest
                # (note this distorts the picks)
                tr2=tr2.trim(starttime=tdat2+tget[0]+tshf,
                             endtime=tdat2+tget[1]+6*dtim+tshf,
                             pad=True)
                tr2.data=tr2.data[0:N]

                t.append(tr1)
                d.append(tr2)

                # extract data and set missing data to zero
                data2=tr2.data.data
                data2[tr2.data.mask]=0.

                if xcapproach==1:

                    # cross-correlate
                    xci=scipy.ndimage.filters.correlate1d(data2,data1)

                    # grab the correct portion
                    ixc = int(data1.size/2)
                    xci=xci[i1+ixc]

                elif xcapproach==2:

                    # grab data and cross-correlate
                    xci=np.fft.rfft(data2[i1],axis=0,n=M*2+Mw)
                    xci2=np.fft.rfft(data1,n=M*2+Mw).reshape([xci.shape[0],1])
                    xci=np.fft.irfft(np.multiply(xci,np.conj(xci2)),axis=0)
                    xci=xci[M:-M,:]

                # detrend the x-c
                dtd=np.linspace(0,1,Mw).reshape([Mw,1])
                dtd1=xci[0:1,:]
                dtds=xci[-1:]-dtd1
                xci=xci-(dtd1+np.multiply(dtd,dtds))

                # also figure out if the data is good enough
                data1=(~tr1.data.mask).astype(float)
                data2=(~tr2.data.mask).astype(float)
                data2=np.cumsum(data2)
                doki=(data2[ix2]-data2[ix])*(sum(data1)/M/(Mw+M))
                doki=doki>0.95

                # add to set
                dok[:,m]=doki
                xc[:,:,m]=xci



        # Save it
        self.dok = dok
        self.crosscorr  = xc
        self.crosscorr_temp = xct
        self.tget = tget
        shift = ((tlook-wlenlook/2.+wintemp[0]-buftemp)[0]-tlook[0])/2.
        self.shift = shift

        self.d = d
        self.t = t
        # commented out, 25-Sep-2020, JCH
        # del self.data
        # del self.template

        # All done
        return


    # -------------------------------------------------------------------------------

    def taperEnergy(self):
        '''
        compute the energy in the tapers within wlenlook
        :return    tapen: taper energy, normalized to a mean of 1
        '''

        tapen=np.sum(np.power(self.tapers,2),axis=1)
        tapen=tapen/(tapen.size*self.Ntap)

        return tapen


    # ------------------------------------------------------------------------------

    def taperCrosscorr(self,taper=True):

        '''
        Taper the cross-correlation bewteen data and template using
        a discrete prolate spheroidal (Slepian) sequences.
        Changes self.xc
        '''

        # Get a couple of parameters
        Mw  = int(np.round(self.params['wlenlook']/self.dtim)) # length of window
        xc  = self.crosscorr.copy()
        xct  = self.crosscorr_temp.copy()
        Nt  = len(self.params['tlook']) # number of indices
        Ns  = len(self.stations['CommonTr']) # number of stations
        Nst = self.Nst
        wlenlook = self.params['wlenlook']
        blim     = self.params['blim']
        dtim     = self.dtim
        dok      = self.dok

        # tapers
        if type(taper) is bool:
            if 0==1:
                tr=obspy.Trace()
                tr.data=np.ones(Mw,dtype=float)
                tr.taper(type='hann',max_percentage=0.4,
                         max_length=tr.stats.delta*tr.stats.npts)
                tap=tr.data.reshape([Mw,1])
                Ntap = 1
                NW = 2
                V = 1.
            elif taper is False:
                tap = np.ones([Mw,1])
                Ntap = 1
                NW = 2
                V = 1.
            else:
                # compute tapers
                # comment out these three lines to exclude taper
                NW = 3
                [tap,V] = spectrum.mtm.dpss(Mw,NW)
                # Ditch crappy tapers
                ix = np.where(V>0.99)[0]
                #ix = ix[0:int(np.ceil(ix.size*0.7))]
                #ix = np.arange(0,1)
                V=V[ix]
                tap=tap[:,ix]
                Ntap = len(V)
        else:
            tap=taper
            Ntap=taper.shape[1]
            NW=2

        # normalize the tapers
        nml=np.power(np.sum(np.power(tap,2),axis=0,keepdims=True),0.5)
        tap=np.multiply(tap,np.divide(tap.shape[0],nml))

        # save the tapers
        self.tapers=tap.copy()
        self.taperevalue=np.atleast_1d(V)

        # repeat and multiply by tapers
        xc  = xc.reshape([Mw,Nt,Nst,1])
        tap = tap.reshape([Mw,1,1,Ntap])
        xc  = np.multiply(xc,tap)

        xct  = xct.reshape([Mw,Nst,1])
        tap = tap.reshape([Mw,1,Ntap])
        xct  = np.multiply(xct,tap)

        self.Ntap = Ntap

        #----------------------------------------------------------

        # fft
        Nft=Mw*2
        xc=np.fft.fft(xc,n=Nft,axis=0)
        xct=np.fft.fft(xct,n=Nft,axis=0)

        # frequencies
        freq=np.fft.fftfreq(Nft,d=dtim)

        # just frequencies in range
        ixf=np.logical_and(freq>=blim[0],freq<=blim[1])
        ixf,=np.where(ixf)

        # frequency spacing
        dfreq=np.median(np.diff(freq))
        self.fspc=2.
        spc=2/self.fspc*float(NW)/wlenlook
        spc=np.arange(0.,len(ixf)-1,spc/dfreq)
        spc=np.round(spc).astype(int)
        ixf = ixf[spc]

        freq=freq[ixf]
        Nf=len(ixf)
        xc=xc[ixf,:,:,:]
        xct=xct[ixf,:,:]

        if (type(taper) is bool)&(taper is True):
            xc=xc.reshape([Nf,Nt,Nst,Ntap])
            xct=xct.reshape([Nf,Nst,Ntap])


        # average the cross-spectra over tapers
        # xc = np.mean(xc,axis=3)
        xct = np.mean(np.abs(xct),axis=2)

        # Save tapered crosscorr
        self.xc = xc
        self.xct = xct
        self.freq = freq

        # All done
        return


    #---------BEGIN FOR WEIGHTING AND NORMALIZATION--------------------------------------------

    def _calc_weights(self,xc=None,xct=None):
        """
        compute the power in the template
        :param        xc: the unnormalized x-c with the target data
                             (default: self.xc)
        :param       xct: the unnormalized x-c of the template with itself
                             (default: self.xct)
        :return     wdiv: what to multiply each xc value by
        :return     wgts: the weights to multiply by after the division
        """

        if xc is None:
            xc=self.xc
        if xct is None:
            xct=self.xct

        # which normalization
        normtype=self.params['normtype']

        # default is just to normalise, not weight
        wgts=np.ones([1,xc.shape[1],xc.shape[2]],dtype=float)

        if normtype=='full':
            # full normalization, proper phase coherence
            wdiv=np.nanmean(np.power(np.abs(xc),2),axis=3)
            wdiv=np.power(wdiv,-0.5)

        if 'template' in normtype:
            # normalize by the template, try to estimate power relative to
            # the template, no regularization

            if 'template-sumfreq' in normtype:
                # may want to average over frequencies first
                xct=np.nanmean(xct,axis=0,keepdims=True)

            wdiv = np.divide(1,np.abs(xct))
            wdiv = wdiv.reshape([xct.shape[0],1,xct.shape[1]])

        # and if we want to change the weighting because some might be noisier
        if 'bounded' in normtype:
            # estimate taper-averaged power, normalized
            xpow=np.power(np.nanmean(np.power(np.abs(xc),2),axis=3),0.5)
            xpow=np.multiply(xpow,wdiv)


            # may want to average over frequency first
            if 'bounded-sumfreq' in normtype:
                xpow=np.nanmean(xpow,axis=0,keepdims=True)

            # compute ratio of median amplitude to amplitude at each station
            # if we wanted all stations to be weighted equally,
            # we'd multiply by this
            wgts = np.divide(np.ma.median(xpow,axis=2,keepdims=True),xpow)

            # want to put all weights between 1/abound and abound to 1
            abound=self.params['abound']
            wbig=wgts>=abound
            wsmall=wgts<=1/abound

            # set middle weights to 1
            wgts[np.logical_and(~wbig,~wsmall)]=1.

            # let large weights go from 1 at the outer edge
            wgts[wbig]=wgts[wbig]/abound

            # let small weights go from 1 at the outer edge
            wgts[wsmall]=wgts[wsmall]*abound

        # set problematic intervals to zero
        doki = self.dok.reshape([1,xc.shape[1],xc.shape[2]])
        if isinstance(wgts,np.ndarray):
            wgts[np.repeat(~doki,wgts.shape[0],axis=0)]=0.

        # save
        self.wdiv=wdiv
        self.wgts=wgts


    def weights_from_taper(self):
        """
        compute station-dependent weighting by tapers
        """


        # compute inter-component coherence
        Nt=len(self.params['tlook'])
        nper=self.stations['nper']
        Cpcomp=np.zeros([Nt,len(nper),self.Ntap],dtype=float)
        Cpcompn=np.zeros([Nt,len(nper),self.Ntap],dtype=float)

        for ktap in range(0,self.Ntap):
            # normalize
            xc=np.multiply(self.xc[:,:,:,ktap],self.wdiv)
            xc=np.multiply(xc,self.wgts)

            # calculate values for each taper
            Cpcomp[:,:,ktap],Cpcompn[:,:,ktap],Ncomp=\
                self._cpcomp_main_calc(xc,self.wgts)

        # normalize
        Cpcomp=np.divide(Cpcomp,np.nanmean(Cpcompn,axis=2,keepdims=True))

        # determine an uncertainty for each station
        Cpcomp=np.power(np.ma.std(Cpcomp,axis=2),-1)

        # but set weights larger than twice the median to the median
        mdn=np.ma.median(Cpcomp,axis=1,keepdims=True)
        ix=Cpcomp>2*mdn
        mdn=np.repeat(mdn,Cpcomp.shape[1],axis=1)
        Cpcomp[ix]=mdn[ix]

        # and need to map the station weighting
        self.apply_station_weighting(Cpcomp)

    def apply_station_weighting(self,statwgts):
        """
        :param       statwgts: the weights to use
                                   [number of times x number of stations]
        """

        # note the station mapping
        ins=self.stations['ins']

        # remember
        saveweight=True
        if saveweight:
            self.statwgts=statwgts

        # reshape as appropriate
        statwgts=statwgts.reshape([1,statwgts.shape[0],statwgts.shape[1]])

        # and multiply
        self.wgts=np.multiply(self.wgts,statwgts[:,:,ins])



    def plot_template_power(self):
        """
        plot the power in the templates
        """

        p=plt.axes()
        iplt=np.arange(0,self.xct.shape[1])
        Ns=len(iplt)
        cols=graphical.colors(Ns)
        h=[]
        for k in range(0,Ns):
            hh,=p.plot(self.freq,np.abs(self.xct[:,iplt[k]]),color=cols[k])
            h.append(hh)
        p.set_xscale('log')
        p.set_yscale('log')

    #---------END FOR WEIGHTING AND NORMALIZATION--------------------------------------------


    #-------------------------------------------------------------
    def computeCp(self,cptype='both'):
        '''
        Compute the phase coherence from the template/data tappered cross correlation.
        Returns a dictionnary stored in self.Cp
        Args:
            * comp  : Compute

        '''

        # Check some stuff
        assert(cptype in ['comp','stat','both']),'cptype must be "stat","comp", or "both"'

        # Get some parameters
        tlook   = self.params['tlook']
        Nt      = len(tlook) # number of indices
        Nf      = self.xc.shape[0]
        Nc      = self.stations['Nc']
        icmp    = self.stations['icmp']
        ins     = self.stations['ins']
        nper    = self.stations['nper']
        nsc     = self.stations['CommonTr']
        xc      = self.xc
        dok     = self.dok
        Ntap    = self.Ntap
        freq    = self.freq

        # Initialize Cp dict
        Cp = {'tlook':tlook,'freq':freq,'Ntap':Ntap,'nsc':nsc}
        self.Cp = Cp

        #------UNCERTAINTY PARAMETERS-----------------------------

        # number of bootstrap resamplings
        Nsboot=50

        # use uncertainty from tapering?
        tapunc=False
        #tapunc=True
        self.params['tapunc']=tapunc


        #-----INITIALIZE OUTPUT-----------------------------------

        if cptype in ['stat','both']:
            # to save inter-station coherence
            Cpstat=np.zeros([Nt,Nsboot+1,self.Ntap],dtype=float)
            Cpstatn=np.zeros([Nt,Nsboot+1,self.Ntap],dtype=float)

        if cptype in ['comp','both']:
            # and inter-station coherence
            Cpcomp=np.zeros([Nt,len(nper),self.Ntap],dtype=float)
            Cpcompn=np.zeros([Nt,len(nper),self.Ntap],dtype=float)

        print('starting cp computation')

        #-----DETERMINE STATION-DEPENDENT WEIGHTING---------------

        # compute the weights to use
        self._calc_weights(self.xc)


        #-----MAY WANT TO WEIGHT BASED ON TAPER-DERIVED UNCERTAINTY------
        if self.params['uncweights']:
            self.weights_from_taper()


        # iterate per taper
        for ktap in range(0,self.Ntap):
            # normalize
            xc=np.multiply(self.xc[:,:,:,ktap],self.wdiv)
            xc=np.multiply(xc,self.wgts)

            #-----------inter-station coherence--------------------------

            # choose a set of stations to bootstrap, from the same set of stations for all
            # time intervals
            mtx=self._pick_bootstrap_stations(statok=nper>=1,Nsboot=Nsboot)
            self.bootsmtx=mtx

            if cptype in ['stat','both']:
                # calculate values for each taper
                Cpstat[:,:,ktap],Cpstatn[:,:,ktap],Nstat=\
                    self._cpstat_main_calc(xc,self.wgts)

                if ktap==self.Ntap-1:
                    # and calculate uncertainties if we're finished
                    self.cpstat_unc_calc(Cpstat,Cpstatn,Nstat)

            #------------inter-component coherence-------------------------
            if cptype in ['comp','both']:

                # calculate values for each taper
                Cpcomp[:,:,ktap],Cpcompn[:,:,ktap],Ncomp=\
                    self._cpcomp_main_calc(xc,self.wgts)


                if ktap==self.Ntap-1:
                    # and calculate uncertainties if we're finished
                    self._cpcomp_unc_calc(Cpcomp,Cpcompn,Ncomp)



#--------------------BEGIN FOR BOOTSTRAPPING-------------------------------

    def _pick_bootstrap_stations(self,statok,Nsboot=20):
        """
        :param         statok: a list of acceptable stations
        :param         Nsboot: how many samplings
        :return           mtx: of matrix of which stations to use in each sample
        """

        # maximum number of stations originally
        ntot=len(statok)
        statok=np.where(statok)[0]

        # in case we don't want to pick all stations
        npick=int(np.ceil(statok.size*0.8))

        # these are the weights for each station for the bootstrap selections
        mtx=np.array([np.bincount(np.random.choice(statok,npick,replace=False),
                                  minlength=ntot)
                      for kb in range(0,Nsboot)]).T.astype(bool)
        if self.params['returnjack']:
            Cp['ijack_stat']=mtx
        mtx=np.append(np.ones([ntot,1],dtype=bool),mtx,axis=1)

        return mtx


#--------------------END FOR BOOTSTRAPPING-------------------------------


#-------------------BEGIN CALCULATING CP VALUES------------------


    def cpstat_unc_calc(self,Cpstat,Cpstatn,Nstat):
        """
        :param        Cpstat: inter-station Cp
        :param       Cpstatn: the normalizations
        :param         Nstat: number of stations per value
        """


        # number of components per interval
        Nci = np.sum(Nstat>=2,axis=2).astype(float)
        Nci = np.ma.masked_array(Nci,mask=Nci==0)
        Nci = Nci.reshape(list(Nci.shape)+[1])

        # go ahead and normalize everything
        Cpstat = np.divide(Cpstat,Nci)

        # average denominator over taper before dividing
        Cpstatn = np.ma.mean(Cpstatn,axis=2,keepdims=True)
        Cpstat=np.divide(Cpstat,Cpstatn)

        # note the bootstrap matrix
        mtx=self.bootsmtx
        Nsboot=mtx.shape[1]-1


        # note whether we want taper uncertainties
        tapunc=self.params['tapunc']

        # get uncertainty
        if Nsboot>=3 and self.Ntap>1 and tapunc:
            # sum bootstrap and taper uncertainties

            Cpstat_std_tap=np.ma.var(Cpstat[:,0,:],axis=1)/self.Ntap
            Cpstat_std_boot=np.ma.mean(Cpstat[:,1:,:],axis=2)

            if self.params['returnjack']:
                self.Cp['Cpstat_jack']=Cpstat_std_boot
            Cpstat_std_boot=np.ma.var(Cpstat_std_boot,axis=1)
            Cpstat_std=np.power(Cpstat_std_boot+Cpstat_std_tap,0.5)

        elif Nsboot>=3:
            # just bootstrap uncertainty

            # taper-averaged bootstrap values
            Cpstat_std=np.ma.mean(Cpstat[:,1:,:],axis=2)

            # save output
            if self.params['returnjack']:
                self.Cp['Cpstat_jack']=Cpstat_std

            # get std
            Cpstat_std=np.ma.std(Cpstat_std,axis=1)

        elif self.Ntap>1:
            # just taper uncertainty

            Cpstat_std=np.ma.std(Cpstat[:,0,:],axis=1)/self.Ntap**0.5
        else:
            # no uncertainty estimate

            Cpstat_std=np.ma.masked_array(np.zeros(Nt,dtype=float),
                                          mask=np.ones(Nt,dtype=bool))

        # and the best estimate, with all stations
        Cpstat = np.ma.mean(Cpstat[:,0,:],axis=1)


        self.Cp['Cpstat'] = Cpstat
        self.Cp['Nstat']  = Nstat
        self.Cp['Cpstat_std'] = Cpstat_std

        return




    def _cpstat_main_calc(self,xc,wgts):
        """
        :param         xc: cross-correlation values
        :param       wgts: weightings
        """

        # Get some parameters
        Nt      = xc.shape[1]  # number of indices
        Nc      = self.stations['Nc']
        icmp    = self.stations['icmp']
        ins     = self.stations['ins']
        dok     = self.dok
        mtx     = self.bootsmtx
        Nsboot  = mtx.shape[1]-1


        # sum, separated by station
        Nstat=np.zeros([Nt,Nsboot+1,Nc])
        Cpstat=np.zeros([Nt,Nsboot+1])
        Cpstatn=np.zeros([Nt,Nsboot+1])

        # compute inter-station coherence for each component
        for ks in range(0,Nc):

            # at each component
            ii=icmp==ks
            mmult=mtx[ins[ii],:]

            if sum(ii)>1:
                # coherence for this component
                Rstati,Rstatn,nn=self._calcCp(xc=xc[:,:,ii],ii=None,dok=dok[:,ii],
                                              wgts=wgts[:,:,ii],mmult=mmult)


                # add to set
                Cpstat=Cpstat+Rstati
                Cpstatn=Cpstatn+Rstatn
                Nstat[:,:,ks]=Nstat[:,:,ks]+nn


        return Cpstat,Cpstatn,Nstat

    def _cpcomp_main_calc(self,xc,wgts):
        """
        :param         xc: cross-correlation values
        :param       wgts: weightings
        """

        # Get some parameters
        Nt      = xc.shape[1]  # number of indices
        Nc      = self.stations['Nc']
        icmp    = self.stations['icmp']
        ins     = self.stations['ins']
        dok     = self.dok
        nper    = self.stations['nper']

        # and consider components per station
        Ncomp=np.zeros([Nt,len(nper)])
        Cpcomp=np.zeros([Nt,len(nper)])
        Cpcompn=np.zeros([Nt,len(nper)])

        # for components
        for ks in range(0,len(nper)):
            # at each station
            ii=ins==ks

            if sum(ii)>1:
                # coherence for this component
                Cpcomp[:,ks],Cpcompn[:,ks],Ncomp[:,ks]=\
                    self._calcCp(xc,ii,dok,wgts)

        return Cpcomp,Cpcompn,Ncomp


    def _cpcomp_unc_calc(self,Cpcomp,Cpcompn,Ncomp,Ncboot=20):
        """
        :param        Cpcomp: inter-component coherence by station
        :param       Cpcompn: normalization factors
        :param         Ncomp: number of components
        :param        Ncboot: number of bootstrap resamplings
        """

        # trash stations with no information
        Ncompi=Ncomp>=2
        statok=np.sum(Ncompi,axis=0)>0
        Ncompi,Cpcomp=Ncompi[:,statok],Cpcomp[:,statok,:]
        Cpcompn=Cpcompn[:,statok,:]
        nsta=Cpcomp.shape[1]

        # for normalization
        Nci = np.sum(Ncompi,axis=1).astype(float)

        # choose various subsets of the stations to bootstrap
        # we'll just assume the same station distribution for all times
        if Ncboot>=3 and nsta>1:
            # which bootstrap stations
            npick=int(nsta*.8)
            mtx=np.array([np.bincount(np.random.choice(nsta,npick,replace=False),
                                      minlength=nsta)
                          for kb in range(0,Ncboot)]).T

            # correct normalization
            Cpcomp_std=np.ma.mean(Cpcomp,axis=2)
            Cpcomp_std=np.ma.dot(Cpcomp_std,mtx)


            if not np.isscalar(Cpcompn):
                Cpcomp_stdn=np.ma.dot(np.mean(Cpcompn,axis=2),mtx)
            else:
                Cpcomp_stdn=Cpcompn

            # normalize and average over stations
            Cpcomp_std=np.ma.divide(np.ma.divide(Cpcomp_std,Cpcomp_stdn),
                                    np.dot(Ncompi,mtx))


            # save the jackknife results if desired
            if self.params['returnjack']:
                self.Cp['Cpcomp_jack']=Cpcomp_std
                self.Cp['ijack_comp']=mtx

            # and to variance
            Cpcomp_std=np.ma.var(Cpcomp_std,axis=1)

            # add tapering?
            if self.Ntap>1 and self.params['tapunc']:
                Cpcomp_std_tap=np.divide(np.ma.mean(Cpcomp,axis=1),
                                         np.ma.mean(Cpcompn,axis=1))
                Cpcomp_std_tap=np.ma.var(Cpcomp_std_tap,axis=1)/self.Ntap
                Cpcomp_std_tap = np.ma.divide(Cpcomp_std_tap,Nci)
                Cpcomp_std=np.power(Cpcomp_std+Cpcomp_std_tap,0.5)
            else:
                Cpcomp_std=np.power(Cpcomp_std,0.5)
        elif self.Ntap>1:
            # just taper-dependent uncertainty
            Cpcomp_std = np.divide(np.ma.mean(Cpcomp,axis=1),
                                   np.ma.mean(Cpcompn,axis=1))
            Cpcomp_std = np.ma.std(np.ma.mean(Cpcomp_std,axis=1),axis=1)
            Cpcomp_std = np.ma.divide(Cpcomp_std,Nci*Ntap**0.5)
        else:
            Cpcomp_std=np.ma.masked_array(np.zeros(Nt,dtype=float),mask=True)

        # normalize the average inter-component coherence
        Cpcomp=np.ma.mean(Cpcomp,axis=2)
        Cpcompn=np.ma.mean(Cpcompn,axis=2)
        Cpcomp=np.divide(np.ma.sum(Cpcomp,axis=1),
                         np.ma.sum(Cpcompn,axis=1))
        Cpcomp = np.ma.divide(Cpcomp,Nci)

        self.Cp['Cpcomp'] = Cpcomp
        self.Cp['Ncomp']  = Ncomp
        self.Cp['Cpcomp_std'] = Cpcomp_std

        return


    def _calcCp(self,xc,ii=None,dok=None,wgts=None,mmult=None,xc2=None,wgts2=None):
        """
        calculate the coherence or coherent energy given the cross-spectra and weights
        :param         xc: all the cross-spectra
        :param         ii: a boolean list of the components/stations to use
        :param        dok: which cross-spectra are acceptable to use
        :param       wgts: any weighting for the cross-spectra
        :param      mmult: a weighting to multiply each station/component by (default: all ones)
        :param        xc2: np.abs(xc)**2, if precomputed
        :param      wgts2: np.abs(wgts)**2, if precomputed
        :return    Rstati: the coherence for this interval
        :return        nn: the number of stations/components used at each time
        """


        # number of frequencies and times
        Nf = xc.shape[0]
        Nt = xc.shape[1]


        if ii is None and mmult is None:
            # default is to use all the stations
            mmult=np.ones(xc.shape[2],dtype=bool)
        elif mmult is None:
            # a weighting per station, if needed
            mmult=ii
        mmult=np.atleast_1d(mmult)


        # note number of dimensions
        if mmult.ndim==1:
            mmult=mmult.reshape([mmult.size,1])
            ndim=1
        else:
            ndim=2

        # compute phase walkout
        Rstati=np.abs(np.dot(xc,mmult))

        # count the number of stations per window
        nn=np.dot(dok,mmult.astype(float))
        nni=np.ma.masked_array(nn,mask=nn<2)

        # value to subtract depends on normalization
        if self.params['normtype']=='full':
            Rsub=1.
        else:
            if xc2 is None:
                xc2=np.power(np.abs(xc),2)
            Rsub=np.ma.divide(np.dot(xc2,mmult),nni)

        # to phase coherence
        Rstati=np.divide(np.divide(np.power(Rstati,2),nni)-Rsub,nni-1.)

        # average over frequencies
        Rstati=np.ma.mean(Rstati,axis=0).reshape([Nt,mmult.shape[1]])

        # if the weights vary with time and location, need to
        # sum the weighted template

        if 'bounded' in self.params['normtype']:
            # average without taking absolute value (Phase walkout)
            Rstatn=np.abs(np.dot(wgts,mmult))

            # subtract mean
            if wgts2 is None:
                wgts2=np.power(np.abs(wgts),2)
            Rsub=np.ma.divide(np.dot(wgts2,mmult),nni)

            # to phase coherence
            Rstatn=np.ma.divide(np.ma.divide(np.power(Rstatn,2),nni)-Rsub,nni-1.)

            # average over frequencies
            Rstatn=np.ma.mean(Rstatn,axis=0).reshape([Nt,mmult.shape[1]])

        else:
            # note a single normalization
            Rstatn=1.

        # check if there are enough stations
        # and set masked values to zero
        if isinstance(Rstati,np.ma.masked_array):
            Rstati=Rstati.data
            Rstati[nni.mask]=0.
            nn[nni.mask]=0.

        # back to single dimension if desired
        if ndim==1:
            Rstati,nn=Rstati.flatten(),nn.flatten()
            if not np.isscalar(Rstatn):
                Rstatn=Rstatn.flatten()

        return Rstati,Rstatn,nn


    # -------------------------------------------------------------------------------
    def _getCommonTraces(self):

        '''
        Get common traces between the template and station
        return:
                * self.stations : dictionnary containing some variables used after
                    -> Nc       : number of distinct components (1, 2, ou 3)
                    -> icmp     : For each trace, indice if which compo it is
                    -> ins      : For each trace, get station number
                    -> nper     : For each trace, get number of traces
                    -> nscT     : Traces in template
                    -> nscD     : Traces in data
                    -> CommonTr : List of traces in both template and data
        '''

        # Networks, stations, components to compare
        nsc1=np.array([tr.stats.network+'.'+tr.stats.station+'.'+
                       tr.stats.channel[-1] for tr in self.template])
        nsc2=np.array([tr.stats.network+'.'+tr.stats.station+'.'+
                       tr.stats.channel[-1] for tr in self.data])
        nsc=np.intersect1d(nsc1,nsc2)
        nsc=np.sort(nsc)

        # split by station
        ns=np.array([vl.split('.')[0]+'.'+vl.split('.')[1] for vl in nsc])
        nsa,ins,nper=np.unique(ns,return_inverse=True,return_counts=True)

        # split by component
        icmp,cmps = self._groupcomp(nsc)
        Nc=len(cmps)

        stations = {'Nc':Nc, 'icmp':icmp, 'ins':ins, 'nper':nper, \
                    'nscT':nsc1, 'nscD':nsc2, 'CommonTr':nsc}

        self.stations = stations

        # all done
        return

    # -------------------------------------------------------------------------------
    def _groupcomp(self,nsc):
        """
        :param     nsc: list of network.station.component
        :return   icmp: index of components for each
        :return   cmps: components considered
        """

        # define groups
        cmps = np.array(['E1','N2','Z3'])

        # initialize
        icmp = np.ndarray(len(nsc),dtype=int)

        for k in range(0,len(nsc)):
            # split component
            vl = nsc[k].split('.')
            vl = vl[-1][-1]

            for m in range(0,len(cmps)):
                if vl in cmps[m]:
                    icmp[k]=m

        # only components that were used
        ii,icmp=np.unique(icmp,return_inverse=True)
        cmps=cmps[ii]

        # all done
        return icmp,cmps


    # -------------------------------------------------------------------------------
    def _expstd(self,Nstat,Nave):
        """
        :param    Nstat:  number of stations used
                    could have a multiple columns if averaged over
                    components
        :param     Nave:  number of values averaged over after
        :return    stde:  expected std
        """

        # number of stations
        if Nstat.ndim>1:
            # number of components
            Nc = Nstat.shape[1]
            # average stations per component?
            # this really shouldn't change
            Nstat = np.mean(Nstat,axis=1)
        else:
            # just one component
            Nc = 1

        # number of pairs
        Np = np.multiply(Nstat,Nstat-1)/2

        # total number of values
        Ntot = Np*Nc*Nave
        Ntot = np.atleast_1d(Ntot).astype(float)

        # std
        stde = np.divide(1.,2.*Ntot)
        stde = np.power(stde,0.5)

        return stde


    # -------------------------------------------------------------------------------
    def _removeUnusedTraces(self,verbose=True):
        '''
        Remove traces which are neither in template or data
        '''

        # Check something
        assert(hasattr(self,'stations')), 'You need to compute stations info first'

        # Create empty obspy stream to fill with template
        st1i = obspy.Stream()

        # Select only used traces
        bo1 = np.isin(self.stations['nscT'],self.stations['CommonTr'])
        ix1 = np.where(bo1)[0]
        [st1i.append(self.template[i]) for i in ix1]

        if verbose:
            removed = self.stations['nscT'][~bo1]
            if removed.size==0:
                print('No traces removed in template')
            else:
                t=[print('Trace {} removed from template'.format(r)) for r in removed]


        # Create empty obspy stream to fill with data
        st2i = obspy.Stream()

        # Select only used traces
        bo2 = np.isin(self.stations['nscD'],self.stations['CommonTr'])
        ix2 = np.where(bo2)[0]
        [st2i.append(self.data[i]) for i in ix2]

        if verbose:
            removed = self.stations['nscD'][~bo2]
            if removed.size==0:
                print('No traces removed in data')
            else:
                t=[print('Trace {} removed from data'.format(r)) for r in removed]

        # Put them in current streams
        self.template = st1i
        self.data     = st2i

        # All done
        return

    # -------------------------------------------------------------------------------
    def _resampleTraces(self,verbose=True):
        '''
        Resample data and template to the same time spacing of necessary
        '''

        # resample to the same time spacing if necessary
        dtim=[tr.stats.delta for tr in self.template]+[tr.stats.delta for tr in self.data]
        dtim=np.unique(np.array(dtim))

        if len(dtim)>1:
            if verbose:
                print('Resampling to common interval')
            self.template=self.template.resample(sampling_rate=1./min(dtim),no_filter=True)
            self.data=self.data.resample(sampling_rate=1./min(dtim),no_filter=True)

        # Save it
        self.dtim = min(dtim)

        # All done
        return


    # -------------------------------------------------------------------------------
    def _makeGrid(self,shgrid=None,shtry=None):
        '''
        Compute shcalc, ishfs, and Nst
        '''

        assert(hasattr(self,'params')),'You must set parameters with setParams() first'
        shgrid = self.params['shgrid']
        shtry  = self.params['shtry']
        nsc    = self.stations['CommonTr']

        # time shifts to calculate
        shcalc = dict((nsci,np.array([0.])) for nsci in nsc)
        for ky in shcalc.keys():
            shtry[ky] = np.atleast_1d(shtry[ky])
            vl = self._minmax(shtry[ky])
            nvl = int(np.ceil(np.diff(vl)[0]/shgrid))
            nvl = np.maximum(nvl,1)
            shcalc[ky] = np.linspace(vl[0],vl[1],nvl)

        # maximum number of shifts to calculate for each station
        # because we'll make a list with one x-c for each station/shift
        Nst=np.sum(np.array([len(vl) for vl in shcalc.values()]))
        i1=0
        ishfs = dict((nsci,np.array([0])) for nsci in nsc)
        for nsci in nsc:
            ishfs[nsci] = np.arange(0,len(shcalc[nsci]))+i1
            i1=i1+len(shcalc[nsci])

        # Save it
        self.Nst = Nst

        # All done
        return shcalc, ishfs

    # -------------------------------------------------------------------------------
    def _minmax(self,x,bfr=1.):
        """
        :param      x:   set of values
        :param    bfr:   how much to multiply the limits by (default: 1.)
        :return   lms:   limits
        """

        # minmax
        lms = np.array([np.min(x),np.max(x)])

        if bfr!=1.:
            lms = np.mean(lms)+np.diff(lms)[0]*bfr*np.array([-.5,.5])

        return lms


    # ---------------------------------------------------------------------------

    def plot(self,toplot=['seismograms','Cpstat','Cpcomp'],p=None,tlm=None):
        """
        plot the results of the coherence analysis
        :param       toplot: a list of the parameters to plot
        :param            p: handles to the plots to use (default: created)
        :param          tlm: time limit to plot (default: all of tlook)
        """

        toplot=np.atleast_1d(toplot)
        Np=len(toplot)

        # make plots if needed
        if p is None:
            f = plt.figure(figsize=(10,8))
            gs,p=gridspec.GridSpec(Np,1),[]
            gs.update(left=0.1,right=0.98)
            gs.update(bottom=0.07,top=0.97)
            gs.update(hspace=0.1,wspace=0.1)
            for gsi in gs:
                p.append(plt.subplot(gsi))
        p=np.atleast_1d(p)

        # note times
        tms=self.Cp['tlook']
        if tlm is None:
            tlm=np.array([np.min(tms),np.max(tms)])

        # go through each plot
        for k in range(0,Np):
            ph=p[k]

            if toplot[k] in ['seismograms']:
                self.plot_seismic(st='target',tlm=tlm,p=ph)

            else:
                # identify the data of interest
                if toplot[k] in ['Cpstat','cpstat']:
                    data=self.Cp['Cpstat']
                    unc=self.Cp['Cpstat_std']
                    lbl='$C_p^{sta}$'
                elif toplot[k] in ['Cpcomp','cpcomp']:
                    data=self.Cp['Cpcomp']
                    unc=self.Cp['Cpcomp_std']
                    lbl='$C_p^{cmp}$'

                # error bars on polygon
                x=np.append(tms,np.flipud(tms))
                fct=1.
                y=np.append(data+fct*unc,np.flipud(data-fct*unc))
                ply = Polygon(np.vstack([x,y]).transpose())
                ply.set_edgecolor('none')
                ply.set_color('lightblue')
                ply.set_alpha(0.7)
                ph.add_patch(ply)

                # set y limits to something sensible
                iok=np.logical_and(tms>=tlm[0],tms<=tlm[1])
                ylm=np.array([np.ma.min(data[iok]-fct*unc[iok]),
                              np.ma.max(data[iok]+fct*unc[iok])])
                ylm=ylm+np.array([-1,1])*.1*np.diff(ylm)[0]
                ph.set_ylim(ylm)

                # best-fitting value
                ph.plot(tms,data,color='navy')
                ph.plot(tlm,[0,0],color='k',linestyle='--',zorder=0)

                # set labels
                ph.set_ylabel(lbl)

        for ph in p:
            ph.set_xlim(tlm)
            ph.set_xlabel('time since reference (s)')

    def plot_seismic(self,st='data',pk=None,tlm=None,istat=None,nmax=5,p=None):

        if st is None:
            st='data'
        if isinstance(st,str):
            if st in ['data','look','target']:
                st=self.data
                if pk is None:
                    pk=self.params['shlook']
            elif st in ['template','temp']:
                st=self.template
                if pk is None:
                    pk=self.params['shtemp']
        if pk is None:
            pk=self.params['shlook']

        if istat is None:
            istat=np.random.choice(len(st),nmax,replace=False)
        if tlm is None:
            tlm=np.array([np.min(self.Cp['tlook']),
                          np.max(self.Cp['tlook'])])
            tlm=tlm+np.array([-1,1])*10
        sti=obspy.Stream()

        for ix in istat:
            sti.append(st[ix])

        self.plot_seismic_help(st=sti,pk=pk,tlm=tlm,p=p)

    def plot_seismic_help(self,st,pk,tlm,p=None):

        if p is None:
            f=plt.figure()
            p=plt.axes()

        shf=1.
        amp=0.5

        h=[]
        cols=graphical.colors(len(st))
        for k in range(0,len(st)):
            tr=st[k].copy()
            tref=tr.stats.starttime+tr.stats[pk]
            tr.trim(starttime=tref+tlm[0],endtime=tref+tlm[1],pad=True)
            tshf=tr.stats.starttime-tref
            data=tr.data/np.max(np.abs(tr.data))
            hh,=p.plot(tr.times()+tshf,data*amp-shf*k,color=cols[k])
            h.append(hh)

        p.set_xlim(tlm)
        p.set_ylim([-shf*(len(st)-1)-amp*1.2,amp*1.2])
        p.set_yticks([])

        return h
