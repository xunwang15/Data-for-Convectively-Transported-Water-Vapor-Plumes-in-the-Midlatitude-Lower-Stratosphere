import xarray as xr
import pandas as pd
#from xradd import *
import matplotlib.pyplot as plt
import numpy as np
from string import ascii_lowercase
degree_sign= u'\N{DEGREE SIGN}'
import time


### read horizontal size data
#ff = xr.open_dataset('h2o_plume_horizontal_size.nc')
ff = pickle.load( open('h2o_plume_horizontal_size.pickle', 'rb') )
lonlms = ff.lonlms.data
latlms = ff.latlms.data
lonovw = ff.lonovw.data
latovw = ff.latovw.data
lonall = ff.lonall.data
latall = ff.latall.data
alt = ff.alt.data
time=ff.time.data
nalt = np.size(alt)

### read mass data
#ff=xr.open_dataset('mass_h2o_ice_ovw_lms.nc')
ff = pickle.load( open('mass_h2o_ice_ovw_lms.pickle', 'rb') )
massovws = ff.massovw_t.data
massalls= ff.massall_t.data
masslmss = ff.masslms_t.data
micealls = ff.massiceall_t.data

### read ndf
hist1=ff.massall_ndf.data
hist2=ff.massovw_ndf.data
hist3=ff.masslms_ndf.data
bins = ff.bins.data



#### generate Figure

fig,ax = plt.subplots(nrows=3,ncols=1)
fig.set_size_inches(8,15, forward=True)
xx = np.arange(0,721,1)
xxt = np.arange(0,13,1)

#### first row: mass pdf
title = ['height','top','bottom']

ax[0].fill_between(bins,hist1/np.nansum(hist1)*100,color='grey',alpha=0.2,label='MH2O')
ax[0].plot(bins,hist2/np.nansum(hist1)*100,color='darkorange',label='OVW MH2O',linewidth=2)
ax[0].plot(bins,hist3/np.nansum(hist1)*100,color='g',label='LMS MH2O',linewidth=2)
ax[0].set_xlabel('MH2O (kg) per grid area ($1km^2$)',fontsize=12)
ax[0].set_ylabel('percentage (%)',fontsize=15)
ax[0].legend(loc='lower left',bbox_to_anchor=(0.04,0.02))
ax[0].set_yscale('log')
ax[0].set_xscale('log')

#### second row: mass time series
ax[1].plot(massalls,color='k',label='total MH2O')
ax2 = ax[1].twinx()
ax2.set_ylabel('total ice mass (kg)',fontsize=14)
ax2.plot(micealls,color='grey',label='total ice mass',linestyle='--')
ax[1].plot(massbelows,color='g',label='total LMS MH2O')
ax[1].plot(masslofts,color='orange',label='total OVW MH2O')

ax[1].set_xticks(xx[::60])
ax[1].set_xticklabels(np.array([str(i)+' hr' for i in xxt+12]))
ax[1].set_xlim([0,720])
ax[1].set_ylabel('MH2O (kg)',fontsize=14)
ax[1].legend(fontsize=12)
ax2.legend(loc='center left',bbox_to_anchor=(0,0.6),fontsize=12)

###### third row: horizontal size ########
xx = np.arange(0,721,1)
xxt = np.arange(0,13,1)
title = ['all','below','loft']
ilev=10
ax[2].plot(xx[:-1],lonbelow[:,ilev],label='zonal diameter at '+str(alt[ilev])+' km',color='g')
ax[2].plot(xx[:-1],latbelow[:,ilev],label='meridional diameter '+str(alt[ilev])+' km',color='g',linestyle=':')

ilev=14
ax[2].plot(xx[:-1],lonloft[:,ilev],label='zonal diameter'+str(alt[ilev])+' km',color='orange')
ax[2].plot(xx[:-1],latloft[:,ilev],label='meridional diameter'+str(alt[ilev])+' km',color='orange',linestyle=':')

ax[2].legend(fontsize=12)
ax[2].set_xticks(xx[::60])
ax[2].set_xticklabels(np.array([str(ii)+' hr' for ii in xxt+12]))

ax[2].set_xlim([0,720])
ax[2].set_ylim([-10,500])
ax[2].set_ylabel('km',fontsize=12)


llabel = ascii_lowercase[:3]
for i in range(3):
    ax[i].annotate('('+llabel[i]+')',xy=(0.98,0.98),xycoords='axes fraction',size=14,ha='right',va='top',color='k')
#plt.savefig('fig5.pdf',bbox_inches='tight')
