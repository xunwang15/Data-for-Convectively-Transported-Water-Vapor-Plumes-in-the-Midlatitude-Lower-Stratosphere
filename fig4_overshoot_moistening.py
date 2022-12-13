import xarray as xr
import pandas as pd
#from xradd import *
import matplotlib.pyplot as plt
import numpy as np
from string import ascii_lowercase
degree_sign= u'\N{DEGREE SIGN}'
import time

#### read data
#f=xr.open_dataset('overshooting_selected_unstable_tke_wv_theta_record.nc')
f= pickle.load( open('overshooting_selected_unstable_tke_wv_theta_record.pickle', 'rb') )
ndthdz2=f.ndthdz2.data
tropst = f.trop.data
troptheta = f.troptheta.data
nwvanom2 =f.nwvanom2.data
nice2 = f.nice2.data
nthwv2 = f.nthwv2.data
nthth2=f.nthth2.data
nthice2=f.nthice2.data
tropall = tropst[:,:,1].flatten()
tropthall = troptheta[:,:,1].flatten()

#### generate figure

fig,ax = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(12,5, forward=True)

cc = ['tab:blue','tab:orange','tab:green']
title = [' max altitude','max potential temperature']

### number vs altitude
idx = np.where(np.nansum(np.nansum(ndthdz2,axis=0),axis=0)>=1)[0]
ax[0].plot(np.nansum(np.nansum(ndthdz2,axis=0),axis=0)[idx],alt[idx],label='TKE>=10 $m^2$ $s^{-2}$\n'+'$d\Theta$/dz<=0.01 $K m^{-1}$',color=cc[0])

idx = np.where(np.nansum(np.nansum(nwvanom2,axis=0),axis=0)>=1)[0]
ax[0].plot(np.nansum(np.nansum(nwvanom2,axis=0),axis=0)[idx],alt[idx],label='significant moistening',color=cc[1])

idx = np.where(np.nansum(np.nansum(nice2[:,:,:,1],axis=0),axis=0)>=1)[0]
ax[0].plot(np.nansum(np.nansum(nice2[:,:,:,1],axis=0),axis=0)[idx],alt[idx],label='IWC>=1e-6 $g m^{-3}$',color=cc[2])
ax[0].set_xscale('log')

### number vs potential temperatrue
idx = np.where(np.nansum(np.nansum(nthth2,axis=0),axis=0)>=1)[0]
ax[1].plot(np.nansum(np.nansum(nthth2,axis=0),axis=0)[idx],thwvbin[1:][idx]-1.25,label='TKE>=10 $m^2$ $s^{-2}$\n'+'$d\Theta$/dz<=0.01 $K m^{-1}$',color=cc[0])

idx = np.where(np.nansum(np.nansum(nthwv2,axis=0),axis=0)>=1)[0]
ax[1].plot(np.nansum(np.nansum(nthwv2,axis=0),axis=0)[idx],thwvbin[1:][idx]-1.25,label='significant moistening',color=cc[1])

idx = np.where(np.nansum(np.nansum(nthice2[:,:,1,:],axis=0),axis=0)>=1)[0]
ax[1].plot(np.nansum(np.nansum(nthice2[:,:,1,:],axis=0),axis=0)[idx],thwvbin[1:][idx]-1.25,label='IWC>=1e-6 $g m^{-3}$',color=cc[2])
ax[1].set_xscale('log')

### plot mean tropopause
idx = np.where(tropall==0)[0]
tropall = np.delete(tropall,idx)
ax[0].plot([0,1e8],[np.nanmean(tropall),np.nanmean(tropall)],color='k',label='tropopause')
ax[0].plot([0,1e8],[np.nanmean(tropall)+np.nanstd(tropall),np.nanmean(tropall)+np.nanstd(tropall)],color='k',linestyle='--')
ax[0].plot([0,1e8],[np.nanmean(tropall)-np.nanstd(tropall),np.nanmean(tropall)-np.nanstd(tropall)],color='k',linestyle='--')

idx = np.where(tropthall==0)[0]
tropthall = np.delete(tropthall,idx)
ax[1].plot([0,1e8],[np.nanmean(tropthall),np.nanmean(tropthall)],color='k',label='tropopause')
ax[1].plot([0,1e8],[np.nanmean(tropthall)+np.nanstd(tropthall),np.nanmean(tropthall)+np.nanstd(tropthall)],color='k',linestyle='--')
ax[1].plot([0,1e8],[np.nanmean(tropthall)-np.nanstd(tropthall),np.nanmean(tropthall)-np.nanstd(tropthall)],color='k',linestyle='--')


ax[1].legend(fontsize=11,loc='center left',bbox_to_anchor=(1.1,0.5),ncol=1)
ax[0].set_xlabel('grid point number',fontsize=14)
ax[1].set_xlabel('grid point number',fontsize=14)
ax[0].set_ylabel('Altitude (km)',fontsize=14)
ax[1].set_ylabel('Potential tempertaure (K)',fontsize=12)
ax[0].set_xlim([0,1e8])
ax[1].set_xlim([0,1e8])


llabel = ascii_lowercase[:12]
k=0
for i in range(2):
    ax[i].annotate('('+llabel[k]+')',xy=(0.98,0.98),xycoords='axes fraction',size=14,ha='right',va='top',color='k')
    k+=1
plt.savefig('fig4.pdf',bbox_inches='tight')
