"""
This program reads water vapor and other meteorological data from the GEM.
We compute and plot the water vapor anomaly at 16.5 km, 15.5 km, and 48.13 N.
We also plot contours of IWC, 380-K potential temperature, and thermal tropopause.
"""

import xarray as xr
import pandas as pd
#from xradd import *
import matplotlib.pyplot as plt
import numpy as np
from string import ascii_lowercase
degree_sign= u'\N{DEGREE SIGN}'
import time

def find_missing(lst):
    return [x for x in range(lst[0], lst[-1]+1) if x not in lst]

starttime = '2013-08-25 12:01:00'
ntime_all = int(12*60)
timeseries = np.arange(1,721,1)
# find the time point where the data is missing, remove this datatime value
datatime = pd.date_range(starttime, periods=ntime_all, freq="1min")
missing_index = find_missing(timeseries)
datatime = np.delete(datatime,missing_index)
itime1 = np.where(datatime==pd.Timestamp(2013, 8, 25, 18, 0))[0][0]


###### read water vapor and variables at the four time steps of the simulation, 2 hours apart
#f1 = xr.open_dataset('h2o_iwc_theta_trop_48N.nc')
#f2 = xr.open_dataset('h2o_iwc_15p5_16p5_km.nc')
#f3 = xr.open_dataset('h2o_zonal_mean_h2o_std.nc')
f1 = pickle.load( open('h2o_iwc_theta_trop_48N.pickle', 'rb') )
f2 = pickle.load( open('h2o_iwc_15p5_16p5_km.pickle', 'rb') )
f3 = pickle.load( open('h2o_zonal_mean_h2o_std.pickle', 'rb') )


lon = f2.lon.data
lat = f2.lat.data
alt = f1.alt.data
nlat = np.size(lat)
nlon = np.size(lon)
nalt= np.size(alt)

wvmean = f3.wvmean.data
wvstd = f3.wvstd.data

h2o1 = f2.h2o.data
iwc1 = f2.iwc.data
h2o2 = f1.h2o.data
iwc2 = f1.iwc.data
theta =f1.theta.data
trop = f1.trop.data
####### generate Figure 1


fig,ax = plt.subplots(nrows=3,ncols=4)
fig.set_size_inches(20,12, forward=True)
levels = np.arange(-25,26,1)

ilev1=np.where(alt==16.5)[0][0]
ilev2=np.where(alt==15.5)[0][0]
ilat =np.where(lat>=48.35)[0][0]
for i in range(4):
    ima = ax[0][i].contourf(lon,lat,h2o1[i,1,:]-wvmean[ilev1,:,np.newaxis],levels=levels,cmap='bwr',extend='both')
    ax[1][i].contourf(lon,lat,h2o1[i,0,:]-wvmean[ilev2,:,np.newaxis],levels=levels,cmap='bwr',extend='both')
    ax[0][i].contour(lon,lat,iwc1[i,1,:],levels=[1e-6],colors='cyan',linewidths=1)
    ax[1][i].contour(lon,lat,iwc1[i,0,:],levels=[1e-6],colors='cyan',linewidths=1)
    ax[0][i].contour(lon,lat,h2o1[i,1,:]-wvmean[ilev1,:,np.newaxis]-3*wvstd[ilev1,:,np.newaxis],\
                 levels=[0],colors='k',linewidths=0.8,linestyles=':')
    ax[1][i].contour(lon,lat,h2o1[i,0,:]-wvmean[ilev2,:,np.newaxis]-3*wvstd[ilev2,:,np.newaxis],\
                 levels=[0],colors='k',linewidths=0.8,linestyles=':')
    ax[0][i].plot(lon,lat[ilat]*np.ones_like(lon),linewidth=1,color='k')
    ax[1][i].plot(lon,lat[ilat]*np.ones_like(lon),linewidth=1,color='k')
    ax[2][i].contourf(lon,alt,h2o2[i,:]-wvmean[:,ilat,np.newaxis],levels=levels,cmap='bwr',extend='both')
    ax[2][i].contour(lon,alt,iwc2[i,:],levels=[1e-6,1e-3],colors='cyan',linewidths=0.8)
    ax[2][i].contour(lon,alt,theta[i,:],levels=[381],colors='k',linewidths=1,linestyles='--')
    ax[2][i].contour(lon,alt,h2o2[i,:]-wvmean[:,ilat,np.newaxis]-3*wvstd[:,ilat,np.newaxis],\
                 levels=[0],colors='k',linewidths=0.8,linestyles=':')
    ax[2][i].plot(lon,trop[i,:],color='k',linewidth=0.8)
    ax[0][i].annotate('16.5 km',xy=(0.98,0.98),xycoords='axes fraction',size=10,ha='right',va='top',color='k')
    ax[1][i].annotate('15.5 km',xy=(0.98,0.98),xycoords='axes fraction',size=10,ha='right',va='top',color='k')
    ax[2][i].annotate(str(round(lat[ilat],2))+degree_sign+'N',xy=(0.98,0.98),xycoords='axes fraction',size=10,ha='right',va='top',color='k')
    ax[0][i].set_title(datatime[itime1+i*120])
    ax[0][i].set_ylabel('Latitude',fontsize=10)
    ax[0][i].set_xlabel('Longitude',fontsize=10)
    ax[1][i].set_ylabel('Latitude',fontsize=10)
    ax[1][i].set_xlabel('Longitude',fontsize=10)
    ax[2][i].set_ylabel('Altitude (km)',fontsize=10)
    ax[2][i].set_xlabel('Longitude',fontsize=10)
    ax[2][i].set_ylim([13,20])


### colorbars
cbaxes = fig.add_axes([0.91,0.67,0.01,0.2])                             # setup colorbar
cc = fig.colorbar(ima, ax=ax[0][-1],cax=cbaxes)
cc.set_ticks(levels[::5])
clabels = np.array([str(round(i,2)) for i in levels[::5]])
cc.set_ticklabels(clabels)
cc.ax.tick_params(labelsize=8)
cc.set_label('ppmv',fontsize=14)

cbaxes = fig.add_axes([0.91,0.4,0.01,0.2])                             # setup colorbar
cc = fig.colorbar(ima, ax=ax[0][-1],cax=cbaxes)
cc.set_ticks(levels[::5])
clabels = np.array([str(round(i,2)) for i in levels[::5]])
cc.set_ticklabels(clabels)
cc.ax.tick_params(labelsize=8)
cc.set_label('ppmv',fontsize=14)

cbaxes = fig.add_axes([0.91,0.14,0.01,0.2])                             # setup colorbar
cc = fig.colorbar(ima, ax=ax[0][-1],cax=cbaxes)
cc.set_ticks(levels[::5])
clabels = np.array([str(round(i,2)) for i in levels[::5]])
cc.set_ticklabels(clabels)
cc.ax.tick_params(labelsize=8)
cc.set_label('ppmv',fontsize=14)


k=0
llabel = ascii_lowercase[:12]
for i in range(3):
    for j in range(4):
        ax[i][j].annotate('('+llabel[k]+')',xy=(0.02,0.98),xycoords='axes fraction',size=14,ha='left',va='top',color='k')

        k+=1s
plt.savefig('fig1.pdf',bbox_inches='tight')
