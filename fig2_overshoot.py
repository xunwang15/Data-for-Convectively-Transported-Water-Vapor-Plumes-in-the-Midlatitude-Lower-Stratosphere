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
itime = np.where(datatime==pd.Timestamp(2013, 8, 25, 18, 50))[0][0]

###### read water vapor and variables at 18:50, 48.6 N. Read tracks of overshoot
# f2 = xr.open_dataset('h2o_iwc_theta_trop_tke_48p6N_8251850.nc')
# f3 = xr.open_dataset('h2o_zonal_mean_h2o_std.nc')

f2 = pickle.load( open('h2o_iwc_theta_trop_tke_48p6N_8251850.pickle', 'rb') )
f3 = pickle.load( open('h2o_zonal_mean_h2o_std.pickle', 'rb') )

lon = f2.lon.data
lat = f2.lat.data
alt = f3.alt.data
nlat = np.size(lat)
nlon = np.size(lon)
nalt= np.size(alt)

wv = f2.h2o.data
iwc = f2.iwc.data
theta=f2.theta.data
tke = f2.tke.data
tt = f2.t.data
trop=f2.trop.data
uu=f2.u.data
ww=f2.w.data

## compute dtheta/dz
delz = (alt[1:]-alt[:-1])*1000.
dthdz = (theta[1:,:]-theta[:-1,:])/delz[:,np.newaxis]
dthdz = np.append(dthdz,dthdz[-1,:][np.newaxis,:],axis=0)

uu2 = uu[:,::5]
ww2 = ww[:,::5]

wvmean = f3.wvmean.data
wvstd = f3.wvstd.data

#ff = xr.open_dataset('selected_overshootingtop.nc')
ff = pickle.load( open('selected_overshootingtop.pickle', 'rb') )
latc = ff.latc.data
lonc = ff.lonc.data
flagt = ff.flagtime.data
du = ff.duration.data
labrecord = ff.label.data
nlab = np.size(labrecord)
ovtime = np.zeros_like(labrecord,dtype='int')
for i in range(nlab):
    idx = np.where(flagt[i,:]>0)[0]
    ovtime[i]=idx[0]+280

####### generate Figure 2
ilat = 289
fig,ax = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(18,7, forward=True)

ima=ax[1].contourf(lon,alt,wv-wvmean[:,ilat,np.newaxis],levels=np.arange(-40,41,1),cmap='bwr',extend='both')
ax[1].plot(lon,trop,color='k',linewidth=2)
ax[1].contour(lon,alt,theta,levels=np.arange(350,450,5),colors='grey',linewidths=0.5)

ax[1].contour(lon,alt,iwc,levels=[1e-3,1],colors='cyan',linewidths=3)
ax[1].contour(lon,alt,dthdz,levels=[0.01],colors='b',linewidths=2)
ax[1].contour(lon,alt,tke,levels=[20],linewidths=2,colors='k',linestyles='--')
ax[1].quiver(lon[::5],alt,uu2,ww2,width=0.002,headwidth=6,headlength=8,scale=900)

levels1 = [380]
fmt = {}
strs = np.array([str(i) for i in levels1])
for l,s in zip(levels1,strs):
    fmt[l] = s
cs=ax[1].contour(lon,alt,tt-ttmean[:,ilat,np.newaxis],levels=np.arange(-15,0,5),colors='magenta',linewidths=3,linestyle='--')
cs=ax[1].contour(lon,alt,theta,levels=[380],colors='grey',linewidths=2)
ax[1].clabel(cs,cs.levels,inline=True,fmt=fmt,fontsize=12)

ax[1].plot([0,0],color='k',linewidth=3,label='tropopause')
ax[1].plot([0,0],color='grey',linewidth=0.8,label='potential temperature')
ax[1].plot([0,0],color='cyan',linewidth=3,label='IWC')
ax[1].plot([0,0],color='b',label='$d\Theta$/dz<=0.01 K/m')
ax[1].plot([0,0],color='k',linestyle='--',linewidth=3,label='TKE>=10 $m^2$ $s^{-2}$')
ax[1].plot([0,0],color='magenta',linestyle='--',linewidth=3,label='temperature anomaly')
ax[1].set_ylim([13,18])
ax[1].set_xlim([264.5,265.4])
ax[1].set_ylabel('Altitude (km)',fontsize=16)
ax[1].set_xlabel('Longitude',fontsize=16)
ax[1].legend(loc='upper center',bbox_to_anchor=(0.5,-0.1),fontsize=12,ncol=3)
ax[1].set_title(datatime[itime],fontsize=15)

cbaxes = fig.add_axes([0.98, 0.22, 0.01, 0.7])
cc = fig.colorbar(ima, ax=ax[1],cax=cbaxes)
cc.set_ticks(np.arange(-40,45,5))
clabels = np.array([str(round(i)) for i in np.arange(-40,45,5)])
cc.set_ticklabels(clabels)
cc.ax.tick_params(labelsize=10)
cc.set_label('ppmv',fontsize=14)


cls = plt.cm.jet(np.linspace(0,1,720))
ticklab = [str(i)+'hr' for i in np.arange(12,25,1)]
for ii in range(len(labrecord)):
    itt = np.where(flagt[ii,:]==1)[0]
    difflat = latc[ii,itt][1:]-latc[ii,itt][:-1]
    ibad = np.where(abs(difflat)>=0.2)[0]
    latc[ii,itt[ibad]]=np.nan
    ax[0].plot(lonc[ii,itt],latc[ii,itt],color=cls[ovtime[ii]-1])

ii=13
itt = np.where(flagt[ii,:]==1)[0]
ax[0].plot(lonc[ii,itt[0]],latc[ii,itt[0]],marker='x',color='k')
ax[0].plot(lonc[ii,itt[-1]],latc[ii,itt[-1]],marker='x',color='k')

jet = cm.get_cmap('jet', 720)
ax1 = fig.add_axes([0.47,0.22,0.01,0.7])  # setup colorbar
cb = mpl.colorbar.ColorbarBase(ax1, orientation='vertical', \
                               cmap=jet,norm=mpl.colors.Normalize(0, 720,1),\
                               ticks=np.arange(0,730,60),\
                               label='Time')
cb.ax.set_yticklabels(ticklab)

ax[0].set_xlabel('Longitude',fontsize=15)
ax[0].set_ylabel('Latitude',fontsize=15)

lat1,lat2 = 46,50.1
lon1,lon2 = 360-100,360-87.5+0.1
ax[0].set_xlim([lon1,lon2])
ax[0].set_ylim([lat1,lat2])

ax[0].set_title('Overshoot tracks',fontsize=16)
plt.tight_layout(w_pad=7)

llabel = ascii_lowercase[:12]
ax[0].annotate('('+llabel[0]+')',xy=(0.02,0.98),xycoords='axes fraction',size=18,ha='left',va='top',color='k')
ax[1].annotate('('+llabel[1]+')',xy=(0.02,0.98),xycoords='axes fraction',size=18,ha='left',va='top',color='k')

plt.savefig('fig2.pdf',bbox_inches='tight')
