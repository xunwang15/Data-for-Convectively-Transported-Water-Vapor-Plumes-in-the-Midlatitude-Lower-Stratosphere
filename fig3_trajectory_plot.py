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

### read GEM wataer vapor data
#f3 = xr.open_dataset('h2o_zonal_mean_h2o_std.nc')
f3 = pickle.load( open('h2o_zonal_mean_h2o_std.pickle', 'rb') )
wvmean = f3.wvmean.data
wvstd = f3.wvstd.data

#ff=xr.open_dataset('h2o_gem_data_for_traj_plot.nc')
ff=pickle.load( open('h2o_gem_data_for_traj_plot.pickle', 'rb') )
h2o = ff.h2o.data
h2o_hor = ff.h2o_hor.data
h2o_profile_lms=ff.h2o_profile_lms.data
h2o_profile_ovw=ff.h2o_profile_ovw.data
iwc= ff.iwc.data
trop = ff.trop.data
theta=ff.theta.data
alt=ff.alt.data
alt3=ff.alt3.data
lat=ff.lat.data
lon=ff.lon.data
wvmean = ff.h2omean.data
wvstd = ff.h2ostd.data
### read trajectory data
#ff=xr.open_dataset('trajectory_data.nc')
ff=pickle.load( open('trajectory_data.pickle', 'rb') )
ftraj_lon_lms=ff.ftraj_lon_lms.data+360.
ftraj_alt_lms=ff.ftraj_alt_lms.data
ftraj_lon_ovw=ff.ftraj_lon_ovw.data+360.
ftraj_alt_ovw=ff.ftraj_alt_ovw.data
ftraj_lon_ovs=ff.ftraj_lon_ovs.data+360.
ftraj_alt_ovs=ff.ftraj_alt_ovs.data

ftraj_h2o_ovs=ff.ftraj_h2o_ovs.data
ftraj_h2o_ovw=ff.ftraj_h2o_ovw.data
ftraj_h2o_lms=ff.ftraj_h2o_lms.data
ftraj_theta_ovs=ff.ftraj_theta_ovs.data
ftraj_theta_ovw=ff.ftraj_theta_ovw.data
ftraj_theta_lms=ff.ftraj_theta_lms.data

hdif_lms=ff.hdif_lms.data
hdif_ovw=ff.hdif_ovw.data
hdif_ovs=ff.hdif_ovs.data

btraj_lon_lms=ff.btraj_lon_lms.data+360.
btraj_alt_lms=ff.btraj_alt_lms.data
btraj_lat_lms=ff.btraj_lat_lms.data
btraj_lon_ovw=ff.btraj_lon_ovw.data+360.
btraj_alt_ovw=ff.btraj_alt_ovw.data
btraj_lat_ovw=ff.btraj_lat_ovw.data
btraj_lon_ovs=ff.btraj_lon_ovs.data+360.
btraj_alt_ovs=ff.btraj_alt_ovs.data
btraj_lat_ovs=ff.btraj_lat_ovs.data

lat_ovw=ff.lat_ovw.data
lon_ovw=ff.lon_ovw.data
lat_lms=ff.lat_lms.data
lon_lms=ff.lon_lms.data
nlms=ff.nlms.data
novw=ff.novw.data
novs=ff.novs.data
### generate figure
nlms=np.array([int(i) for i in nlms])
novw=np.array([int(i) for i in novw])
novs=np.array([int(i) for i in novs])


lat1,lat2 = 46,50.1
lon1,lon2 = 360-100,360-87.5+0.1
fig,ax = plt.subplots(nrows=3,ncols=4)
fig.set_size_inches(20,13, forward=True)

################## ##################
################## second row ##################
ilat = np.where(lat>=48.6)[0][0]
levels = np.arange(0,15.5,0.5)

ccmap = 'binary'
for i,itt in zip(range(4),range(0,80,20)):

    ima = ax[1][i].contourf(lon,alt,h2o[i,:]-wvmean[:,ilat,np.newaxis],levels=levels,cmap='binary',extend='both')
    ax[1][i].contour(lon,alt,h2o[i,:]-wvmean[:,ilat,np.newaxis],levels=[1],colors='k',linewidths=2)
    ax[1][i].contour(lon,alt,iwc[i,:],levels=[1e-3],linewidths=1.5,colors='cyan')

    ax[1][i].scatter(ftraj_lon_lms[:,itt],ftraj_alt_lms[:,itt],color='limegreen',s=8)
    ax[1][i].scatter(ftraj_lon_ovw[:,itt],ftraj_alt_ovw[:,itt],color='orange',s=8)
    ax[1][i].scatter(ftraj_lon_ovs[:,itt],ftraj_alt_ovs[:,itt],color='b',s=5)

    ax[1][i].plot(lon,trop[i,:],color='magenta',linewidth=2)
    ax[1][i].contour(lon,alt,theta[i,:],levels=[380],linewidths=3,colors='magenta',linestyles=':')
    ax[1][i].set_ylim([12,18])
    ax[1][i].set_xlim([263,268])

ax[1][0].set_ylim([13.5,17.5])
ax[1][0].set_xlim([264.6,265.8])

#cbaxes = fig.add_axes([0.99,0.06,0.01,0.24])
cbaxes = fig.add_axes([0.99,0.4,0.01,0.23]) # setup colorbar
cc = fig.colorbar(ima, ax=ax[2][3],cax=cbaxes)
cc.set_ticks(levels[::2])
clabels = np.array([str(round(i,2)) for i in levels[::2]])
cc.set_ticklabels(clabels)
cc.ax.tick_params(labelsize=8)
cc.set_label('ppmv',fontsize=14)

ax[1][0].set_title(datatime[itime],fontsize=11)
ax[1][1].set_title(datatime[itime+20],fontsize=11)
ax[1][2].set_title(datatime[itime+40],fontsize=11)
ax[1][3].set_title(datatime[itime+60])

################## ##################
################## first row ##################

ilat = np.where(lat>=48.6)[0][0]

for i in range(3):
    ima = ax[0][i].contourf(lon,alt,h2o[0,:]-wvmean[:,ilat,np.newaxis],levels=levels,cmap='binary',extend='both')
    ax[0][i].contour(lon,alt,theta[0,:],levels=np.arange(360,450,10),linewidths=0.5,colors='k')

for ii in nlms[::20]:
    ax[0][2].plot(btraj_lon_lms[ii,:],btraj_alt_lms[ii,:] ,color='limegreen',linewidth=0.8)
for ii in novw[::10]:
    ax[0][1].plot(btraj_lon_ovw[ii,:],btraj_alt_ovw[ii,:] ,color='orange',linewidth=0.8)
for ii in novs[::5]:
    ax[0][0].plot(btraj_lon_ovs[ii,:],btraj_alt_ovs[ii,:] ,color='b',linewidth=0.8)

for i in range(3):
    ax[0][i].plot(lon,trop[0,:],linewidth=2,color='magenta')
    ax[0][i].contour(lon,alt,theta[0,:],levels=[380],linewidths=3,colors='magenta',linestyles=':')
    ax[0][i].contour(lon,alt,h2o[0,:]-wvmean[:,ilat,np.newaxis],levels=[1],colors='k',linewidths=2)
    ax[0][i].contour(lon,alt,iwc[0,:],levels=[1e-3],linewidths=1.5,colors='cyan')


cbaxes = fig.add_axes([0.75,0.73,0.01,0.24]) # setup colorbar
cc = fig.colorbar(ima, ax=ax[0][2],cax=cbaxes)
cc.set_ticks(levels[::2])
clabels = np.array([str(round(i,2)) for i in levels[::2]])
cc.set_ticklabels(clabels)
cc.ax.tick_params(labelsize=8)
cc.set_label('ppmv',fontsize=14)

ax[0][0].set_ylim([12,18])
ax[0][0].set_xlim([263.5,266])
ax[0][1].set_ylim([12,18])
ax[0][1].set_xlim([263.5,266])
ax[0][2].set_ylim([12,18])
ax[0][2].set_xlim([263.5,266])

ax[0][3].remove()
ax2 = fig.add_subplot(3, 4, 4, projection='3d')

for ii in nlms[::10]:
    ax2.scatter3D(btraj_lon_lms[ii,:120],btraj_lat_lms[ii,:120],btraj_alt_lms[ii,:120] ,color='limegreen',s=1)
for ii in novs[::5]:
    ax2.scatter3D(btraj_lon_ovs[ii,:120],btraj_lat_ovs[ii,:120],btraj_alt_ovs[ii,:120] ,color='b',s=1)
for ii in novw[::5]:
    ax2.scatter3D(btraj_lon_ovw[ii,:120],btraj_lat_ovw[ii,:120],btraj_alt_ovw[ii,:120] ,color='orange',s=1)

ax2.set_zlim([10,20])
ax2.set_zlabel('km',fontsize=11)
ax2.annotate('longitude',xy=(0.5,-0.01),xycoords='axes fraction',size=11,ha='center',va='top')
ax2.set_ylabel('latitude',fontsize=11)
ax2.azim = -80
ax2.dist = 7
ax2.elev = 15

ax[0][0].set_ylabel('km',fontsize=10)
ax[0][1].set_ylabel('km',fontsize=10)
ax[0][3].set_ylabel('km',fontsize=10)


for j in range(3):
    ax[0][j].set_xlabel('Longitude',fontsize=10)


ax[1][3].set_ylabel('km',fontsize=10)
for i in range(1,3):
    for j in range(3):
        ax[i][j].set_ylabel('km',fontsize=10)

ax[1][3].set_ylabel('Longitude',fontsize=10)
for i in range(3):
    for j in range(3):
        ax[i][j].set_xlabel('Longitude',fontsize=10)


plt.tight_layout(w_pad=1)


######## third row ##########


ax[2][1].plot(np.nanmean(wvmean[15:,:],axis=1),alt[15:],color='k',label='$\overline{H2O_0}$')
ax[2][1].plot(np.nanmean(wvmean[15:,:],axis=1)+np.nanmean(wvstd[15:,:],axis=1),alt[15:],color='grey',label='$\overline{H2O_0}+STD_0$')
ax[2][1].plot(np.nanmean(wvmean[15:,:],axis=1)+3*np.nanmean(wvstd[15:,:],axis=1),alt[15:],color='grey',linestyle='--',label='$\overline{H2O_0}+3STD_0$')
ax[2][1].scatter(0,0,color='lime',label='LMS parcel')
ax[2][1].scatter(0,0,color='orange',label='OVW parcel')
ax[2][1].scatter(0,0,color='b',label='OVW parcel')
ax[2][1].legend()

#ax[2][1].scatter(h2o_traj22[:,-1],alt_traj22[:,-1] ,s=3,color='lightgrey')
ax[2][1].scatter(ftraj_h2o_lms[:,-1],ftraj_alt_lms[:,-1],s=3,color='lime')
ax[2][1].scatter(ftraj_h2o_ovw[:,-1],ftraj_alt_ovw[:,-1],s=3,color='orange')
ax[2][1].scatter(ftraj_h2o_ovs[:,-1],ftraj_alt_ovs[:,-1],s=5,color='b')
ax[2][1].set_xlim([0,80])
ax[2][1].set_ylim([12,20])


#ax[2][0].scatter(diff1,th_traj22[:,-1],s=1,color='lightgrey')
ax[2][0].scatter(hdif_ovw,ftraj_theta_ovw[:,-1],s=3,color='C01')
ax[2][0].scatter(hdif_lms,ftraj_theta_lms[:,-1],s=3,color='lime')

xx = hdif_ovs*1.
yy = ftraj_theta_ovs[:,-1]*1.
ax[2][0].scatter(xx,yy,s=5,color='b')


ax[2][0].plot([0,0],[350,410],color='k')
ax[2][0].scatter(0,0,color='lime',label='LMS parcel')
ax[2][0].scatter(0,0,color='orange',label='OVW parcel')
ax[2][0].scatter(0,0,color='b',label='overshoot parcel')
ax[2][0].legend()

ax[2][0].set_xlim([-10,70])
ax[2][0].set_ylim([345,415])
ax[2][0].set_xticks(np.arange(-10,80,10))

ax[2][2].plot(np.nanmean(wvmean[15:,:],axis=1),alt[15:],color='k',label='$\overline{H2O_0}$')
ax[2][2].plot(np.nanmean(wvmean[15:,:],axis=1)+np.nanmean(wvstd[15:,:],axis=1),alt[15:],color='grey',label='$\overline{H2O_0}+STD_0$')
ax[2][2].plot(np.nanmean(wvmean[15:,:],axis=1)+3*np.nanmean(wvstd[15:,:],axis=1),alt[15:],color='grey',linestyle='--',label='$\overline{H2O_0}+3STD_0$')

ax[2][2].plot(h2o_profile_lms,alt3,color='g',label='LMS profile')
ax[2][2].plot(h2o_profile_ovw,alt3,color='orange',label='OVW profile')
ax[2][2].set_xlim([0,80])
ax[2][2].set_ylim([12,20])
ax[2][2].legend()




ax[2][0].set_title(datatime[-1])
ax[2][1].set_title(datatime[-1])
ax[2][2].set_title(datatime[-1])
ax[2][0].set_xlabel('water vapor anomaly (ppmv)',fontsize=13)
ax[2][1].set_xlabel('water vapor mixing ratio (ppmv)',fontsize=13)

ax[2][1].set_ylabel('altitude (km)',fontsize=13)
ax[2][0].set_ylabel('potential temperature (K)',fontsize=13)

ax[2][2].set_xlabel('water vapor mixing ratio (ppmv)',fontsize=13)
ax[2][2].set_ylabel('altitude (km)',fontsize=13)

ax[2][3].contour(lon,lat,h2o_hor[0,:]-wvmean[29,:,np.newaxis],levels=[10],colors='magenta',linewidths=1)
ax[2][3].contour(lon,lat,h2o_hor[1,:]-wvmean[33,:,np.newaxis],levels=[2],colors='b',linewidths=1)
ax[2][3].scatter(lon_lms+360,lat_lms,s=3,color='lime')
ax[2][3].scatter(lon_ovw+360,lat_ovw,s=3,color='orange')
ax[2][3].plot(lon[0],lat[0],color='b',label='16.5 km moistening')
ax[2][3].plot(lon[0],lat[0],color='magenta',label='15.5 km moistening')
ax[2][3].scatter(0,0,color='lime',label='LMS parcel')
ax[2][3].scatter(0,0,color='orange',label='OVW parcel')
ax[2][3].set_ylabel('latitude',fontsize=11)
ax[2][3].set_xlabel('longitude',fontsize=11)
ax[2][3].legend()
ax[2][3].set_title(datatime[-1])
ax[2][3].set_xlim([lon1,lon2])
ax[2][3].set_ylim([lat1,lat2])

k=0
llabel = ascii_lowercase[:12]
for i in range(2):
    for j in range(4):
        ax[i][j].annotate('('+llabel[k]+')',xy=(0.02,0.98),xycoords='axes fraction',size=14,ha='left',va='top',color='k')
        k+=1

for j in range(4):
    ax[2][j].annotate('('+llabel[k]+')',xy=(0.98,0.02),xycoords='axes fraction',size=14,ha='right',va='bottom',color='k')
    k+=1
ax2.annotate('(d)',xy=(0.02,0.98),xycoords='axes fraction',size=14,ha='left',va='top',color='k')

for i in range(3):
    ax[0][i].set_title(str(datatime[itime])+' 120 min back trajectory',fontsize=12)
