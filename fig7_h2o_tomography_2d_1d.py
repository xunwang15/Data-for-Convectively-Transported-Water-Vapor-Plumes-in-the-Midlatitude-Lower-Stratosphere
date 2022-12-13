import xarray as xr
import pandas as pd
#from xradd import *
import matplotlib.pyplot as plt
import numpy as np
from string import ascii_lowercase
degree_sign= u'\N{DEGREE SIGN}'
import time

### define colormap
cf = open('colormapbgyr.txt','r')
data_temp = cf.readlines()
data_temp = ''.join(data_temp).replace('\n',' ')
data_temp = " ".join(data_temp.split())
data_temp = data_temp.split(' ')
data = np.array([int(i) for i in data_temp])
data = np.reshape(data,(int(np.size(data)/3),3))
import matplotlib as mpl # in python
cm_bgyr = mpl.colors.ListedColormap(data/255.0)
#plt.imshow(..., cmap=cm) # for example
del data

### read data
#ff = xr.open_dataset('h2o_gem_gem-mls_gem-show_gem-airs_2d_1d_600min_268E.nc')
ff=pickle.load( open('h2o_gem_gem-mls_gem-show_gem-airs_2d_1d_600min_268E.pickle', 'rb') )
HU1=ff.h2o_gem.data
HU0=ff.h2o_gem0.data
HUstd=ff.h2o_gem_std.data
lat = ff.lat.data
pp = ff.pp.data
troppp=ff.tropp.data
cloudtop_gem=ff.cloudtop_gem.data
h2o_gem_mls=ff.h2o_gem_mls.data
h2o_gem_show=ff.h2o_gem_show.data
h2o_gem_airs=ff.h2o_gem_airs.data
pshow=ff.pshow.data
pairs=ff.pairs.data
pmls=ff.pmls.data
latshow=ff.latshow.data
latairs=ff.latairs.data
latmls=ff.latmls.data
hshow1d=ff.h2o_gem_show_1d.data
hmls1d=ff.h2o_gem_mls_1d.data
hairs1d=ff.h2o_gem_airs_1d.data


##### Generate figure
fig,ax = plt.subplots(nrows=3,ncols=2,gridspec_kw={'height_ratios': [1, 1, 1.8]})
fig.set_size_inches(13,17, forward=True)
title=['GEM','GEM-MLS','GEM-SHOW','GEM-AIRS']
### plot the tomography of GEM and GEM-MLS water vapor
vmax=40
levels = np.arange(0,vmax+5,5)
ax[0][0].pcolormesh(lat,pp,HU1,vmin=0,vmax=vmax,cmap=cm_bgyr)
image=ax[0][1].pcolormesh(latmls,pmls,h2o_gem_mls,vmax=vmax,cmap=cm_bgyr)

cbaxes = fig.add_axes([0.92,0.72,0.01,0.15])                             # setup colorbar
cc = fig.colorbar(image, ax=ax[0][1],cax=cbaxes)
cc.set_ticks(levels)
clabels = np.array([str(i) for i in levels])
cc.set_ticklabels(clabels)
cc.ax.tick_params(labelsize=13)
cc.set_label('ppmv',fontsize=14)

### plot the tomography of gem-airs and gem-show water vapor
image=ax[1][1].pcolormesh(latairs,pairs,h2o_gem_airs,vmin=0,vmax=vmax,cmap=cm_bgyr)
ax[1][0].pcolormesh(latshow,pshow,h2o_gem_show,vmin=0,vmax=vmax,cmap=cm_bgyr)

cbaxes = fig.add_axes([0.92,0.5,0.01,0.15])                             # setup colorbar
cc = fig.colorbar(image, ax=ax[1][1],cax=cbaxes)
cc.set_ticks(levels)
clabels = np.array([str(i) for i in levels])
cc.set_ticklabels(clabels)
cc.ax.tick_params(labelsize=13)
cc.set_label('ppmv',fontsize=14)

k=0
for i in range(2):
    for j in range(2):
        ax[i][j].plot(lat,cloudtop_gem,linewidth=0.8,color='cyan')
        ax[i][j].plot(lat,troppp,linewidth=1,color='k')
        ax[i][j].set_ylim([170,40])
        ax[i][j].set_ylabel('hPa',fontsize=15)
        ax[1][j].set_xlabel('Latitude',fontsize=12)
        ax[i][j].set_title(title[k],fontsize=14)
        k+=1

lat11 = 47.9
i=np.where(lat>=lat11)[0][0]
ax[2][0].plot(HU0[:,i],pp,linewidth=2,color='k',label='GEM $\overline{H2O_0}$')
ax[2][0].plot(HU0[:,i]+HUstd[:,i],pp,linewidth=2,color='grey',linestyle='--',label='GEM $\overline{H2O_0}+STD_0$')
ax[2][0].plot([0,60],[np.nanmean(troppp),np.nanmean(troppp)],color='k',linestyle='-.',label='tropopause')

ax[2][0].plot(HU1[:,i],pp,linewidth=3,color='C0',label='GEM')
ax[2][0].plot(hmls1d[1,:,0],pmls,linewidth=2,color='purple',linestyle='--')
ax[2][0].plot(hmls1d[0,:,0],pmls,linewidth=2,color='purple',label='GEM-MLS')
ax[2][0].plot(hairs1d[1,:,0],pairs,linewidth=2,color='r',linestyle='--')
ax[2][0].plot(hairs1d[0,:,0],pairs,linewidth=2,color='r',label='GEM-AIRS')
ax[2][0].plot(hshow1d[1,:,0],pshow,linewidth=2,color='orange',linestyle='--')
ax[2][0].plot(hshow1d[0,:,0],pshow,linewidth=2,color='orange',label='GEM-SHOW')

ax[2][0].set_ylim([160,40])
ax[2][0].set_xlim([0,60])
ax[2][0].grid()
ax[2][0].set_xlabel('Water vapor (ppmv)',fontsize=15)
ax[2][0].set_ylabel('hPa',fontsize=15)


lat22 = 47
i=np.where(lat>=lat22)[0][0]
ax[2][1].plot(HU0[:,i],pp,linewidth=2,color='k',label='GEM $\overline{H2O_0}$')
ax[2][1].plot(HU0[:,i]+HUstd[:,i],pp,linewidth=2,color='grey',linestyle='--',label='GEM $\overline{H2O_0}+STD_0$')
ax[2][1].plot([0,60],[np.nanmean(troppp),np.nanmean(troppp)],color='k',linestyle='-.',label='tropopause')
ax[2][1].plot(HU1[:,i],pp,linewidth=3,color='C0',label='GEM')

ax[2][1].plot(hmls1d[1,:,1],pmls,linewidth=2,color='purple',linestyle='--')
ax[2][1].plot(hmls1d[0,:,1],pmls,linewidth=2,color='purple',label='GEM-MLS')
ax[2][1].plot(hairs1d[1,:,1],pairs,linewidth=2,color='r',linestyle='--')
ax[2][1].plot(hairs1d[0,:,1],pairs,linewidth=2,color='r',label='GEM-AIRS')
ax[2][1].plot(hshow1d[1,:,1],pshow,linewidth=2,color='orange',linestyle='--')
ax[2][1].plot(hshow1d[0,:,1],pshow,linewidth=2,color='orange',label='GEM-SHOW')


ax[2][1].set_ylim([160,40])
ax[2][1].set_xlim([0,60])
ax[2][1].grid()
ax[2][1].set_xlabel('Water vapor (ppmv)',fontsize=15)
ax[2][1].set_ylabel('hPa',fontsize=15)
ax[2][1].legend(fontsize=12,bbox_to_anchor=(0.99,0.6),loc='center right')
ax[2][0].legend(fontsize=12,bbox_to_anchor=(0.99,0.6),loc='center right')
ax[2][0].set_title('LMS (48'+degree_sign+'N 268'+degree_sign+'E)',fontsize=16)
ax[2][1].set_title('overworld (47'+degree_sign+'N 268'+degree_sign+'E)',fontsize=16)

ax[1][0].set_xticks(np.arange(43,52,1))
ax[1][1].set_xticks(np.arange(43,52,1))
ax[1][0].set_xlim([lat[0],lat[-1]])
ax[1][1].set_xlim([lat[0],lat[-1]])
k=0
xy = [0.99,0.99,0.97,0.99,0.99,0.99,0.99]
colors = ['white','k','white','white','k','k']
llabel = ascii_lowercase[:6]
for i in range(3):
    for j in range(2):
        ax[i][j].annotate('('+llabel[k]+')',xy=(xy[k],0.97),xycoords='axes fraction',size=14,ha='right',va='top',color=colors[k])

        k+=1

plt.savefig('fig7.pdf',bbox_inches='tight')
