import xarray as xr
import pandas as pd
#from xradd import *
import matplotlib.pyplot as plt
import numpy as np
from string import ascii_lowercase
degree_sign= u'\N{DEGREE SIGN}'
import time

## the horizontal resolutions
res = np.arange(10,110,10)
res = np.append(res,np.array([200,400,600]))
res = np.append(1,res)
nres = np.size(res)

### read GEM water vapor data at 22:00 (600 min)
#ff = xr.open_dataset('GEM_h2o_100hPa_120hPa_001min_600min.nc')
ff=pickle.load( open('GEM_h2o_100hPa_120hPa_001min_600min.pickle', 'rb') )
HU1 = ff.h2o_600min.data
HU0 = ff.h2o_001min.data
wvmean = np.nanmean(HU0,axis=-1)
wvstd = np.nanstd(HU0,axis=-1)
### find locations where the water vapor is significant moistening
idx1 = np.where(HU1-wvmean[:,:,np.newaxis]-3*wvstd[:,:,np.newaxis]>=0)
flagv = np.zeros_like(HU1)
flagv[idx1]=1
### store the GEM water vapor data at locations with significant moistening
h2o1 = HU1[1]*flagv[1]
h2o1[h2o1==0]=np.nan
h2o2 = HU1[0]*flagv[0]
h2o2[h2o2==0]=np.nan
h2o01 = HU0[1]*flagv[1]
h2o02 = HU0[0]*flagv[0]
h2o01[h2o01==0]=np.nan
h2o02[h2o02==0]=np.nan

idxn = np.where(np.isnan(h2o1.flatten()))
h2o11=np.delete(h2o1.flatten(),idxn)
idxn = np.where(np.isnan(h2o2.flatten()))
h2o21=np.delete(h2o2.flatten(),idxn)
idxn = np.where(np.isnan(h2o01.flatten()))
h2o011=np.delete(h2o01.flatten(),idxn)
idxn = np.where(np.isnan(h2o02.flatten()))
h2o021=np.delete(h2o02.flatten(),idxn)

idxf1 = np.where(flagv[1]==1)
npoint1 = np.size(idxf1[1])
idxf2 = np.where(flagv[0]==1)
npoint2 = np.size(idxf2[1])
#ilev01 120, ilev0 100

### now read GEM water vapor data averaged using a range of horizontal resolutions (from 1km to 600 km)
# read pressure level data
#f=xr.open_dataset('avk_data_nospacing_horres1km.nc')
f=pickle.load( open('avk_data_nospacing_horres1km.pickle', 'rb') )
p_inst1 = f.p1.isel(lat=1).isel(lon=1).data
p_inst2 = f.p2.isel(lat=1).isel(lon=1).data
p_inst3=f.p3.isel(lat=1).isel(lon=1).data

# specify pressure levels -> 100 hPa and 120 hPa
ilev1 = np.where(np.logical_and(p_inst1>=98,p_inst1<=101))[0]
ilev2 = np.where(np.logical_and(p_inst2>=98,p_inst2<=101))[0]-1
ilev3 = np.where(np.logical_and(p_inst3>=98,p_inst3<=101))[0]-1
ilev11 = np.where(np.logical_and(p_inst1>=120,p_inst1<=122))[0]
ilev21 = np.where(np.logical_and(p_inst2>=120,p_inst2<=123))[0]
ilev31 = np.where(np.logical_and(p_inst3>=120,p_inst3<=122))[0]

##### read the GEM water vapor prepared using the range of horizontal resolutions (from 1km to 600 km)
#ff=xr.open_dataset('h2o_gem_mls_show_airs_100hPa_120hPa_600min_nkm.nc')
ff=pickle.load( open('h2o_gem_mls_show_airs_100hPa_120hPa_600min_nkm.pickle', 'rb') )
hhmls1=ff.hhmls1.data
hhmls2=ff.hhmls2.data
hhairs1=ff.hhairs1.data
hhairs2=ff.hhairs2.data
hhshow1=ff.hhshow1.data
hhshow2=ff.hhshow2.data

### read GEM-MLS, GEM-SHOW, GEM-AIRS water vapor at significant moistening grid points
#ff = xr.open_dataset('h2o_gem_mls_show_airs_100hPa_120hPa_600min_268E_at_sig_moi.nc')
ff=pickle.load( open('h2o_gem_mls_show_airs_100hPa_120hPa_600min_268E_at_sig_moi.pickle', 'rb') )
hhavkmls1=ff.h2o_gem_mls_100.data
hhavkmls2=ff.h2o_gem_mls_120.data
hhavkshow1=ff.h2o_gem_show_100.data
hhavkshow2=ff.h2o_gem_show_120.data
hhavkairs1=ff.h2o_gem_airs_100.data
hhavkairs2=ff.h2o_gem_airs_120.data

#### generate figure
fig,ax = plt.subplots(nrows=2,ncols=3)
fig.set_size_inches(22,13, forward=True)
title=['GEM-SHOW-nkm','GEM-MLS-nkm','GEM-AIRS-nkm']

xx = np.linspace(0,42,nres+2)
wdt = 2
wdt0= 3

ax[0][1].violinplot(h2o11,positions=[xx[0]],showmeans=True,widths=wdt0,quantiles=[0.05,0.95])
for i in range(nres):
    data = hhmls1[i,:]
    idxn = np.where(np.isnan(data))
    data = np.delete(data,idxn)
    if i<=10:
        vp=ax[0][1].violinplot(data,positions=[xx[i+1]],showmeans=True,widths=wdt,quantiles=[0.05,0.95])
    else:
        vp=ax[0][1].violinplot(data,positions=[xx[i+2]],showmeans=True,widths=wdt,quantiles=[0.05,0.95])
    for vv in vp['bodies']:
        vv.set_color('orange')
    vp['cbars'].set_color('orange')
    vp['cmeans'].set_color('orange')
    vp['cmins'].set_color('orange')
    vp['cmaxes'].set_color('orange')
    vp['cquantiles'].set_color('orange')
ax[0][0].violinplot(h2o11,positions=[xx[0]],showmeans=True,widths=wdt0,quantiles=[0.05,0.95])
for i in range(nres):
    data = hhshow1[i,:]
    idxn = np.where(np.isnan(data))
    data = np.delete(data,idxn)
    if i<=9:

        vp=ax[0][0].violinplot(data,positions=[xx[i+1]],showmeans=True,widths=wdt,quantiles=[0.05,0.95])
    else:
        vp=ax[0][0].violinplot(data,positions=[xx[i+2]],showmeans=True,widths=wdt,quantiles=[0.05,0.95])
    for vv in vp['bodies']:
        vv.set_color('orange')
    vp['cbars'].set_color('orange')
    vp['cmeans'].set_color('orange')
    vp['cmins'].set_color('orange')
    vp['cmaxes'].set_color('orange')
    vp['cquantiles'].set_color('orange')
ax[0][2].violinplot(h2o11,positions=[xx[0]],showmeans=True,widths=wdt0,quantiles=[0.05,0.95])
for i in range(nres):
    data = hhairs1[i,:]
    idxn = np.where(np.isnan(data))
    data = np.delete(data,idxn)
    if i<=1:
        vp=ax[0][2].violinplot(data,positions=[xx[i+1]],showmeans=True,widths=wdt,quantiles=[0.05,0.95])
    else:
        vp=ax[0][2].violinplot(data,positions=[xx[i+2]],showmeans=True,widths=wdt,quantiles=[0.05,0.95])
    for vv in vp['bodies']:
        vv.set_color('orange')
    vp['cbars'].set_color('orange')
    vp['cmeans'].set_color('orange')
    vp['cmins'].set_color('orange')
    vp['cmaxes'].set_color('orange')
    vp['cquantiles'].set_color('orange')


ax[1][1].violinplot(h2o21,positions=[xx[0]],showmeans=True,widths=wdt0,quantiles=[0.05,0.95])
for i in range(nres):
    data = hhmls2[i,:]
    idxn = np.where(np.isnan(data))
    data = np.delete(data,idxn)
    if i<=10:
        vp=ax[1][1].violinplot(data,positions=[xx[i+1]],showmeans=True,widths=wdt,quantiles=[0.05,0.95])
    else:
        vp=ax[1][1].violinplot(data,positions=[xx[i+2]],showmeans=True,widths=wdt,quantiles=[0.05,0.95])
    for vv in vp['bodies']:
        vv.set_color('orange')
    vp['cbars'].set_color('orange')
    vp['cmeans'].set_color('orange')
    vp['cmins'].set_color('orange')
    vp['cmaxes'].set_color('orange')
    vp['cquantiles'].set_color('orange')
ax[1][0].violinplot(h2o21,positions=[xx[0]],showmeans=True,widths=wdt0,quantiles=[0.05,0.95])
for i in range(nres):
    data = hhshow2[i,:]
    idxn = np.where(np.isnan(data))
    data = np.delete(data,idxn)
    if i<=9:
        vp=ax[1][0].violinplot(data,positions=[xx[i+1]],showmeans=True,widths=wdt,quantiles=[0.05,0.95])
    else:
        vp=ax[1][0].violinplot(data,positions=[xx[i+2]],showmeans=True,widths=wdt,quantiles=[0.05,0.95])
    for vv in vp['bodies']:
        vv.set_color('orange')
    vp['cbars'].set_color('orange')
    vp['cmeans'].set_color('orange')
    vp['cmins'].set_color('orange')
    vp['cmaxes'].set_color('orange')
    vp['cquantiles'].set_color('orange')
ax[1][2].violinplot(h2o21,positions=[xx[0]],showmeans=True,widths=wdt0,quantiles=[0.05,0.95])
for i in range(nres):
    data = hhairs2[i,:]
    idxn = np.where(np.isnan(data))
    data = np.delete(data,idxn)
    if i<=1:
        vp=ax[1][2].violinplot(data,positions=[xx[i+1]],showmeans=True,widths=wdt,quantiles=[0.05,0.95])
    else:
        vp=ax[1][2].violinplot(data,positions=[xx[i+2]],showmeans=True,widths=wdt,quantiles=[0.05,0.95])
    for vv in vp['bodies']:
        vv.set_color('orange')
    vp['cbars'].set_color('orange')
    vp['cmeans'].set_color('orange')
    vp['cmins'].set_color('orange')
    vp['cmaxes'].set_color('orange')
    vp['cquantiles'].set_color('orange')

ax[0][1].violinplot(hhavkmls1,positions=[xx[12]],showmeans=True,widths=wdt,quantiles=[0.05,0.95])
ax[1][1].violinplot(hhavkmls2,positions=[xx[12]],showmeans=True,widths=wdt,quantiles=[0.05,0.95])
ax[0][0].violinplot(hhavkshow1,positions=[xx[11]],showmeans=True,widths=wdt,quantiles=[0.05,0.95])
ax[1][0].violinplot(hhavkshow2,positions=[xx[11]],showmeans=True,widths=wdt,quantiles=[0.05,0.95])
ax[0][2].violinplot(hhavkairs1,positions=[xx[3]],showmeans=True,widths=wdt,quantiles=[0.05,0.95])
ax[1][2].violinplot(hhavkairs2,positions=[xx[3]],showmeans=True,widths=wdt,quantiles=[0.05,0.95])


ax[1][1].plot(0,0,linestyle=':',color='tab:blue',linewidth=2,label='GEM water vapor statistics')
ax[1][1].plot(0,0,linestyle='--',color='k',linewidth=2,label='GEM $\overline{H2O_0}+STD_0$')
ax[1][1].plot(0,0,linestyle='-',color='k',linewidth=2,label='GEM $\overline{H2O_0}$')
ax[1][1].legend(fontsize=18,bbox_to_anchor=(0.5,-0.2),loc='upper center',ncol=3,)

xticklabs = np.array([str(ii)+'km' for ii in res[:10]])
xticklabs = np.append('GEM',xticklabs)
xticklabs = np.append(xticklabs,'GEM-SHOW')
xticklabs = np.append(xticklabs,np.array([str(ii)+'km' for ii in res[10:]]))

xticklabm = np.array([str(ii)+'km' for ii in res[:11]])
xticklabm = np.append('GEM',xticklabm)
xticklabm = np.append(xticklabm,'GEM-MLS')
xticklabm = np.append(xticklabm,np.array([str(ii)+'km' for ii in res[11:]]))

xticklaba = np.array([str(ii)+'km' for ii in res[:2]])
xticklaba = np.append('GEM',xticklaba)
xticklaba = np.append(xticklaba,'GEM-AIRS')
xticklaba = np.append(xticklaba,np.array([str(ii)+'km' for ii in res[2:]]))
llabel = ascii_lowercase[:6]
k=0
for i in range(2):
    for j in range(3):
        ax[0][j].set_ylim([3,28])
        ax[1][j].set_ylim([0,70])
        ax[0][j].plot(xx,np.ones_like(xx)*np.nanmean(h2o11),linestyle=':',color='tab:blue')
        ax[0][j].plot(xx,np.ones_like(xx)*np.percentile(h2o11,5),linestyle=':',color='tab:blue')
        ax[0][j].plot(xx,np.ones_like(xx)*np.percentile(h2o11,95),linestyle=':',color='tab:blue')
        ax[1][j].plot(xx,np.ones_like(xx)*np.nanmean(h2o21),linestyle=':',color='tab:blue')
        ax[1][j].plot(xx,np.ones_like(xx)*np.percentile(h2o21,5),linestyle=':',color='tab:blue')
        ax[1][j].plot(xx,np.ones_like(xx)*np.percentile(h2o21,95),linestyle=':',color='tab:blue')

        ax[0][j].plot(xx,np.ones_like(xx)*np.nanmean(h2o011),color='k')
        ax[1][j].plot(xx,np.ones_like(xx)*np.nanmean(h2o021),color='k')
        ax[0][j].plot(xx,np.ones_like(xx)*(np.nanmean(h2o011)+3*np.nanstd(h2o011)),linestyle='--',color='k')
        ax[1][j].plot(xx,np.ones_like(xx)*(np.nanmean(h2o021)+3*np.nanstd(h2o021)),linestyle='--',color='k')

        ax[i][j].set_ylabel('ppmv',fontsize=12)
        ax[i][j].set_xticks(xx)
        ax[i][0].set_xticklabels(xticklabs,rotation=90)
        ax[i][1].set_xticklabels(xticklabm,rotation=90)
        ax[i][2].set_xticklabels(xticklaba,rotation=90)
        ax[0][j].set_title(title[j])
        ax[0][j].annotate('100 hPa',xy=(0.5,0.97),xycoords='axes fraction',size=12,ha='center',va='top')
        ax[1][j].annotate('120 hPa',xy=(0.5,0.97),xycoords='axes fraction',size=12,ha='center',va='top')



        ax[i][j].annotate('('+llabel[k]+')',xy=(0.99,0.97),xycoords='axes fraction',size=14,ha='right',va='top')
        k+=1

plt.savefig('fig8.pdf',bbox_inches='tight')
