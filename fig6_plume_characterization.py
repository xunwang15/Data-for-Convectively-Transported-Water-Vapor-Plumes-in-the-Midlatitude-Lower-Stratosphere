import xarray as xr
import pandas as pd
#from xradd import *
import matplotlib.pyplot as plt
import numpy as np
from string import ascii_lowercase
degree_sign= u'\N{DEGREE SIGN}'
import time

#### read moistening segment variable ndf
#ff=xr.open_dataset('moistening_segment_vertsize_wv_theta_ndf.nc')
ff=pickle.load( open('moistening_segment_vertsize_wv_theta_ndf.pickle', 'rb') )
verts = ff.vertical_size.data
bin1 = ff.bin1.data
top_alt = ff.top_alt.data
bin2=ff.bin2.data
bot_alt=ff.bot_alt.data
h2o_mean=ff.h2o_mean.data
bin3=ff.bin3.data
h2o_max=ff.h2o_max.data
h2o_peak_alt=ff.h2o_peak_alt.data
delta_theta=ff.delta_theta.data
top_theta=ff.top_theta.data
bot_theta=ff.bot_theta.data
bin4=ff.bin4.data
bin5=ff.bin5.data

### read overshoot data
#f=xr.open_dataset('overshooting_selected_unstable_tke_wv_theta_record.nc')
f=pickle.load( open('overshooting_selected_unstable_tke_wv_theta_record.pickle', 'rb') )
altmax = f.altmax.data
nlab = 243
altmaxall = np.zeros((4,nlab))
for i in range(nlab):
    altmaxall[0,i] = np.nanmax(altmax[:,i,0])
    altmaxall[2,i] = np.nanmax(altmax[:,i,3])
ohist1,obins1 = np.histogram(altmaxall[0,:],bins=np.arange(14,18,0.25))
ohist4,obins4 = np.histogram(altmaxall[2,:],bins=np.arange(340,420,5))



### generate figure
title = ['vertical size','top altitude','bottom altitude']
fig,ax = plt.subplots(nrows=3,ncols=3)
fig.set_size_inches(18,16, forward=True)

###### first row: vertical size


ax[0][0].fill_between(bin1,verts[0,:]/verts[0,:].sum()*100,alpha=0.2,color='grey',label='all')
ax[0][0].plot(bin1,verts[1,:]/verts[0,:].sum()*100,marker='o',color='C0',label='ICE')
ax[0][0].plot(bin1,verts[2,:]/verts[0,:].sum()*100,marker='o',color='darkorange',label='NOICE')
ax[0][0].legend(fontsize=12)


ax[0][1].fill_between(bin2,top_alt[0,:]/top_alt[0,:].sum()*100,alpha=0.2,color='grey',label='all')
ax[0][1].plot(bin2,top_alt[1,:]/top_alt[0,:].sum()*100,marker='o',color='C0',label='ICE')
ax[0][1].plot(bin2,top_alt[2,:]/top_alt[0,:].sum()*100,marker='o',color='darkorange',label='NOICE')
ax[0][1].legend(fontsize=12,loc='upper right',bbox_to_anchor=(0.999,0.9))

ax1 = ax[0][1].twinx()
ax1.plot(obins1[1:],ohist1,marker='o',color='k',label='overshoot',ms=4)
ax1.set_ylim([-6,120])
ax1.legend(fontsize=12,loc='upper right',bbox_to_anchor=(0.999,0.999))


ax[0][2].fill_between(bin2,bot_alt[0,:]/bot_alt[0,:].sum()*100,alpha=0.2,color='grey',label='all')
ax[0][2].plot(bin2,bot_alt[1,:]/bot_alt[0,:].sum()*100,marker='o',color='C0',label='ICE')
ax[0][2].plot(bin2,bot_alt[2,:]/bot_alt[0,:].sum()*100,marker='o',color='darkorange',label='NOICE')
ax[0][2].legend(fontsize=12)
for i in range(3):
    ax[0][i].set_title(title[i],fontsize=18)
    ax[0][i].set_xlabel('km',fontsize=18)
    ax[0][i].set_ylabel('Percentage (%)',fontsize=12)
ax[0][0].set_xlim([0,3.5])
ax1.set_ylabel('overshoot number',fontsize=12)


###### second row: theta

ax[1][0].fill_between(bin4,delta_theta[0,:]/np.sum(delta_theta[0,:])*100,alpha=0.2,color='grey',label='all')
ax[1][0].plot(bin4,delta_theta[1,:]/np.sum(delta_theta[0,:])*100,marker='o',color='darkorange',label='NOICE',linestyle='-')
ax[1][0].plot(bin4,delta_theta[2,:]/np.sum(delta_theta[0,:])*100,marker='o',color='C0',label='ICE',linestyle='-')

ax[1][1].fill_between(bin5,top_theta[0,:]/np.sum(top_theta[0,:])*100,alpha=0.2,color='grey',label='all')
ax[1][1].plot(bin5,top_theta[1,:]/np.sum(top_theta[0,:])*100,marker='o',color='darkorange',label='NOICE')
ax[1][1].plot(bin5,top_theta[2,:]/np.sum(top_theta[0,:])*100,marker='o',color='C0',label='ICE')

ax[1][2].fill_between(bin5,bot_theta[0,:]/np.sum(bot_theta[0,:])*100,alpha=0.2,color='grey',label='all')
ax[1][2].plot(bin5,bot_theta[1,:]/np.sum(bot_theta[0,:])*100,marker='o',color='darkorange',label='NOICE',linestyle='-')
ax[1][2].plot(bin5,bot_theta[2,:]/np.sum(bot_theta[0,:])*100,marker='o',color='C0',label='ICE',linestyle='-')


ax2 = ax[1][1].twinx()
ax2.plot(obins4[1:],ohist4,marker='o',color='k',label='overshoot',ms=4)
ax2.set_ylim([-6,120])
ax2.legend(fontsize=12,loc='upper right',bbox_to_anchor=(0.98,0.98))
ax2.set_ylabel('overshoot number',fontsize=12)


ax[1][0].set_title('delta theta',fontsize=18)
ax[1][1].set_title('max theta',fontsize=18)
ax[1][2].set_title('min theta',fontsize=18)
for i in range(3):
    ax[1][i].set_xlabel('K',fontsize=18)
    ax[1][i].set_ylabel('Percentage (%)',fontsize=12)
ax[1][0].legend(fontsize=12,loc='upper right')
ax[1][1].legend(fontsize=12,loc='upper right',bbox_to_anchor=(0.98,0.9))
ax[1][2].legend(fontsize=12,loc='upper right')


############ third row: wv mixing ratio

ax[2][0].fill_between(bin3,h2o_mean[0,:]/np.sum(h2o_mean[0,:])*100,alpha=0.2,color='grey',label='all')
ax[2][0].plot(bin3,h2o_mean[1,:]/np.sum(h2o_mean[0,:])*100,marker='o',ms=5,color='darkorange',label='NOICE',linestyle='-')
ax[2][0].plot(bin3,h2o_mean[2,:]/np.sum(h2o_mean[0,:])*100,marker='o',ms=5,color='C0',label='ICE',linestyle='-')

ax[2][1].fill_between(bin3,h2o_max[0,:]/np.sum(h2o_max[0,:])*100,alpha=0.2,color='grey',label='all')
ax[2][1].plot(bin3,h2o_max[1,:]/np.sum(h2o_max[0,:])*100,marker='o',ms=5,color='darkorange',label='NOICE')
ax[2][1].plot(bin3,h2o_max[2,:]/np.sum(h2o_max[0,:])*100,marker='o',ms=5,color='C0',label='ICE')

ax[2][2].fill_between(bin2,h2o_peak_alt[0,:]/np.sum(h2o_peak_alt[0,:])*100,alpha=0.2,color='grey',label='all')

ax3 = ax[2][2].twinx()
ax3.plot(obins1[1:],ohist1,marker='o',color='k',label='overshoot',ms=4)
ax3.set_ylim([-6,120])
ax3.legend(fontsize=12)
ax3.set_ylabel('overshoot number',fontsize=12)

ax[2][2].plot(bin2,h2o_peak_alt[1,:]/np.sum(h2o_peak_alt[0,:])*100,marker='o',color='darkorange',label='NOICE')
ax[2][2].plot(bin2,h2o_peak_alt[2,:]/np.sum(h2o_peak_alt[0,:])*100,marker='o',color='C0',label='ICE')

ax3.legend(fontsize=12,loc='upper right',bbox_to_anchor=(0.98,0.98))

ax[2][0].set_title('mean water vapor mixing ratio',fontsize=18)
ax[2][1].set_title('max water vapor mixing ratio',fontsize=18)
ax[2][2].set_title('peak altitude',fontsize=18)
ax[2][0].set_xlabel('ppmv',fontsize=18)
ax[2][1].set_xlabel('ppmv',fontsize=18)
ax[2][2].set_xlabel('km',fontsize=18)

ax[2][0].set_ylabel('Percentage (%)',fontsize=12)
ax[2][1].set_ylabel('Percentage (%)',fontsize=12)
ax[2][2].set_ylabel('Percentage (%)',fontsize=12)

ax[2][0].legend(fontsize=12,loc='upper right')
ax[2][1].legend(fontsize=12,loc='upper right')
ax[2][2].legend(fontsize=12,loc='upper right',bbox_to_anchor=(0.98,0.9))
plt.tight_layout(w_pad=1)


k=0
llabel = ascii_lowercase[:9]
for i in range(3):
    for j in range(3):
        ax[i][j].annotate('('+llabel[k]+')',xy=(0.02,0.98),xycoords='axes fraction',size=14,ha='left',va='top',color='k')

        k+=1

plt.tight_layout(w_pad=1)
plt.savefig('fig6.pdf',bbox_inches='tight')
