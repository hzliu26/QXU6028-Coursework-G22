"""Original scientific figures; PDF/SVG are vector, PNG is 600 dpi."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

BLUE='#2166AC'; RED='#B2182B'; GOLD='#B7791F'; GREY='#626B73'


def setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':9,
        'axes.titlesize':9,'axes.titleweight':'normal','axes.linewidth':.7,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7.5,
        'legend.frameon':False,'lines.linewidth':1.3,'xtick.direction':'out','ytick.direction':'out',
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white'})


def panel(ax,label,title):
    ax.set_title(title,loc='left',pad=10)
    ax.text(-.16,1.07,label,transform=ax.transAxes,fontweight='bold',fontsize=11,va='bottom')


def save(fig,folder,name):
    for ext in ('pdf','svg','png'):
        fig.savefig(folder/f'{name}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.07)
    plt.close(fig)


def make_figures(root,result,diag,arrays):
    setup();folder=root/'figures';folder.mkdir(exist_ok=True)
    a1,a2,a3=arrays;d1,d2,d3=diag['L1'],diag['L2'],diag['L3']
    fig,axs=plt.subplots(1,2,figsize=(7.05,2.65),layout='constrained')
    df=a1['sweep'];ef=df.EF_eV.to_numpy()
    axs[0].plot(ef,df.n_band_cm2/1e12,'o-',color=BLUE,ms=3,label=r'$n_{\rm band}$')
    axs[0].plot(ef,df.n_fermi_cm2/1e12,'s-',color=RED,ms=3,label=r'$n_F$')
    axs[0].set(xlabel=r'$E_F$ (eV)',ylabel=r'Carrier count ($10^{12}$ cm$^{-2}$)');axs[0].legend()
    panel(axs[0],'a','Finite temperature carrier counts')
    axs[1].plot(a1['dense_ef'],a1['dense_factor'],color=BLUE)
    axs[1].scatter(ef,df.correction_factor,s=14,color=BLUE,zorder=3)
    for value in (2,4):
        cross=d1['crossings_eV'][str(value)]
        axs[1].plot([0,cross,cross],[value,value,1],':',color=GREY,lw=.8)
        axs[1].text(cross+.006,value-.23,f'{cross:.3f} eV',fontsize=7,color=GREY)
    op=d1['parameters']['EF_op_eV'];C=result['L1']['correction_factor']
    axs[1].scatter([op],[C],s=33,color=RED,zorder=4)
    axs[1].annotate(f'Operating point\n$C$ = {C:.3f}',(op,C),xytext=(.17,2.1),arrowprops=dict(arrowstyle='-',lw=.6),fontsize=7.5)
    axs[1].set(xlabel=r'$E_F$ (eV)',ylabel=r'Correction factor $n_{\rm band}/n_F$',xlim=(0,.26),ylim=(.85,7))
    panel(axs[1],'b','Carrier model changes extracted mobility')
    save(fig,folder,'Figure_1_Carrier_counting')

    fig,axs=plt.subplots(2,2,figsize=(7.05,5.15),layout='constrained')
    ef=a2['ef'];curves=a2['curves'];s=a2['selected'];j=a2['j'];xy=a2['xy']
    best_gap=a2['best_gap']
    alternative=best_gap if s==0 else 0
    def model_label(model_index):
        if model_index==0:
            return 'Cone'
        return f"Hard gap ({a2['gaps'][model_index]:g} $k_BT$)"
    axs[0,0].plot(curves[alternative,:,0],curves[alternative,:,1],'--',color=GREY,label=model_label(alternative))
    axs[0,0].plot(curves[s,:,0],curves[s,:,1],color=RED,label=model_label(s)+' (selected)')
    axs[0,0].scatter(xy[:,0],xy[:,1],color=BLUE,s=11,zorder=3,label='Repeat means')
    axs[0,0].set(xlabel=r'Normalised signed $R_H$',ylabel=r'Normalised $R_s$');axs[0,0].legend(loc='center',bbox_to_anchor=(.5,.48))
    panel(axs[0,0],'a','Joint transport fit')
    axs[0,1].plot(a2['gaps'][1:],a2['residual'][1:],color=RED)
    axs[0,1].axhline(a2['residual'][0],color=GREY,ls='--',label='Cone RMS')
    axs[0,1].axhline(.85*a2['residual'][0],color=GOLD,ls=':',label='0.85 × cone RMS')
    axs[0,1].scatter([result['L2']['Eg_kT']],[a2['residual'][s]],s=28,color=RED,zorder=3)
    axs[0,1].set(xlabel=r'Candidate gap $E_g/(k_BT)$',ylabel='Pair space RMS residual')
    axs[0,1].legend(loc='upper center',bbox_to_anchor=(.5,-.27),ncol=2,columnspacing=1)
    panel(axs[0,1],'b','Residual across candidate gaps')
    t=a2['trajectory']
    axs[1,0].fill_between(t.measurement_index,t.EF_p16_eV*1e3,t.EF_p84_eV*1e3,color=BLUE,alpha=.22,label='16–84% repeat interval')
    axs[1,0].plot(t.measurement_index,t.EF_eV*1e3,color=BLUE,marker='o',ms=2,lw=.9)
    axs[1,0].axhline(0,color=GREY,lw=.6,ls=':');axs[1,0].set(xlabel='Measurement index',ylabel=r'Assigned $E_F$ (meV)');axs[1,0].legend()
    panel(axs[1,0],'c','Acquisition order is retained')
    for channel,color,label in ((0,BLUE,r'$R_H$'),(1,RED,r'$R_s$')):
        residual=xy[:,channel]-curves[s,j,channel]
        axs[1,1].plot(a2['ids'],residual,'o-',ms=2,lw=.7,color=color,label=label)
    axs[1,1].axhline(0,color=GREY,lw=.6,ls=':');axs[1,1].legend()
    axs[1,1].set(xlabel='Measurement index',ylabel='Normalised data minus fitted signal')
    panel(axs[1,1],'d','Residual structure')
    save(fig,folder,'Figure_2_DoS_inference')

    fig,axs=plt.subplots(2,2,figsize=(7.05,5.1),layout='constrained')
    cmap='RdBu_r';vmax=np.max(np.abs(a3['m']))*100
    im=axs[0,0].imshow(a3['m'].reshape(10,10)*100,origin='lower',extent=(-3,3,-3,3),cmap=cmap,vmin=-vmax,vmax=vmax,interpolation='nearest')
    for k,f in enumerate(result['L3']['features'],1):
        x,y=f['x'],f['y']
        axs[0,0].scatter(x,y,marker='o',s=48,facecolors='none',edgecolors='k',lw=.8)
        offset=(6,6) if y< -2 else (-8,-11)
        axs[0,0].annotate(str(k),(x,y),xytext=offset,textcoords='offset points',fontsize=7,fontweight='bold')
    axs[0,0].set(xlabel=r'$x/s$',ylabel=r'$y/s$',xticks=[-3,0,3],yticks=[-3,0,3])
    fig.colorbar(im,ax=axs[0,0],label='Relative conductivity deviation (%)',fraction=.045,pad=.04)
    panel(axs[0,0],'a','Zero mean conductivity map')
    sens=a3['sensitivity']
    axs[0,1].loglog(sens.roughness_norm,sens.residual_norm,'o-',ms=4,color=BLUE)
    selected_index=int(np.where(sens.eta_factor==d3['eta_factor_selected'])[0][0])
    for k,row in sens.iterrows():
        axs[0,1].annotate(f'{row.eta_factor:g}',(row.roughness_norm,row.residual_norm),xytext=(4,3),textcoords='offset points',fontsize=6.8)
    axs[0,1].scatter(sens.roughness_norm.iloc[selected_index],sens.residual_norm.iloc[selected_index],s=42,color=RED,zorder=4,label='Selected')
    axs[0,1].axhline(d3['mean_scan_noise_norm_estimate'],color=GREY,ls=':',label='Mean scan noise norm')
    axs[0,1].set(xlabel=r'Roughness $\|Lm\|_2$',ylabel=r'Residual $\|Am-\bar d\|_2$');axs[0,1].legend(loc='center left',bbox_to_anchor=(.01,.46))
    panel(axs[0,1],'b',r'Sensitivity to $\eta/\eta_0$')
    for psf,width,pos,color in zip(a3['psfs'],a3['widths'],[(5,5),(2,7)],[BLUE,RED]):
        row,col=pos
        x=a3['coords']-a3['coords'][col]
        axs[1,0].plot(x,psf[row]/psf[row,col],'o-',ms=3,color=color,label=f'({row}, {col}); width {width:.2f} s')
    axs[1,0].axhline(.5,color=GREY,lw=.7,ls=':');axs[1,0].set(xlabel='Distance from injection along row (s)',ylabel='Normalised reconstructed response')
    axs[1,0].legend(fontsize=6.8);panel(axs[1,0],'c','Resolution from injected features')
    # Noise limit and threshold shown on separate axes for legibility at very different scales.
    axs[1,1].hist(a3['noise_maxima']*100,bins=25,color=BLUE,alpha=.8,edgecolor='white',lw=.4)
    axs[1,1].axvline(result['L3']['detection_limit_fraction']*100,color=RED,ls='--',label=r'$\delta_{95}$')
    axs[1,1].set(xlabel='Maximum noise map amplitude (%)',ylabel='Noise realisations');axs[1,1].legend()
    panel(axs[1,1],'d','Familywise noise detection limit')
    save(fig,folder,'Figure_3_Spatial_reconstruction')

    fig,axs=plt.subplots(2,2,figsize=(7.05,5.0),layout='constrained')
    t=a2['trajectory'];ef_data=t.EF_eV.to_numpy()
    for ax,ch,color,ylabel in ((axs[0,0],0,BLUE,r'Normalised signed $R_H$'),(axs[0,1],1,RED,r'Normalised $R_s$')):
        ax.plot(a2['ef'],a2['curves'][a2['selected'],:,ch],color=color,label='Selected model')
        ax.scatter(ef_data,a2['xy'][:,ch],s=12,facecolors='white',edgecolors=color,lw=.7,label='Measured pairs')
        ax.set(xlabel=r'Assigned $E_F$ (eV)',ylabel=ylabel);ax.legend()
    panel(axs[0,0],'a','Hall channel overlay');panel(axs[0,1],'b','Sheet resistance overlay')
    axs[1,0].semilogy(np.arange(1,101),a3['singular'],'.-',ms=3,color=BLUE)
    axs[1,0].set(xlabel='Singular value index',ylabel='Singular value')
    panel(axs[1,0],'c','Weak spatial modes limit inversion')
    maxima=100*np.max(np.abs(a3['maps']),axis=1)
    threshold=100*d3['threshold_fraction']
    axs[1,1].plot(a3['factors'],maxima,'o-',color=BLUE,ms=4,lw=1)
    axs[1,1].axhline(threshold,color=RED,ls='--',label=f'{threshold:.1f}% specification')
    axs[1,1].margins(y=.12)
    axs[1,1].set(xscale='log',xlabel=r'Regularisation factor $\eta/\eta_0$',ylabel='Maximum map amplitude (%)')
    axs[1,1].legend(loc='best',frameon=True,facecolor='white',framealpha=.95,edgecolor='none')
    panel(axs[1,1],'d','Decision across tested smoothing levels')
    save(fig,folder,'Figure_4_Validation')

    fig,axs=plt.subplots(1,3,figsize=(7.05,2.3),layout='constrained')
    for k in range(3):
        axs[0].plot(a2['ids'],a2['reps'][:,k,0],'.-',ms=2,lw=.6,alpha=.6)
        axs[1].plot(a2['ids'],a2['reps'][:,k,1],'.-',ms=2,lw=.6,alpha=.6)
    axs[0].axhline(0,color=GREY,ls=':',lw=.7)
    for lo,hi in d2['zero_crossing_brackets']:
        axs[0].axvspan(lo,hi,alpha=.23,color=GOLD)
    bracket_text=', '.join(f'{lo}–{hi}' for lo,hi in d2['zero_crossing_brackets'])
    axs[0].text(.02,.04,'Zero crossing: '+bracket_text,transform=axs[0].transAxes,
                va='bottom',fontsize=7,bbox=dict(facecolor='white',alpha=.9,linewidth=0,pad=1.4))
    axs[0].set(xlabel='Measurement index',ylabel=r'Raw signed $R_H$ (supplied units)')
    axs[1].set(xlabel='Measurement index',ylabel=r'Raw $R_{xx}$ (supplied units)')
    vals,counts=np.unique(a2['boot_gap'],return_counts=True)
    axs[2].bar(vals,counts,width=.16,color=BLUE)
    axs[2].set(xlabel=r'Selected $E_g/(k_BT)$',ylabel='Resampling runs')
    panel(axs[0],'a','Hall repeats');panel(axs[1],'b','Resistance repeats');panel(axs[2],'c','Gap stability')
    save(fig,folder,'Figure_S1_Repeat_checks')
