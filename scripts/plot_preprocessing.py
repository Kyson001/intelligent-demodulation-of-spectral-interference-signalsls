"""Python/matplotlib quantitative evidence; figures trace directly to CSV/NPZ.

Figure contract: results/preprocessing/figure_contract.md.
English figure labels avoid ambiguous font substitution; Chinese explanations in report.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/preprocessing'
FIG=OUT/'figures'
FIG.mkdir(exist_ok=True)
COLORS={'A':'#3274A1','B':'#D58432','C':'#8863A6'}
LABELS={'A':'A: Basic','B':'B: Envelope','C':'C: Envelope + Hann'}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                     'font.size':8,'axes.labelsize':8,'axes.titlesize':9,
                     'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.65,
                     'lines.linewidth':1.1,'savefig.dpi':300,'legend.frameon':False})


def save(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=300,facecolor='white')
    fig.savefig(FIG/f'{name}.svg',facecolor='white')
    svg=FIG/f'{name}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    fig.savefig(FIG/f'{name}.pdf',facecolor='white')
    plt.close(fig)


def ecdf(ax,values,**kwargs):
    v=np.sort(np.asarray(values))
    ax.plot(v,np.arange(1,len(v)+1)/len(v),**kwargs)


def label_panels(axes):
    for letter,ax in zip('abcdefghijklmnopqrstuvwxyz',np.ravel(axes)):
        title=ax.get_title()
        ax.set_title('')
        ax.set_title(f'{letter}  '+title,loc='left',fontweight='bold',pad=9)
        ax.tick_params(direction='out',length=3)


def traces():
    d=np.load(OUT/'file2000_traces.npz')
    k=d['k']; support=d['support']; raw=d['raw_k']
    lam=2*np.pi/k*1e9
    fig,axes=plt.subplots(2,3,figsize=(10.2,6.1),layout='constrained')
    ax=axes[0,0]
    ax.plot(lam,raw/np.max(raw),color='#66717E',lw=.5,label='Input')
    ax.plot(lam,d['background']/np.max(raw),color='#20252A',lw=1.2,label='Background proxy')
    ax.set(xlabel='Wavelength (nm)',ylabel='Relative intensity',title='Input and envelope')
    ax.legend(loc='upper right')
    ax=axes[0,1]
    zoom=(lam>=1559)&(lam<=1561)
    for name in COLORS:
        y=d[f'{name}_signal']; scale=np.sqrt(np.mean(y*y))
        ax.plot(lam[zoom],(y/scale)[zoom],color=COLORS[name],label=LABELS[name])
    ax.set(xlabel='Wavelength (nm)',ylabel='Signal / full-band RMS',title='Local fringes')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax=axes[0,2]
    ax.plot(lam,d['hann'],color=COLORS['C'],label='Hann multiplier')
    ax.plot(lam,support.astype(float),color='#777777',ls='--',label='Common phase-fit mask')
    ax.set(xlabel='Wavelength (nm)',ylabel='Weight / mask',ylim=(-.05,1.15),title='Window and shared support')
    ax.legend(loc='lower center')
    ax=axes[1,0]
    center=float(d['A_phase_opd_m'])*1e3
    for name in COLORS:
        axis=d[f'{name}_fft_axis']*1e3; amp=d[f'{name}_fft_amplitude']
        where=abs(axis-center)<.18
        ax.plot((axis[where]-center)*1e3,20*np.log10(np.maximum(amp[where]/amp[where].max(),1e-5)),color=COLORS[name])
    ax.set(xlabel='OPD offset from A phase fit (um)',ylabel='FFT amplitude (dB, own peak)',ylim=(-80,3),title='FFT peak shape')
    ax=axes[1,1]
    for name in COLORS:
        ax.plot(lam[support],d[f'{name}_phase_curve'][support],color=COLORS[name],alpha=.8)
    ax.set(xlabel='Wavelength (nm)',ylabel='Unwrapped phase (rad)',title='Extracted phase')
    ax=axes[1,2]
    kc=2*np.pi/(1560e-9)
    common=(k-kc)*float(d['A_phase_opd_m'])+float(d['A_phase_rad'])
    for name in COLORS:
        ax.plot(lam[support],(d[f'{name}_phase_curve']-common)[support]*1e3,color=COLORS[name])
    ax.set(xlabel='Wavelength (nm)',ylabel='Phase minus A fit (mrad)',title='Phase after common detrending')
    label_panels(axes)
    fig.legend([plt.Line2D([],[],color=COLORS[n]) for n in COLORS],list(LABELS.values()),
               loc='outside upper center',ncol=3,frameon=False)
    save(fig,'01_spectrum_fft_phase')

    # Boundary examples: first and last files, selected before looking at outcomes.
    fig,axes=plt.subplots(2,3,figsize=(10.2,5.7),layout='constrained')
    for row,sample in enumerate(['file0001','file4000']):
        d=np.load(OUT/f'{sample}_traces.npz'); k=d['k']; lam=2*np.pi/k*1e9; s=d['support']
        center=float(d['A_phase_opd_m'])*1e3
        common=(k-2*np.pi/(1560e-9))*float(d['A_phase_opd_m'])+float(d['A_phase_rad'])
        for name in COLORS:
            signal=d[f'{name}_signal']; z=(lam>=1559)&(lam<=1561)
            axes[row,0].plot(lam[z],(signal/np.sqrt(np.mean(signal**2)))[z],color=COLORS[name])
            axis=d[f'{name}_fft_axis']*1e3; amp=d[f'{name}_fft_amplitude']; z=abs(axis-center)<.15
            axes[row,1].plot((axis[z]-center)*1e3,20*np.log10(np.maximum(amp[z]/amp[z].max(),1e-5)),color=COLORS[name])
            axes[row,2].plot(lam[s],(d[f'{name}_phase_curve']-common)[s]*1e3,color=COLORS[name])
        axes[row,0].set(title=f'{sample}: local fringes',xlabel='Wavelength (nm)',ylabel='Signal / RMS')
        axes[row,0].xaxis.set_major_locator(MaxNLocator(4))
        axes[row,0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axes[row,1].set(title='FFT shape',xlabel='OPD offset (um)',ylabel='Amplitude (dB)',ylim=(-80,3))
        axes[row,2].set(title='Phase change',xlabel='Wavelength (nm)',ylabel='Phase minus A fit (mrad)')
    label_panels(axes)
    fig.legend([plt.Line2D([],[],color=COLORS[n]) for n in COLORS],list(LABELS.values()),
               loc='outside upper center',ncol=3,frameon=False)
    save(fig,'02_boundary_examples')


def population():
    df=pd.read_csv(OUT/'real_metrics.csv')
    fig,axes=plt.subplots(2,2,figsize=(8,6),layout='constrained')
    for name in COLORS:
        sub=df[df.method==name]
        ecdf(axes[0,0],sub.fwhm_um,color=COLORS[name],label=LABELS[name])
        ecdf(axes[0,1],sub.phase_residual_mrad,color=COLORS[name])
        if name!='A':
            ecdf(axes[1,0],abs(sub.delta_phase_vs_A_mrad),color=COLORS[name],label=f'{name} - A')
            ecdf(axes[1,1],abs(sub.delta_opd_vs_A_nm),color=COLORS[name],label=f'{name} - A')
    axes[0,0].set(title='FFT amplitude peak width',xlabel='FWHM (um OPD)',ylabel='Empirical cumulative fraction')
    axes[0,0].legend(loc='lower right')
    axes[0,1].set(title='Phase-fit residual (not true error)',xlabel='Weighted spectral residual RMS (mrad)',ylabel='Empirical cumulative fraction',xscale='log')
    axes[1,0].set(title='Paired center-phase change',xlabel='Absolute circular phase difference (mrad)',ylabel='Empirical cumulative fraction',xscale='log')
    axes[1,1].set(title='Paired OPD change',xlabel='Absolute phase-slope OPD difference (nm)',ylabel='Empirical cumulative fraction',xscale='log')
    axes[1,0].legend(loc='lower right'); axes[1,1].legend(loc='lower right')
    label_panels(axes)
    save(fig,'03_population_diagnostics')


def synthetic():
    df=pd.read_csv(OUT/'synthetic_metrics.csv')
    cases=['Clean','White 40 dB','White 20 dB','Offset + 40 dB']
    fig,axes=plt.subplots(1,3,figsize=(11,3.8),layout='constrained')
    for ax,metric,title,ylabel in zip(axes,['opd_error_nm','phase_error_mrad','normalized_input_rmse'],
                                   ['Known OPD error','Known phase error','Input noise sensitivity'],
                                   ['Absolute error (nm)','Circular absolute error (mrad)','Normalized input RMSE']):
        for name in COLORS:
            if metric=='normalized_input_rmse' and name=='C':
                continue
            med,p95=[],[]
            shown_cases=cases[1:] if metric=='normalized_input_rmse' else cases
            for case in shown_cases:
                v=abs(df[(df.method==name)&(df.case==case)][metric].to_numpy())
                med.append(np.median(v)); p95.append(np.quantile(v,.95))
            x=np.arange(len(shown_cases))+{'A':-.16,'B':0,'C':.16}[name]
            med=np.maximum(med,1e-8); p95=np.maximum(p95,1e-8)
            ax.plot(x,med,'o-',color=COLORS[name],ms=4,label=f'{name}: median')
            ax.plot(x,p95,'_',color=COLORS[name],ms=9)
        ticks=['40 dB','20 dB','Offset\n+ 40 dB'] if metric=='normalized_input_rmse' else ['Clean','40 dB','20 dB','Offset\n+ 40 dB']
        ax.set(title=title,ylabel=ylabel,yscale='log',xticks=np.arange(len(ticks)))
        ax.set_xticklabels(ticks)
        ax.set_xlim(-.4,len(ticks)-.6)
    label_panels(axes)
    fig.legend([plt.Line2D([],[],marker='o',color=COLORS[n],ms=4) for n in COLORS],
               ['A: Basic','B: Envelope','C: Envelope + Hann'],loc='outside upper center',ncol=3,frameon=False)
    fig.supxlabel('Synthetic controls: 60 paired signals per condition; dots = median, ticks = 95th percentile',fontsize=8)
    save(fig,'04_synthetic_controls')


if __name__=='__main__':
    traces()
    population()
    synthetic()
    print('Exported four figure sets: PNG, SVG, PDF')
