"""QXU6028 G22 analysis, implemented from the supplied Practical Methods Guide.

All densities are cm^-2, energies eV, and L3 coordinates probe pitches.
No hidden-truth data are used. Random seeds are recorded in diagnostics.json.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

KB = 8.617333e-5
NA = 3.816e15
CHARGE = 1.602176634e-19
SEED_L2 = 602822
SEED_L3 = 602823


def fermi(E, ef, T):
    return 1.0 / (np.exp(np.clip((E - ef) / (KB*T), -60, 60)) + 1.0)


def band_count(ef, T, c):
    E = np.linspace(0., 2.5, 20001)
    return float(NA*np.trapezoid(c*E*fermi(E, ef, T), E))


def fermi_count(ef, T, c, numerical=False):
    if numerical:
        E = np.linspace(ef, 2.5, 20001)
        return float(NA*np.trapezoid(c*E*fermi(E, ef, T), E))
    kt = KB*T
    return float(NA*c*kt*(ef*np.log(2)+(np.pi**2/12)*kt))


def l1_analysis(folder):
    par = json.loads((folder/'L1_params.json').read_text())
    df = pd.read_csv(folder/'L1_data.csv')
    T, c, op, rho = [par[k] for k in ('T','c','EF_op_eV','rho_ohm_sq')]
    checks = {str(v): band_count(v,T,c)/fermi_count(v,T,c) for v in (0.,.073,.1,.2)}
    assert abs(checks['0.0']-1) < 1e-5
    assert abs(checks['0.1']-2.60) < .04
    assert abs(checks['0.2']-5.10) < .04
    assert abs(fermi_count(op,T,c,True)/fermi_count(op,T,c)-1) < 1e-5
    nf = np.array([fermi_count(v,T,c) for v in df.EF_eV])
    ratio = df.n_band_cm2.to_numpy()/nf
    assert np.all(np.diff(ratio)>0)
    nb_op = band_count(op,T,c)
    nf_op = fermi_count(op,T,c)
    factor = nb_op/nf_op
    mu = 1/(CHARGE*nf_op*rho)
    dense_ef = np.linspace(0.,.3,1201)
    dense_factor = np.array([band_count(v,T,c)/fermi_count(v,T,c) for v in dense_ef])
    crossings = {str(v):float(np.interp(v,dense_factor,dense_ef)) for v in (2,4)}
    df['n_fermi_cm2'] = nf
    df['correction_factor'] = ratio
    result = dict(correction_factor=factor,n_fermi_op_cm2=nf_op,mobility_corrected_cm2_Vs=mu)
    diag = dict(parameters=par,checks=checks,n_band_op_cm2=nb_op,
                mobility_band_cm2_Vs=1/(CHARGE*nb_op*rho),crossings_eV=crossings,
                numerical_closed_relative_error=abs(fermi_count(op,T,c,True)/nf_op-1))
    return result,diag,dict(sweep=df,dense_ef=dense_ef,dense_factor=dense_factor)


def model_library(T, c):
    kt = KB*T
    E = np.linspace(-2.5,2.5,4000)
    ef = np.linspace(-8*kt,9*kt,400)
    f = fermi(E[None,:],ef[:,None],T)
    upper = E[None,:]>=ef[:,None]
    lower = E[None,:]<=ef[:,None]
    gaps = np.r_[0.,np.arange(.5,8.001,.25)]
    curves=[]
    for gap in gaps:
        D=gap*kt/2
        g=c*np.maximum(np.abs(E)-D,0.)
        n=NA*np.trapezoid(np.where(upper,g*f,0.),E,axis=1)
        p=NA*np.trapezoid(np.where(lower,g*(1-f),0.),E,axis=1)
        rh=(p-n)/(p+n)**2
        rs=1/(p+n)
        curves.append(np.column_stack((rh/np.max(np.abs(rh)),rs/np.max(rs))))
    return ef,gaps,np.array(curves)


def normalise_pairs(raw):
    return raw/np.array([np.max(np.abs(raw[:,0])),np.max(raw[:,1])])


def fit_pairs(raw, curves, gaps):
    xy=normalise_pairs(raw)
    distances=np.sum((curves[:,None,:,:]-xy[None,:,None,:])**2,axis=-1)
    indices=np.argmin(distances,axis=2)
    residual=np.sqrt(np.mean(np.min(distances,axis=2),axis=1))
    best_gap=1+int(np.argmin(residual[1:]))
    selected=best_gap if residual[best_gap]<.85*residual[0] and gaps[best_gap]>=1 else 0
    return selected,indices[selected],residual,best_gap,xy


def l2_analysis(folder,c):
    df=pd.read_csv(folder/'L2_data.csv')
    meta=json.loads((folder/'L2_meta.json').read_text())
    # pd.unique explicitly preserves acquisition order instead of sorting by inferred EF.
    ids=pd.unique(df.measurement_index)
    reps=np.stack([df[df.measurement_index==i].sort_values('repeat')[['R_H_raw','R_xx_raw']].to_numpy() for i in ids])
    assert reps.shape==(len(ids),3,2)
    means=reps.mean(axis=1)
    sd=reps.std(axis=1,ddof=1)
    brackets=np.flatnonzero(means[:-1,0]*means[1:,0]<0)
    assert len(brackets)>0,'The supplied sweep must retain a Hall sign change.'
    ef,gaps,curves=model_library(meta['T_K'],c)
    selected,j,residual,best_gap,xy=fit_pairs(means,curves,gaps)
    rng=np.random.default_rng(SEED_L2)
    boot_gap=[];boot_ef=[];boot_selected=[]
    for _ in range(200):
        sample=reps[np.arange(len(ids)),rng.integers(0,3,len(ids)),:]
        s,ij,_,_,_=fit_pairs(sample,curves,gaps)
        boot_selected.append(s)
        boot_gap.append(gaps[s])
        boot_ef.append(ef[ij])
    boot_gap=np.array(boot_gap);boot_ef=np.array(boot_ef)
    gap_lo,gap_med,gap_hi=np.percentile(boot_gap,[16,50,84])
    lo,hi=np.percentile(boot_ef,[16,84],axis=0)
    model='cone' if selected==0 else 'hard_gap'
    result=dict(model=model,Eg_kT=float(gaps[selected]),
                Eg_uncertainty_kT=[float(gap_lo),float(gap_hi)],EF_trajectory_eV=ef[j].tolist())
    diag=dict(T_K=meta['T_K'],measurement_count=len(ids),seed=SEED_L2,bootstrap_runs=200,
              uncertainty_field_definition='[16th percentile, 84th percentile] of selected gaps, consistent with the supplied historical G22 checkpoint.',
              gap_bootstrap_median_kT=float(gap_med),gap_bootstrap_interval_kT=[float(gap_lo),float(gap_hi)],
              hard_gap_selection_fraction=float(np.mean(np.array(boot_selected)>0)),
              r_cone=float(residual[0]),r_best_gap=float(residual[best_gap]),
              best_gap_candidate_kT=float(gaps[best_gap]),residual_ratio=float(residual[best_gap]/residual[0]),
              zero_crossing_brackets=[[int(ids[k]),int(ids[k+1])] for k in brackets],
              EF_min_eV=float(ef[j].min()),EF_max_eV=float(ef[j].max()),
              selected_curve_pair_RMS=float(residual[selected]))
    traj=pd.DataFrame(dict(measurement_index=ids,EF_eV=ef[j],EF_p16_eV=lo,EF_p84_eV=hi,
                           RH_mean=means[:,0],Rs_mean=means[:,1],RH_sd=sd[:,0],Rs_sd=sd[:,1],
                           RH_normalised=xy[:,0],Rs_normalised=xy[:,1],
                           RH_fitted=curves[selected,j,0],Rs_fitted=curves[selected,j,1]))
    return result,diag,dict(df=df,reps=reps,means=means,sd=sd,ids=ids,ef=ef,gaps=gaps,
        curves=curves,selected=selected,j=j,residual=residual,best_gap=best_gap,xy=xy,
        boot_gap=boot_gap,boot_ef=boot_ef,trajectory=traj)


def gradient_matrix(npix=10):
    rows=[]
    for i in range(npix):
        for j in range(npix):
            k=i*npix+j
            for v in ([k+1] if j<npix-1 else [])+([k+npix] if i<npix-1 else []):
                r=np.zeros(npix*npix);r[k]=-1;r[v]=1;rows.append(r)
    return np.array(rows)


def inverse_operator(A,L,eta):
    N=A.shape[1]
    P=np.eye(N)-np.ones((N,N))/N
    M=P@(A.T@A+eta*L.T@L)@P+1e-9*np.eye(N)
    B=np.linalg.lstsq(M,P@A.T,rcond=None)[0]
    return P@B


def extract_features(m,delta):
    image=m.reshape(10,10);coords=np.linspace(-2.7,2.7,10)
    features=[]
    for sign,label in ((1,'positive'),(-1,'negative')):
        z=sign*image
        # Connected regions above a declared, sign-specific amplitude threshold.
        threshold=max(delta,.35*float(z.max()))
        mask=z>=threshold;visited=set()
        for i,j in zip(*np.where(mask)):
            if (i,j) in visited:continue
            stack=[(i,j)];component=[];visited.add((i,j))
            while stack:
                a,b=stack.pop();component.append((a,b))
                for u in range(max(0,a-1),min(10,a+2)):
                    for v in range(max(0,b-1),min(10,b+2)):
                        if mask[u,v] and (u,v) not in visited:
                            visited.add((u,v));stack.append((u,v))
            a,b=max(component,key=lambda ij:z[ij])
            features.append(dict(sign=label,x=float(coords[b]),y=float(coords[a]),
                                 amplitude=float(image[a,b]),pixel_row=int(a),pixel_column=int(b)))
    return sorted(features,key=lambda v:abs(v['amplitude']),reverse=True)


def l3_analysis(folder,threshold=.030):
    A=np.load(folder/'L3_A.npy')
    reps=np.loadtxt(folder/'L3_d.csv',delimiter=',')
    assert A.shape==(128,100) and reps.shape==(3,128)
    row_sum=float(np.max(np.abs(A.sum(axis=1))))
    assert row_sum<1e-11
    d=reps.mean(axis=0);sigma=reps.std(axis=0,ddof=1)
    L=gradient_matrix()
    singular=np.linalg.svd(A,compute_uv=False)
    eta0=.01*singular[0]**2/np.linalg.norm(L,2)**2
    factors=np.array([.01,.1,1,10,100])
    records=[];maps=[]
    for factor in factors:
        B=inverse_operator(A,L,eta0*factor);m=B@d;maps.append(m)
        records.append(dict(eta_factor=float(factor),eta=float(eta0*factor),
            residual_norm=float(np.linalg.norm(A@m-d)),roughness_norm=float(np.linalg.norm(L@m)),
            maximum_fraction=float(np.max(np.abs(m)))))
    noise_norm=float(np.sqrt(np.sum(sigma**2)/3))
    # Depart from eta0 only when its residual clearly exceeds repeat-based noise.
    # The closest discrepancy on a log scale gives a fixed, data-only selection rule.
    selected=2
    if records[2]['residual_norm']>3*noise_norm:
        selected=int(np.argmin([abs(np.log(rec['residual_norm']/noise_norm)) for rec in records]))
    eta=eta0*factors[selected];B=inverse_operator(A,L,eta);m=maps[selected]
    assert abs(m.mean())<1e-10
    rng=np.random.default_rng(SEED_L3)
    noises=rng.normal(size=(500,128))*sigma[None,:]
    noise_maps=noises@B.T
    maxima=np.max(np.abs(noise_maps),axis=1)
    delta=float(np.percentile(maxima,95))
    amplitude=float(np.max(np.abs(m)))
    confidence='clear pass' if amplitude+delta<threshold else ('clear fail' if amplitude-delta>threshold else 'borderline')
    coords=np.linspace(-2.7,2.7,10);X,Y=np.meshgrid(coords,coords)
    psfs=[];widths=[]
    for i,j in ((5,5),(2,7)):
        q=np.zeros(100);q[i*10+j]=1;q-=q.mean()
        qhat=(B@(A@q)).reshape(10,10)
        w=np.maximum(qhat,0)
        radius2=(X-coords[j])**2+(Y-coords[i])**2
        variance=float(np.sum(w*radius2)/(2*np.sum(w)))
        widths.append(2.355*np.sqrt(variance));psfs.append(qhat)
    features=extract_features(m,delta)
    result=dict(map=m.tolist(),features=features,resolution_probe_pitch=float(max(widths)),
                detection_limit_fraction=delta,**{'pass':bool(amplitude<=threshold),'confidence':confidence})
    diag=dict(seed=SEED_L3,noise_runs=500,max_absolute_row_sum=row_sum,map_mean=float(m.mean()),
              eta0=float(eta0),eta_selected=float(eta),eta_factor_selected=float(factors[selected]),
              eta_selection_reason='Use eta0 unless its residual exceeds 3 times the estimated mean-scan noise norm; then select the tested factor closest to that noise norm on a log scale. No hidden truth used.',
              threshold_fraction=threshold,maximum_fraction=amplitude,
              decision_envelope_fraction=[max(0.,amplitude-delta),amplitude+delta],
              detection_noise_convention='Channel-wise sample SD of single scans (ddof=1), as prescribed; not divided by sqrt(3).',
              noise_sd_median=float(np.median(sigma)),mean_scan_noise_norm_estimate=noise_norm,
              eta0_residual_norm=float(records[2]['residual_norm']),
              residual_norm=float(np.linalg.norm(A@m-d)),resolution_method='Prescribed positive-lobe second-moment FWHM equivalent',
              resolution_positions_row_column=[[5,5],[2,7]],resolution_values_probe_pitch=[float(v) for v in widths],
              numerical_rank=int(np.linalg.matrix_rank(A)),feature_threshold_rule='8-connected sign-specific regions above max(delta95, 0.35 times sign-specific maximum)',
              noise_sensitivity={str(scale):dict(
                  map_noise_RMS_median=float(np.median(np.sqrt(np.mean((scale*noise_maps)**2,axis=1)))),
                  maximum_amplitude_p05_p95=np.percentile(np.max(np.abs(m[None,:]+scale*noise_maps),axis=1),[5,95]).tolist(),
                  detection_limit_fraction=float(scale*delta)) for scale in (1,2,4)})
    return result,diag,dict(A=A,reps=reps,d=d,sigma=sigma,L=L,B=B,m=m,singular=singular,
        eta0=eta0,factors=factors,sensitivity=pd.DataFrame(records),maps=maps,noise_maxima=maxima,
        noise_maps=noise_maps,psfs=psfs,widths=widths,coords=coords)


def input_hashes(folder):
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file()}


def analyse(folder,commit):
    manifest=json.loads((folder/'manifest.json').read_text())
    for item in manifest['files']:
        assert hashlib.sha256((folder/item['name']).read_bytes()).hexdigest()==item['sha256'], 'Raw input hash mismatch: '+item['name']
    l1,d1,a1=l1_analysis(folder)
    l2,d2,a2=l2_analysis(folder,d1['parameters']['c'])
    l3,d3,a3=l3_analysis(folder)
    par=d1['parameters']
    result=dict(schema_version='QXU6028-results-v1',group=par['group'],dataset_set=par['dataset_set'],
                pipeline_commit=commit,L1=l1,L2=l2,L3=l3)
    diagnostics=dict(input_sha256=input_hashes(folder),L1=d1,L2=d2,L3=d3)
    return result,diagnostics,(a1,a2,a3)
