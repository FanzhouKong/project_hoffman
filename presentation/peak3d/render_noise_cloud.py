"""Explain the actual two-pass local noise-floor calculation with synthetic point clouds."""
import hashlib
import json
import zipfile
from dataclasses import replace

from render_figures import (ROOT, OUT, NAVY, TEAL, ORANGE, MUTED, LINE, RED, plt, np,
                            to_rgba, Poly3DCollection, PdfPages, FIGS, canvas, save,
                            params, from_scans, K)
from render_merge_clouds import callout, export_subfigures
from peak3d.pick import _merge, pick_cloud, TRACE_MIN_SCANS, SIDEBAND_PPM, SIDEBAND_RATIO


def synthetic_cloud():
    rng=np.random.default_rng(271027)
    rt=np.arange(181)/60
    delta=np.linspace(-.006,.006,17)
    weights=np.exp(-.5*(delta/.0028)**2)
    masses,values=[],[]
    # Uniform-in-log-mass background with local intensity differences and one sparse cell.
    for s,t in enumerate(rt):
        m=np.exp(rng.uniform(np.log(400),np.log(400.12),14))
        band=np.clip(((np.log(m)-np.log(400))/(np.log(400.12/400))*4).astype(int),0,3)
        block=min(int(t),2)
        keep=~((band==3)&(block==2)) | (rng.random(len(m))<.035)
        m,band=m[keep],band[keep]
        y=(4.5+1.4*band+1.1*block)*rng.lognormal(0,.22,len(m))
        mm,yy=[m],[y]
        for mass,apex,height,width in [(400.025,.72,160,.13),(400.083,1.76,135,.16)]:
            signal=height*np.exp(-.5*((t-apex)/width)**2)*weights
            active=signal>2.5
            if active.any():
                mm.append(mass+delta[active]+rng.uniform(-.00002,.00002,active.sum()))
                yy.append(signal[active])
            # Sparse satellite-like points: below 1/10 of a nearby strong trace.
            if s%4==0 and abs(t-apex)<.16:
                mm.append(np.array([mass+.020+rng.uniform(-.002,.002)]))
                yy.append(np.array([3.0]))
        masses.append(np.concatenate(mm)); values.append(np.concatenate(yy))
    return from_scans("noise_cloud_demo",rt,masses,values)


def compute_noise(cloud):
    P=replace(params(),span=3.,floor=float(np.percentile(cloud.inten,1)),n_mz_bands=4)
    sig=K.per_point_sigma(cloud.inten,P.sig_a,P.sig_b)
    nbr=K.find_neighbors(cloud.off,cloud.mz,cloud.inten,sig,P.K,P.tol_max_ppm,
                         P.same_scan_k,P.same_scan_floor_ppm,P.link_da)
    eff=K.effective_intensity(nbr,cloud.inten)
    lab,roots=K.dense_labels(K.resolve_roots(K.link_steepest(nbr,eff,cloud.inten)))
    B=len(roots)
    smin,smax,_=K.basin_extent(lab,cloud.off,B)
    lo,hi=float(cloud.mz.min()),float(cloud.mz.max())*(1+1e-9)
    args=(cloud.rt,lo,hi,P.n_mz_bands,P.rt_block_min,P.noise_cell_min_points,P.floor)
    initial,_=K.noise_cell_medians(cloud.off,cloud.mz,cloud.inten,lab,smin,smax,*args)
    scans=cloud.scan_index()
    cells=np.array([K.cell_of(float(m),float(t),cloud.rt[0],lo,hi,P.n_mz_bands,
                             initial.shape[1],P.rt_block_min)
                    for m,t in zip(cloud.mz,cloud.rt[scans])])
    bands,blocks=cells.T
    smooth=K.smooth_traces(nbr,eff,P.n_smooth)
    lab1,B1,*_= _merge(P,nbr,lab,eff,smooth,eff[roots],scans[roots].astype(np.int64),
                       cloud.mz[roots],sig[roots].astype(float),cloud.off,cloud.mz,cloud.inten,
                       initial[bands[roots],blocks[roots]],B)
    smin1,smax1,_=K.basin_extent(lab1,cloud.off,B1)
    isolated=K.short_basin_mask(lab1,smin1,smax1,2)
    trace=~K.short_basin_mask(lab1,smin1,smax1,TRACE_MIN_SCANS-1)
    side=K.sideband_mask(cloud.off,cloud.mz,cloud.inten,isolated,trace,SIDEBAND_PPM,SIDEBAND_RATIO)
    noise=isolated&~side
    med,counts=K.cell_medians(cloud.off,cloud.mz,cloud.inten,noise,*args)
    # Check the illustrated first-pass masks against the complete picker calculation.
    picked=pick_cloud(cloud,P)
    np.testing.assert_allclose(med,picked.noise_surface,rtol=0,atol=0)
    for i in range(med.shape[0]):
        for j in range(med.shape[1]):
            vals=cloud.inten[noise&(bands==i)&(blocks==j)]
            assert len(vals)==counts[i,j]
            expected=float(np.median(vals)) if len(vals)>=P.noise_cell_min_points else P.floor
            assert np.isclose(med[i,j],expected)
    assert noise.sum()>100 and side.sum()>0 and trace.sum()>100
    assert np.any(counts<50) and np.any(counts>=50)
    edges=np.exp(np.linspace(np.log(lo),np.log(hi),P.n_mz_bands+1))
    return P,dict(noise=noise,side=side,trace=trace,isolated=isolated,med=med,counts=counts,
                  bands=bands,blocks=blocks,edges=edges,lab=lab1)


def draw_noise(cloud,P,D):
    fig=canvas("N1","Estimate the local noise floor from isolated centroids",
               "After an initial basin merge, short-lived points supply the noise estimate; coherent peaks do not.",
               "Select isolated points → exclude nearby sidebands → take a median in each time–m/z cell.",
               "Synthetic data • exact picker noise-floor calculation • four mass bands shown for clarity (default: 16) • 1-minute cells")
    rt=cloud.rt[cloud.scan_index()]
    mz=(cloud.mz-400)*1000
    y=cloud.inten
    noise,side=D["noise"],D["side"]
    other=~(noise|side)
    titles=["1   Peaks and scattered points","2   Select noise samples","3   Estimate each cell's floor"]
    axes=[]
    for j in range(3):
        ax=fig.add_axes([.018+j*.32,.285,.305,.465],projection="3d",computed_zorder=False)
        axes.append(ax)
        ax.view_init(elev=28,azim=-64)
        ax.set_box_aspect((1.4,1,1.12))
        if j==0:
            ax.scatter(rt,mz,y,c="#7E9CAF",s=5.5,alpha=.55,edgecolors="none",depthshade=False,zorder=3)
        elif j==1:
            ax.scatter(rt[other],mz[other],y[other],c="#A9BAC7",s=5,alpha=.24,edgecolors="none",depthshade=False,zorder=2)
            ax.scatter(rt[noise],mz[noise],y[noise],c=ORANGE,s=9,alpha=.8,edgecolors="none",depthshade=False,zorder=4)
            ax.scatter(rt[side],mz[side],y[side],c=RED,s=17,marker="x",linewidths=.9,depthshade=False,zorder=5)
        else:
            ax.scatter(rt[noise],mz[noise],y[noise],c=ORANGE,s=10,alpha=.65,edgecolors="none",depthshade=False,zorder=4)
            for band in range(D["med"].shape[0]):
                for block in range(D["med"].shape[1]):
                    m0,m1=(D["edges"][band:band+2]-400)*1000
                    level=D["med"][band,block]
                    sparse=D["counts"][band,block]<P.noise_cell_min_points
                    color=MUTED if sparse else TEAL
                    tile=[(block,m0,level),(block+1,m0,level),(block+1,m1,level),(block,m1,level)]
                    ax.add_collection3d(Poly3DCollection([tile],facecolor=to_rgba(color,.25),
                                         edgecolor=to_rgba(color,.8),linewidth=.8,zorder=2))
        for t in range(4):
            ax.plot([t,t],[0,120],[0,0],color=LINE,lw=.6,zorder=1)
        for mass in D["edges"]:
            ax.plot([0,3],[(mass-400)*1000]*2,[0,0],color=LINE,lw=.6,zorder=1)
        ax.set(xlim=(0,3),ylim=(0,120),zlim=(0,180 if j<2 else max(22,float(y[noise].max())*1.06)),
               xticks=[0,1,2],yticks=[0,60,120],zticks=[0,80,160] if j<2 else [0,10,20])
        ax.set_yticklabels(["400.00","400.06","400.12"],fontsize=9)
        ax.set_xlabel("Retention time (min)",fontsize=10,labelpad=5)
        ax.set_ylabel("m/z",fontsize=10,labelpad=5)
        ax.set_zlabel("Centroid intensity",fontsize=10,labelpad=5)
        ax.tick_params(labelsize=9,pad=0)
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
            axis.pane.fill=False
            axis._axinfo["grid"]["color"]=to_rgba(LINE,.45)
        fig.text(.045+j*.32,.77,titles[j],fontsize=15.5,weight="bold")
    top=int(np.argmax(y))
    callout(axes[0],"Coherent peak cloud",(rt[top],mz[top],y[top]),(.36,.95),TEAL,10)
    eligible=np.flatnonzero(noise&(rt>1)&(rt<2)&(mz<35))
    p=int(eligible[np.argmax(y[eligible])])
    callout(axes[1],"Noise samples",(rt[p],mz[p],y[p]),(.19,.44),ORANGE,10)
    eligible=np.flatnonzero(side)
    p=int(eligible[np.argmin(rt[eligible])])
    callout(axes[1],"Sideband excluded",(rt[p],mz[p],y[p]),(.08,.72),RED,10)
    b,k=0,1
    level=float(D["med"][b,k])
    mid_mz=(np.mean(D["edges"][b:b+2])-400)*1000
    callout(axes[2],f"Cell median = {level:.1f}",(k+.5,mid_mz,level),(.14,.9),TEAL,10)
    sparse=np.argwhere(D["counts"]<50)[-1]
    b,k=map(int,sparse)
    mid_mz=(np.mean(D["edges"][b:b+2])-400)*1000
    callout(axes[2],"Sparse cell\nuses file floor",(k+.5,mid_mz,float(D["med"][b,k])),(.75,.77),MUTED,10)
    captions=["Longer structures hold coherent signal.\nShort basins supply candidate noise points.",
              "Use basins spanning at most 2 scans.\nExclude points near much stronger traces.",
              "Median of eligible raw intensities per cell.\nFewer than 50 points? Use the file's 1st percentile."]
    for j,caption in enumerate(captions):
        fig.text(.045+j*.32,.215,caption,fontsize=11.5,color=MUTED,linespacing=1.5)
    fig.text(.5,.169,"Orange = eligible noise samples   •   Red crosses = excluded sidebands   •   Panel 3 enlarges the low-intensity range",
             fontsize=10.5,color=MUTED,ha="center")
    save(fig,"07_cloud_noise_estimation")


def main():
    cloud=synthetic_cloud()
    P,D=compute_noise(cloud)
    draw_noise(cloud,P,D)
    with PdfPages(OUT/"peak3d_noise_subfigures.pdf") as pdf:
        panels=export_subfigures(pdf)
    fig,axs=plt.subplots(1,3,figsize=(16,5.5),layout="constrained")
    for ax,stem in zip(axs,panels):
        ax.imshow(plt.imread(OUT/f"{stem}.png")); ax.axis("off")
    fig.savefig(OUT/"noise_subfigures_preview.png",dpi=120)
    cloud.save_npz(OUT/"07_cloud_noise_estimation.npz")
    np.savez(OUT/"noise_selection.npz",**D)
    records=dict(n_centroids=cloud.n_points,n_noise=int(D["noise"].sum()),
                 n_sidebands_excluded=int(D["side"].sum()),n_trace=int(D["trace"].sum()),
                 noise_cell_counts=D["counts"].tolist(),noise_floor=D["med"].tolist(),
                 file_fallback=P.floor,params=P.to_dict(),exact_match_to_pick_cloud=True,
                 cell_medians_independently_verified=True,panel_stems=panels)
    records["source_sha256"]={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in [
        "peak3d/kernels.py","peak3d/pick.py","peak3d/estimate.py",
        "presentation/peak3d/render_figures.py","presentation/peak3d/render_merge_clouds.py",
        "presentation/peak3d/render_noise_cloud.py"]}
    (OUT/"noise_verification.json").write_text(json.dumps(records,indent=2)+"\n")
    files=["NOISE_NOTES.md","render_noise_cloud.py","render_noise.sbatch","render_figures.py",
           "render_merge_clouds.py","noise_verification.json","noise_selection.npz",
           "07_cloud_noise_estimation.npz","peak3d_noise_subfigures.pdf","noise_subfigures_preview.png"]
    files.extend(f"{stem}.{ext}" for stem in ["07_cloud_noise_estimation"]+panels for ext in ["png","svg","pdf"])
    with zipfile.ZipFile(OUT/"Peak3D_noise_figures.zip","w",zipfile.ZIP_DEFLATED) as bundle:
        for name in files:
            bundle.write(OUT/name,arcname=f"peak3d_noise/{name}")
    print(json.dumps(records,indent=2))
    plt.close("all")


if __name__=="__main__":
    main()
