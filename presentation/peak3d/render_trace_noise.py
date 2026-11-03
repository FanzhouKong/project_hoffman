"""Trace-noise teaching figures, verified against the repository's EIC kernels."""
import hashlib
import json
import zipfile

from render_figures import (ROOT, OUT, NAVY, TEAL, ORANGE, MUTED, LINE, plt, np,
                            to_rgba, Poly3DCollection, PdfPages, FIGS, canvas, save,
                            from_scans, K)
from render_merge_clouds import callout, export_subfigures


def example():
    rng = np.random.default_rng(271028)
    scans = np.arange(161)
    rt = scans / 60
    raw = 24 + 105*np.exp(-.5*((scans-80)/13)**2) + rng.normal(0,6,len(scans))
    delta = np.linspace(-.0018,.0018,19)
    weights = np.exp(-.5*(delta/.00085)**2)
    masses, values = [], []
    for s in scans:
        other = (12+72*np.exp(-.5*((s-105)/16)**2))*weights
        masses.append(np.concatenate([400+delta,400.008+delta]))
        values.append(np.concatenate([raw[s]*weights,other]))
    cloud = from_scans("trace_noise_demo",rt,masses,values)
    # This mass window includes the target cloud but excludes the second ion.
    tol = .002
    e = np.array([max(cloud.inten[cloud.off[s]:cloud.off[s+1]][
                    abs(cloud.mz[cloud.off[s]:cloud.off[s+1]]-400)<=tol]) for s in scans])
    np.testing.assert_allclose(e,raw,rtol=1e-6)
    kern = K.gaussian_kernel(10.)
    half = len(kern)//2
    smooth = np.convolve(np.pad(e,half,mode="edge"),kern,mode="valid")
    bounds = (53,107)
    selected = (scans<bounds[0]) | (scans>bounds[1])
    residual = e-smooth
    r = residual[selected]
    noise = 1.4826*np.median(abs(r-np.median(r)))
    E = K.eic_shape(cloud.off,cloud.mz,cloud.inten,np.array([400.]),np.array([tol]),
                    np.array([80]),np.array([bounds[0]]),np.array([bounds[1]]),80,kern)[0]
    np.testing.assert_allclose(noise,E[K.ES_NOISE],rtol=1e-10)
    sg = np.full(len(e),np.nan)
    own_selected = np.zeros(len(e),bool)
    for j in range(2,len(e)-2):
        if np.all(e[j-2:j+3]>0):
            own_selected[j] = True
            sg[j] = np.dot([-3,12,17,12,-3],e[j-2:j+3])/35
    corrected = (e-sg)*np.sqrt(35/18)
    r = corrected[own_selected]
    own = 1.4826*np.median(abs(r-np.median(r)))
    np.testing.assert_allclose(own,E[K.ES_OWN],rtol=1e-10)
    np.testing.assert_allclose(own,K.window_noise(cloud.off,cloud.mz,cloud.inten,400.,tol,0,161))
    return cloud,dict(rt=rt,raw=e,smooth=smooth,selected=selected,residual=residual,
                       sg=sg,own_selected=own_selected,corrected=corrected,
                       noise=float(noise),own=float(own),bounds=bounds,cell_floor=4.)


def axes_style(ax,residual=False):
    ax.view_init(elev=27,azim=-65)
    ax.set_box_aspect((1.45,.8,1.1))
    ax.set(xlim=(0,2.67),ylim=(-2.3,10),xticks=[0,1,2],yticks=[0,8],
           zlim=(-24,24) if residual else (0,155),
           zticks=[-20,0,20] if residual else [0,60,120])
    ax.set_yticklabels(["400.000","400.008"])
    ax.set_xlabel("Retention time (min)",fontsize=10,labelpad=5)
    ax.set_ylabel("m/z",fontsize=10,labelpad=7)
    ax.set_zlabel("Corrected residual" if residual else "Centroid intensity",fontsize=10,labelpad=5)
    ax.tick_params(labelsize=9,pad=0)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.pane.fill=False
        axis._axinfo["grid"]["color"]=to_rgba(LINE,.5)


def draw(cloud,D,own=False):
    rt,e = D["rt"],D["raw"]
    fit = D["sg"] if own else D["smooth"]
    chosen = D["own_selected"] if own else D["selected"]
    residual = D["corrected"] if own else D["residual"]
    noise = D["own"] if own else D["noise"]
    stem = "09_cloud_trace_own_noise" if own else "08_cloud_trace_noise"
    fig = canvas("N3" if own else "N2",
        "Measure a trace's own scan-to-scan fluctuations" if own else "Estimate trace noise from fluctuations around a smooth signal",
        "Optional own mode: a five-point local quadratic follows the peak; residual spread estimates its noise." if own else
        "Default EIC check: use raw − smoothed residuals outside the candidate's peak bounds.",
        "Extract an ion trace → fit each five-scan neighbourhood → estimate robust residual spread." if own else
        "Extract an ion trace → subtract its smooth trend → compare trace noise with the local cell floor.",
        "Synthetic data • exact own_noise calculation • optional eic_noise=own; also used by merge_test=own" if own else
        "Synthetic data • exact eic_shape calculation • default eic_noise=max • illustrative cell floor = 4.0")
    titles = ["1   Read a trace from the cloud",
              "2   Fit five neighbouring scans" if own else "2   Measure off-peak fluctuations",
              "3   Measure robust noise spread"]
    axes=[]
    cloud_rt = cloud.rt[cloud.scan_index()]
    cloud_mz = (cloud.mz-400)*1000
    target = abs(cloud.mz-400)<=.002
    for j in range(3):
        ax=fig.add_axes([.018+j*.32,.285,.305,.465],projection="3d",computed_zorder=False)
        axes.append(ax)
        axes_style(ax,j==2)
        if j<2:
            ax.scatter(cloud_rt[~target],cloud_mz[~target],cloud.inten[~target],
                       c="#A9BAC7",s=4,alpha=.16,edgecolors="none",depthshade=False,zorder=1)
            ax.scatter(cloud_rt[target],cloud_mz[target],cloud.inten[target],
                       c=TEAL,s=5,alpha=.35 if j==0 else .12,edgecolors="none",depthshade=False,zorder=2)
            ax.scatter(rt,np.zeros(len(rt)),e,c=NAVY if j==0 else MUTED,s=9,
                       alpha=.8 if j==0 else .45,edgecolors="none",depthshade=False,zorder=4)
        if j==1:
            ax.plot(rt,np.zeros(len(rt)),fit,c=TEAL,lw=2,zorder=5)
            for s in np.flatnonzero(chosen):
                ax.plot([rt[s]]*2,[0,0],[fit[s],e[s]],color=ORANGE,lw=1.5,alpha=.85,zorder=6)
            ax.scatter(rt[chosen],np.zeros(chosen.sum()),e[chosen],c=ORANGE,s=13,
                       edgecolors="none",depthshade=False,zorder=7)
            if not own:
                left,right=rt[list(D["bounds"])]
                tile=[(left,-2,0),(right,-2,0),(right,2,0),(left,2,0)]
                ax.add_collection3d(Poly3DCollection([tile],facecolor=to_rgba(TEAL,.15),
                                   edgecolor=to_rgba(TEAL,.6),zorder=1))
            else:
                # A real five-point stencil near the peak's rising slope.
                s=67
                ids=np.arange(s-2,s+3)
                ax.scatter(rt[ids],np.zeros(5),e[ids],facecolors="none",edgecolors=NAVY,
                           s=60,linewidths=1.2,depthshade=False,zorder=9)
        if j==2:
            # One residual per scan: no artificial replication across the mass axis.
            med=float(np.median(residual[chosen]))
            ax.set_zlabel("Corrected residual" if own else "Raw − smoothed",fontsize=10,labelpad=5)
            for level in [med-noise,med+noise]:
                tile=[(0,-2,level),(2.67,-2,level),(2.67,2,level),(0,2,level)]
                ax.add_collection3d(Poly3DCollection([tile],facecolor=to_rgba(TEAL,.11),
                                   edgecolor=to_rgba(TEAL,.55),linewidth=.7,zorder=1))
            ax.plot([0,2.67],[0,0],[med,med],color=MUTED,lw=1,ls="--",zorder=2)
            for s in np.flatnonzero(chosen):
                ax.plot([rt[s]]*2,[0,0],[med,residual[s]],color=ORANGE,lw=.7,alpha=.5,zorder=3)
            ax.scatter(rt[chosen],np.zeros(chosen.sum()),residual[chosen],c=ORANGE,s=18,
                       edgecolors="none",depthshade=False,zorder=4)
            ax.text2D(.04,.94,f"Noise = 1.4826 × MAD = {noise:.1f}",transform=ax.transAxes,
                      fontsize=11,weight="bold",color=NAVY)
            ax.text2D(.04,.87,"Planes: median ± noise",transform=ax.transAxes,fontsize=10,color=TEAL)
        fig.text(.045+j*.32,.77,titles[j],fontsize=14.5,weight="bold")
    callout(axes[0],"Maximum intensity\nin each scan",(rt[80],0,e[80]),(.13,.91),NAVY,10)
    if own:
        callout(axes[1],"Five-point local fit",(rt[67],0,fit[67]),(.03,.89),TEAL,10)
        captions=["Select the ion's m/z window.\nKeep the strongest centroid in each scan.",
                  "Fit the local quadratic, including the centre.\nKeep only five-scan stretches with signal.",
                  "Correct residuals by √(35/18), then take MAD.\nAt least 10 valid residuals are required."]
    else:
        callout(axes[1],"Smooth signal",(rt[80],0,fit[80]),(.16,.95),TEAL,10)
        callout(axes[1],"Off-peak residuals",(rt[24],0,e[24]),(.20,.6),ORANGE,10)
        captions=["Select the ion's m/z window.\nKeep the strongest centroid in each scan.",
                  "Orange: raw − smooth outside peak bounds.\nTinted base: candidate interval, excluded here.",
                  f"Robust spread of the selected residuals.\nDefault reference = max({noise:.1f}, cell floor 4.0)."]
    for j,caption in enumerate(captions):
        fig.text(.045+j*.32,.215,caption,fontsize=11,color=MUTED,linespacing=1.5)
    fig.text(.5,.169,"Same ion and time window in every panel • Panel 3 expands the residual scale • MAD = median absolute deviation from the median",
             fontsize=10,color=MUTED,ha="center")
    save(fig,stem)


def main():
    cloud,D=example()
    draw(cloud,D)
    draw(cloud,D,own=True)
    with PdfPages(OUT/"peak3d_trace_noise.pdf") as pdf:
        for _,fig in FIGS:
            pdf.savefig(fig)
    with PdfPages(OUT/"peak3d_trace_noise_subfigures.pdf") as pdf:
        panels=export_subfigures(pdf)
    cloud.save_npz(OUT/"trace_noise_cloud.npz")
    np.savez(OUT/"trace_noise_calculation.npz",**D)
    records=dict(trace_noise=D["noise"],own_noise=D["own"],cell_floor=D["cell_floor"],
                 default_reference=max(D["noise"],D["cell_floor"]),
                 off_peak_samples=int(D["selected"].sum()),own_samples=int(D["own_selected"].sum()),
                 verified_against=["eic_shape.ES_NOISE","eic_shape.ES_OWN","window_noise"],panels=panels)
    records["source_sha256"]={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                             ["peak3d/kernels.py","peak3d/features.py","presentation/peak3d/render_trace_noise.py"]}
    (OUT/"trace_noise_verification.json").write_text(json.dumps(records,indent=2)+"\n")
    names=["peak3d_trace_noise.pdf","peak3d_trace_noise_subfigures.pdf","TRACE_NOISE_NOTES.md",
           "trace_noise_verification.json","trace_noise_cloud.npz","trace_noise_calculation.npz",
           "render_trace_noise.py","render_trace_noise.sbatch","render_figures.py","render_merge_clouds.py"]
    names += [f"{stem}.{ext}" for stem in [s for s,_ in FIGS]+panels for ext in ("png","svg","pdf")]
    with zipfile.ZipFile(OUT/"Peak3D_trace_noise_figures.zip","w",zipfile.ZIP_DEFLATED) as bundle:
        for name in names:
            bundle.write(OUT/name,arcname=f"peak3d_trace_noise/{name}")
    with zipfile.ZipFile(OUT/"Peak3D_07_subfigures.zip","w",zipfile.ZIP_DEFLATED) as bundle:
        for j in range(1,4):
            for ext in ("png","svg","pdf"):
                name=f"07_cloud_noise_estimation_panel_{j}.{ext}"
                assert "HOW PEAK PICKING WORKS" not in (OUT/f"07_cloud_noise_estimation_panel_{j}.svg").read_text()
                bundle.write(OUT/name,arcname=name)
        bundle.write(OUT/"peak3d_noise_subfigures.pdf",arcname="peak3d_noise_subfigures.pdf")
    print(json.dumps(records,indent=2))
    plt.close("all")


if __name__ == "__main__":
    main()
