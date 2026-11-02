"""Point-cloud demonstrations of Peak3D persistence merging; run with render_merge.sbatch."""
import hashlib
import json
import zipfile

from scipy.interpolate import PchipInterpolator

from render_figures import (
    ROOT, OUT, NAVY, TEAL, ORANGE, MUTED, LINE, RED,
    plt, np, to_rgba, Poly3DCollection, ConvexHull, proj3d,
    PdfPages, FIGS, canvas, save, graph, params, from_scans, K,
)


def example(saddle_height):
    scan = np.arange(51)
    profile = PchipInterpolator([0,18,25,32,50], [1,100,saddle_height,70,1])(scan)
    delta = np.linspace(-.0014,.0014,15)
    weights = np.exp(-.5*(delta/.00065)**2)
    # The three contributing centre pieces sum to the intended effective apex height.
    normalizer = weights[6:9].sum()
    rng = np.random.default_rng(2710)
    masses = [400+delta+rng.uniform(-.000025,.000025,len(delta)) for s in scan]
    values = [profile[s]*weights/normalizer for s in scan]
    cloud = from_scans(f"merge_demo_saddle_{saddle_height}",scan/60,masses,values)
    original = (cloud.mz.copy(),cloud.inten.copy(),cloud.off.copy())
    P=params()
    g=graph(cloud,P)
    assert len(g["roots"])==2
    assert g["n"]==(1 if saddle_height==60 else 2)
    for before,after in zip(original,(cloud.mz,cloud.inten,cloud.off)):
        assert np.array_equal(before,after), "Merging must not change centroids"
    lower=int(g["roots"][np.argmin(g["eff"][g["roots"]])])
    higher=int(g["roots"][np.argmax(g["eff"][g["roots"]])])
    ea,eb,es=K.saddle_edges(g["nbr"],g["initial"],g["eff"])
    saddle=float(es.max())
    # Locate an actual graph edge realizing the connecting saddle, to label its lower endpoint.
    boundary=[]
    for p in range(cloud.n_points):
        for q in g["nbr"][p]:
            if q>=0 and g["initial"][p]!=g["initial"][q]:
                boundary.append((min(float(g["eff"][p]),float(g["eff"][q])),p,int(q)))
    edge=max(boundary,key=lambda e:e[0])
    saddle_point=edge[1] if g["eff"][edge[1]]<=g["eff"][edge[2]] else edge[2]
    lower_height=float(g["eff"][lower])
    persistence=lower_height-saddle
    threshold=max(P.rel_frac*lower_height,P.k_noise*P.floor)
    separation=abs(int(cloud.scan_index()[higher])-int(cloud.scan_index()[lower]))
    assert .8*P.fwhm_scans<=separation<P.far_fwhm*P.fwhm_scans
    assert np.isclose(saddle,saddle_height,atol=1e-4), (saddle,saddle_height)
    assert np.isclose(lower_height,70,atol=1e-4)
    assert (persistence<threshold)==(g["n"]==1)
    if g["n"]==1:
        assert np.all(g["lab"]==g["lab"][higher])
        assert np.array_equal(g["survivors"],[higher])
    else:
        assert g["lab"][lower]!=g["lab"][higher]
        for k in np.unique(g["initial"]):
            assert len(np.unique(g["lab"][g["initial"]==k]))==1
    return cloud,g,dict(saddle=saddle,lower_apex=lower_height,higher_apex=float(g["eff"][higher]),
                         persistence=persistence,threshold=threshold,apex_separation_scans=separation,
                         local_noise=P.floor,median_fwhm_scans=P.fwhm_scans,
                         initial_basins=len(g["roots"]),surviving_basins=g["n"],
                         points=cloud.n_points,centroids_unchanged=True,
                         lower=lower,higher=higher,saddle_point=saddle_point)


def callout(ax,text,xyz,where,color=NAVY,fontsize=11):
    xx,yy,_=proj3d.proj_transform(*xyz,ax.get_proj())
    ax.annotate(text,(xx,yy),xytext=where,textcoords="axes fraction",
                ha="center",va="center",fontsize=fontsize,weight="bold",color=color,
                bbox=dict(boxstyle="round,pad=.23",fc="white",ec="none",alpha=.96),
                arrowprops=dict(arrowstyle="->",color=color,lw=1.25),zorder=25)


def draw_example(saddle_height):
    cloud,g,meta=example(saddle_height)
    merged=g["n"]==1
    stem="05_cloud_split_merging" if merged else "06_cloud_separation_preserved"
    title="A shallow split becomes one basin" if merged else "A deep valley keeps two basins separate"
    takeaway=("Merging changes basin membership. The point cloud itself stays unchanged." if merged else
              "The lower apex clears the persistence threshold, so both basins retain their own apex.")
    fig=canvas("M1" if merged else "M2",title,
               "Compare the lower apex with the saddle connecting the two initial basins.",takeaway,
               "Synthetic point clouds • actual Peak3D decisions • default fractional rule shown; spacing and noise also affect merging")
    titles=["1   Two initial basins","2   Measure the connecting dip",
            "3   Merge into basin A" if merged else "3   Keep basins A and B"]
    h=g["eff"]
    scan=cloud.scan_index()
    mz=(cloud.mz-400)*1000
    visible=h>=2
    high,low,sad=(meta[k] for k in ("higher","lower","saddle_point"))
    point=lambda p:(scan[p],mz[p],h[p])
    initial_colors={int(g["initial"][high]):TEAL,int(g["initial"][low]):ORANGE}
    final_colors={int(g["lab"][high]):TEAL}
    if not merged:
        final_colors[int(g["lab"][low])]=ORANGE
    for j in range(3):
        ax=fig.add_axes([.018+j*.32,.285,.305,.465],projection="3d",computed_zorder=False)
        ax.view_init(elev=28,azim=-65)
        ax.set_box_aspect((1.4,.95,1.12))
        labels=g["lab"] if j==2 else g["initial"]
        colors=final_colors if j==2 else initial_colors
        for k in np.unique(labels):
            selected=(labels==k)&visible
            color=colors[int(k)] if j!=1 else "#8CA3B5"
            xy=np.column_stack([scan[selected],mz[selected]])
            hull=xy[ConvexHull(xy).vertices]
            ax.add_collection3d(Poly3DCollection([np.column_stack([hull,np.zeros(len(hull))])],
                                facecolor=to_rgba(color,.12),edgecolor=to_rgba(color,.5),
                                linewidth=.8,zorder=1))
            ax.scatter(scan[selected],mz[selected],h[selected],s=11,c=color,alpha=.72,
                       edgecolors="none",depthshade=False,zorder=3)
        roots=g["survivors"] if j==2 else g["roots"]
        ax.scatter(scan[roots],mz[roots],h[roots],s=240,marker="*",c=NAVY,
                   edgecolors="white",linewidths=1,depthshade=False,zorder=12)
        ax.set(xlim=(0,50),ylim=(-1.8,1.8),zlim=(0,125),xticks=[10,25,40],
               yticks=[-1,1],zticks=[0,50,100])
        ax.set_yticklabels(["399.999","400.001"],fontsize=9)
        ax.set_xlabel("Retention time (scans)",fontsize=10,labelpad=5)
        ax.set_ylabel("m/z",fontsize=10,labelpad=5)
        ax.set_zlabel("Effective intensity",fontsize=10,labelpad=5)
        ax.tick_params(labelsize=9,pad=0)
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
            axis.pane.fill=False
            axis._axinfo["grid"]["color"]=to_rgba(LINE,.45)
        if j==0:
            ax.text(scan[high],mz[high],h[high]+9,"A",ha="center",fontsize=13,weight="bold",color=TEAL,zorder=20)
            ax.text(scan[low],mz[low],h[low]+9,"B",ha="center",fontsize=13,weight="bold",color=ORANGE,zorder=20)
        elif j==1:
            ax.scatter([scan[sad]],[mz[sad]],[h[sad]],s=65,marker="D",c=RED,
                       edgecolors="white",linewidths=.8,depthshade=False,zorder=14)
            callout(ax,f"Lower apex = {meta['lower_apex']:.0f}",point(low),(.79,.90))
            callout(ax,f"Saddle = {meta['saddle']:.0f}",point(sad),(.15,.48),RED)
            # A vertical measurement at the lower apex; the dashed segment transfers saddle height.
            ax.plot([scan[sad],scan[low]],[mz[sad],mz[low]],[h[sad],h[sad]],
                    color=RED,ls=":",lw=1.6,zorder=8)
            ax.plot([scan[low],scan[low]],[mz[low],mz[low]],[h[sad],h[low]],
                    color=RED,lw=2,zorder=9)
            ax.text(scan[low]+2,mz[low],(h[sad]+h[low])/2,f"Δ = {meta['persistence']:.0f}",
                    color=RED,fontsize=12,weight="bold",zorder=20)
        else:
            callout(ax,"Surviving apex A" if merged else "Apex A",point(high),(.23,.94),TEAL)
            if merged:
                ax.scatter([scan[low]],[mz[low]],[h[low]],s=70,marker="o",facecolors="white",
                           edgecolors=MUTED,linewidths=1.2,depthshade=False,zorder=12)
                callout(ax,"Former root B",point(low),(.83,.76),MUTED)
            else:
                callout(ax,"Apex B",point(low),(.83,.76),ORANGE)
        fig.text(.045+j*.32,.77,titles[j],fontsize=16,weight="bold")
    captions=["Each coloured cloud has its own apex.\nThe basins meet at a connecting saddle.",
              f"Persistence = {meta['lower_apex']:.0f} − {meta['saddle']:.0f} = {meta['persistence']:.0f}.\n"
              f"{meta['persistence']:.0f} {'<' if merged else '≥'} threshold {meta['threshold']:.0f} → {'MERGE' if merged else 'KEEP SEPARATE'}.",
              "All points now share basin A.\nThe lower maximum remains in the data." if merged else
              "Two basins and two apexes survive.\nTheir memberships remain separate."]
    for j,caption in enumerate(captions):
        fig.text(.045+j*.32,.215,caption,fontsize=12,color=MUTED,linespacing=1.5)
    fig.text(.5,.169,"Here: threshold = max(0.5 × lower apex, 3 × local noise) = 35.   Apex spacing = 1.4 median widths.",
             ha="center",fontsize=11,color=MUTED)
    save(fig,stem)
    cloud.save_npz(OUT/f"{stem}.npz")
    return stem,meta


def export_subfigures(pdf):
    """Export just each axes and its annotations; omit all slide and panel titles/text."""
    outputs=[]
    for stem,fig in FIGS:
        slide_artists=list(fig.texts)+list(fig.artists)
        original_visibility=[artist.get_visible() for artist in slide_artists]
        axes_visibility=[ax.get_visible() for ax in fig.axes]
        try:
            for artist in slide_artists:
                artist.set_visible(False)
            for j,target in enumerate(fig.axes):
                for ax in fig.axes:
                    ax.set_visible(ax is target)
                fig.canvas.draw()
                bbox=target.get_tightbbox(fig.canvas.get_renderer())
                bbox=bbox.transformed(fig.dpi_scale_trans.inverted()).padded(.16)
                panel_stem=f"{stem}_panel_{j+1}"
                for ext in ("png","svg","pdf"):
                    fig.savefig(OUT/f"{panel_stem}.{ext}",dpi=300,bbox_inches=bbox)
                pdf.savefig(fig,bbox_inches=bbox)
                svg=(OUT/f"{panel_stem}.svg").read_text()
                assert "HOW PEAK PICKING WORKS" not in svg
                assert "Two initial basins" not in svg
                assert "Measure the connecting dip" not in svg
                assert "Retention time" in svg
                outputs.append(panel_stem)
        finally:
            for artist,visible in zip(slide_artists,original_visibility):
                artist.set_visible(visible)
            for ax,visible in zip(fig.axes,axes_visibility):
                ax.set_visible(visible)
    return outputs


def main():
    results=dict(draw_example(saddle) for saddle in [60,15])
    with PdfPages(OUT/"peak3d_cloud_merging_demo.pdf") as pdf:
        for stem,fig in FIGS:
            pdf.savefig(fig)
    fig,axs=plt.subplots(2,1,figsize=(16,18),layout="constrained")
    for ax,(stem,_) in zip(axs,FIGS):
        ax.imshow(plt.imread(OUT/f"{stem}.png")); ax.axis("off")
    fig.savefig(OUT/"cloud_merging_preview.png",dpi=110)
    with PdfPages(OUT/"peak3d_05_06_subfigures.pdf") as pdf:
        panels=export_subfigures(pdf)
    fig,axs=plt.subplots(2,3,figsize=(16,10),layout="constrained")
    for ax,stem in zip(axs.flat,panels):
        ax.imshow(plt.imread(OUT/f"{stem}.png")); ax.axis("off")
    fig.savefig(OUT/"subfigures_preview.png",dpi=120)
    results["subfigures"]={"count":len(panels),"stems":panels,"titles_removed":True,
                           "slide_captions_removed":True,"axes_and_annotations_retained":True}
    results["source_sha256"]={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in [
        "peak3d/kernels.py","peak3d/pick.py","peak3d/estimate.py",
        "presentation/peak3d/render_figures.py","presentation/peak3d/render_merge_clouds.py"]}
    results["scope"]="Synthetic demonstrations of the default fractional merge rule; not benchmark evidence."
    (OUT/"cloud_merging_verification.json").write_text(json.dumps(results,indent=2)+"\n")
    files=["peak3d_cloud_merging_demo.pdf","cloud_merging_preview.png","cloud_merging_verification.json",
           "MERGING_NOTES.md","render_merge_clouds.py","render_figures.py","render_merge.sbatch"]
    files.extend(f"{stem}.{ext}" for stem,_ in FIGS for ext in ["png","svg","pdf","npz"])
    panel_files=[f"{stem}.{ext}" for stem in panels for ext in ("png","svg","pdf")]
    panel_files.extend(["peak3d_05_06_subfigures.pdf","subfigures_preview.png"])
    with zipfile.ZipFile(OUT/"Peak3D_05_06_subfigures.zip","w",zipfile.ZIP_DEFLATED) as bundle:
        for name in panel_files:
            bundle.write(OUT/name,arcname=f"peak3d_subfigures/{name}")
    files.extend(panel_files)
    with zipfile.ZipFile(OUT/"Peak3D_cloud_merging.zip","w",zipfile.ZIP_DEFLATED) as bundle:
        for name in files:
            bundle.write(OUT/name,arcname=f"peak3d_cloud_merging/{name}")
    print(json.dumps(results,indent=2))
    plt.close("all")


if __name__=="__main__":
    main()
