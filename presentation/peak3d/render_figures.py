"""Slide figures using synthetic centroids and the repository's actual picking kernels.

Run on a compute node with render.sbatch. No benchmark or source data are modified.
"""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from peak3d import kernels as K
from peak3d.estimate import Params
from peak3d.features import eic_check
from peak3d.io import from_scans
from peak3d.pick import _merge

OUT = Path(__file__).resolve().parent
NAVY, TEAL, ORANGE, PURPLE = "#142D4E", "#008C95", "#E78A32", "#8863B6"
MUTED, LIGHT, LINE, RED = "#53677D", "#F2F6FA", "#DCE5EE", "#C75157"
PALETTE = [TEAL, ORANGE, PURPLE, "#438CCB", "#C3597C", "#91A747"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "text.color": NAVY,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": LINE, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 16,
    "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.facecolor": "white",
})
FIGS, METADATA = [], {}


def canvas(step, title, subtitle, takeaway, foot="Synthetic illustration • current repository implementation"):
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(.055, .945, "Peak3D  /  HOW PEAK PICKING WORKS", fontsize=12, weight="bold", color=TEAL)
    fig.text(.945, .945, f"{step:02d}" if isinstance(step, int) else step,
             ha="right", fontsize=13, weight="bold", color=TEAL)
    fig.text(.055, .883, title, fontsize=27, weight="bold")
    fig.text(.055, .838, subtitle, fontsize=14, color=MUTED)
    fig.add_artist(FancyBboxPatch((.045, .071), .91, .082, transform=fig.transFigure,
                                boxstyle="round,pad=0.009,rounding_size=0.012", ec="none", fc=LIGHT))
    fig.text(.065, .111, takeaway, fontsize=17, weight="bold", va="center")
    fig.text(.055, .028, foot, fontsize=10, color=MUTED)
    return fig


def save(fig, stem):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=190)
    FIGS.append((stem, fig))


def params():
    return Params(dt=1/60, span=2, floor=1, fwhm_med=10/60, fwhm_scans=10,
                  sig_a=.5, sig_b=0, K=1, n_smooth=9, scan_gain=False)


def graph(cloud, P):
    sig = K.per_point_sigma(cloud.inten, P.sig_a, P.sig_b)
    nbr = K.find_neighbors(cloud.off, cloud.mz, cloud.inten, sig, P.K,
                           P.tol_max_ppm, P.same_scan_k, P.same_scan_floor_ppm, P.link_da)
    eff = K.effective_intensity(nbr, cloud.inten)
    parent = K.link_steepest(nbr, eff, cloud.inten)
    initial, roots = K.dense_labels(K.resolve_roots(parent))
    scans = cloud.scan_index()
    smoothed = K.smooth_traces(nbr, eff, P.n_smooth)
    lab, n, survivors, persistence, _, _ = _merge(
        P, nbr, initial, eff, smoothed, eff[roots], scans[roots].astype(np.int64),
        cloud.mz[roots], sig[roots].astype(float), cloud.off, cloud.mz, cloud.inten,
        np.full(len(roots), P.floor), len(roots))
    assert np.all(K.resolve_roots(parent)[parent] == K.resolve_roots(parent))
    assert len(lab) == cloud.n_points and np.all(lab >= 0)
    return dict(nbr=nbr, eff=eff, parent=parent, initial=initial, roots=roots,
                lab=lab, survivors=roots[survivors], n=n, persistence=persistence)


def cloud_from_traces(traces, masses, dt=1/60, jitter=0):
    rng = np.random.default_rng(14)
    masses = np.asarray(masses)
    n = len(traces[0])
    return from_scans("presentation_synthetic", np.arange(n)*dt,
                      [masses + rng.normal(0, jitter, len(masses)) for _ in range(n)],
                      [np.array([v[s] for v in traces]) for s in range(n)])


def neat(ax, xlabel="Scan (retention-time order)", ylabel="Intensity (a.u.)"):
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.grid(axis="y", color=LINE, lw=.7, alpha=.6)
    ax.set_axisbelow(True)


def three_d_example():
    s = np.arange(43)
    a = 1 + 85*np.exp(-.5*((s-11)/3.3)**2) + 65*np.exp(-.5*((s-31)/3.5)**2)
    b = 1 + 90*np.exp(-.5*((s-16)/4)**2)
    # A main centroid plus a nearby split piece, and a separate co-eluting ion.
    masses = [[400+.00012*np.sin(t*.5),400.0015+.00012*np.sin(t*.5),
               400.008+.00010*np.cos(t*.4)] for t in s]
    values = [[a[t],.2*a[t],b[t]] for t in s]
    cloud = from_scans("3d_explanation", s/60, masses, values)
    g = graph(cloud,params())
    assert g["n"] == 3, g["n"]
    assert all(g["lab"][cloud.off[t]] == g["lab"][cloud.off[t]+1] for t in s)
    assert all(g["lab"][cloud.off[t]] != g["lab"][cloud.off[t]+2] for t in s)
    return cloud,g


def figure_what_is_3d():
    cloud,g = three_d_example()
    s = cloud.scan_index()
    m = (cloud.mz-400)*1000
    h = g["eff"]
    colors = np.array([PALETTE[k] for k in g["lab"]])
    fig = canvas("A", "What does the 3D in Peak3D mean?",
                 "An intensity landscape over retention time and m/z, represented by connected centroids.",
                 "Time and m/z define local neighbours; intensity directs their paths toward apexes.",
                 "Synthetic centroid graph • actual Peak3D links and basin labels • no fitted surface or voxel grid")
    ax = fig.add_axes([.02,.235,.49,.55],projection="3d")
    ax.view_init(elev=27,azim=-66)
    ax.set_box_aspect((1.6,1,1.1))
    for k in np.unique(g["lab"]):
        sel = g["lab"]==k
        ax.scatter(s[sel],m[sel],h[sel],s=20,color=PALETTE[k],depthshade=False)
        ax.scatter(s[sel],m[sel],np.zeros(sel.sum()),s=11,color=PALETTE[k],alpha=.20,depthshade=False)
    for p,q in enumerate(g["parent"]):
        if p==q:
            continue
        ax.plot([s[p],s[q]],[m[p],m[q]],[h[p],h[q]],color=colors[p],lw=.8,alpha=.6)
        if s[p] % 3 == 0 and h[p]>8:
            d = np.array([s[q]-s[p],m[q]-m[p],h[q]-h[p]])
            ax.quiver(s[p],m[p],h[p],*d,color=colors[p],arrow_length_ratio=.3,linewidth=1.1)
    roots = g["survivors"]
    for k,p in enumerate(roots):
        ax.scatter([s[p]],[m[p]],[h[p]],marker="*",s=160,color=NAVY,depthshade=False)
        ax.text(s[p],m[p],h[p]+10,chr(65+k),fontsize=14,weight="bold",color=NAVY)
    ax.set(xlim=(0,42),ylim=(-1,10),zlim=(0,125),xticks=[0,10,20,30],yticks=[0,8],zticks=[0,50,100])
    ax.set_yticklabels(["400.000","400.008"],fontsize=10)
    ax.set_xlabel("Retention time (scan order)",labelpad=7,fontsize=11)
    ax.set_ylabel("m/z",labelpad=6,fontsize=12)
    ax.set_zlabel("Effective intensity",labelpad=7,fontsize=11)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.pane.fill=False
        axis._axinfo["grid"]["color"]=to_rgba(LINE,.65)
    fig.text(.555,.762,"The same graph, viewed from above",fontsize=17,weight="bold")
    ax = fig.add_axes([.585,.493,.345,.222])
    for p,q in enumerate(g["parent"]):
        if p != q:
            ax.annotate("",(s[q],m[q]),(s[p],m[p]),
                        arrowprops=dict(arrowstyle="-|>",mutation_scale=7,lw=.8,
                                        color=colors[p],shrinkA=2,shrinkB=2),zorder=1)
    ax.scatter(s,m,s=9+h*.18,c=colors,edgecolors="white",linewidths=.3,zorder=2)
    for k,p in enumerate(roots):
        ax.scatter(s[p],m[p],marker="*",s=110,color=NAVY,zorder=3)
        ax.text(s[p],m[p]+1.4,chr(65+k),ha="center",fontsize=12,weight="bold")
    ax.set(xlim=(-1,43),ylim=(-1,11),xticks=[0,10,20,30,40],yticks=[0,8])
    ax.set_yticklabels(["400.000","400.008"])
    ax.set_xlabel("Retention time (scan order)",fontsize=10)
    ax.set_ylabel("m/z",fontsize=11)
    ax.tick_params(labelsize=10)
    fig.text(.555,.388,"Position: retention time + m/z",fontsize=15,weight="bold")
    fig.text(.555,.351,"Links use nearby scans and pair-specific mass tolerance.",fontsize=12,color=MUTED)
    fig.text(.555,.295,"Height: effective intensity",fontsize=15,weight="bold")
    fig.text(.555,.258,"Same-scan split pieces contribute to the local height.",fontsize=12,color=MUTED)
    fig.text(.555,.202,"Colour: centroids assigned to the same surviving apex.",fontsize=12,color=MUTED)
    fig.text(.065,.202,"A and C: same m/z, different RT.   B: a co-eluting ion at a different m/z.",fontsize=10.5,color=MUTED)
    METADATA["3d_explanation"]={"surviving_basins":g["n"],"split_membership_verified":True,
                               "separate_mass_verified":True,"points":cloud.n_points}
    cloud.save_npz(OUT/"synthetic_3d_graph.npz")
    save(fig,"00a_what_3d_means")


def figure_comparison():
    cloud,g = three_d_example()
    s = cloud.scan_index()
    m = (cloud.mz-400)*1000
    fig = canvas("B", "The difference is how peak candidates are formed",
                 "Both approaches start with retention time, m/z and intensity. They organise the data differently.",
                 "Trace-based: form traces, then find peaks.   Peak3D: form apex basins directly from centroids.",
                 "Architecture schematic • centWave / MassCube: ROI then segmentation; asari: composite mass tracks • sources in notes")
    for y,color in [(.455,PURPLE),(.18,TEAL)]:
        fig.add_artist(FancyBboxPatch((.045,y),.91,.26,transform=fig.transFigure,
                       boxstyle="round,pad=0.004,rounding_size=0.01",facecolor=to_rgba(color,.035),
                       edgecolor=to_rgba(color,.15),zorder=0))
    fig.text(.06,.736,"TRACE-BASED DETECTION",color=PURPLE,weight="bold",fontsize=14)
    fig.text(.06,.430,"Peak3D CANDIDATE DETECTION",color=TEAL,weight="bold",fontsize=14)
    # First lane: a generic trace-based architecture, not a run of a named competitor.
    a = fig.add_axes([.08,.525,.19,.13])
    for ion,color in [(0,TEAL),(1,PURPLE)]:
        mask = cloud.mz < 400.004 if ion==0 else cloud.mz>400.004
        lo,hi=(-.5,2.1) if ion==0 else (7.5,8.5)
        a.axhspan(lo,hi,color=color,alpha=.09)
        a.scatter(s[mask],m[mask],s=5+cloud.inten[mask]*.09,color=color)
    a.set(ylim=(-1,10),xlim=(-1,43),xticks=[],yticks=[0,8])
    a.set_yticklabels(["m₁","m₂"],fontsize=11)
    a.set_xlabel("Retention time",fontsize=10,labelpad=2)
    a.set_ylabel("m/z",fontsize=10,labelpad=3)
    fig.text(.075,.680,"1  Assemble traces / ROIs",fontsize=14,weight="bold")
    a = fig.add_axes([.385,.525,.20,.13])
    x=np.arange(43)
    for k,color,offset in [(0,TEAL,0),(2,PURPLE,1.3)]:
        inds=cloud.off[:-1]+k
        y=cloud.inten[inds]/100
        a.plot(x,y+offset,lw=1.8,color=color)
        for p in g["survivors"]:
            if (k==0 and cloud.mz[p]<400.004) or (k==2 and cloud.mz[p]>400.004):
                a.plot(s[p],float(cloud.inten[p]/100+offset),"o",ms=5,color=color)
    a.set(xlim=(-1,43),xticks=[],yticks=[])
    a.set_xlabel("Retention time",fontsize=10,labelpad=2)
    a.set_ylabel("Intensity",fontsize=10,labelpad=3)
    fig.text(.38,.680,"2  Find peaks within traces",fontsize=14,weight="bold")
    fig.text(.69,.680,"3  Report peak candidates",fontsize=14,weight="bold")
    fig.text(.69,.584,"A trace is the parent object.\nChromatographic segmentation\ndefines peaks inside it.",fontsize=13,color=MUTED,linespacing=1.6)
    fig.text(.075,.475,"Trace building may use intensity and adaptive tolerances; implementations differ.",fontsize=10.5,color=MUTED)
    # Second lane: actual neighbourhood graph and merged basin labels.
    for j,left in enumerate([.08,.385]):
        a=fig.add_axes([left,.25,.19 if j==0 else .20,.12])
        cols = np.array([PALETTE[k] for k in g["lab"]]) if j else np.full(cloud.n_points,NAVY)
        for p,q in enumerate(g["parent"]):
            if p != q:
                a.plot([s[p],s[q]],[m[p],m[q]],lw=.7,color=cols[p],alpha=.6)
        a.scatter(s,m,s=5+cloud.inten*.11,c=cols)
        if j:
            for k,p in enumerate(g["survivors"]):
                a.scatter(s[p],m[p],marker="*",s=70,color=NAVY,zorder=3)
                a.text(s[p],m[p]+1.1,chr(65+k),fontsize=10,ha="center",weight="bold")
        a.set(ylim=(-1,11),xlim=(-1,43),xticks=[],yticks=[0,8])
        a.set_yticklabels(["m₁","m₂"],fontsize=11)
        a.set_xlabel("Retention time",fontsize=10,labelpad=2)
        a.set_ylabel("m/z",fontsize=10,labelpad=3)
    fig.text(.075,.393,"1  Link local centroids uphill",fontsize=14,weight="bold")
    fig.text(.38,.393,"2  Form and merge apex basins",fontsize=14,weight="bold")
    fig.text(.69,.393,"3  Validate peak candidates",fontsize=14,weight="bold")
    fig.text(.69,.298,"A basin is the candidate object.\nApply basin gates, an EIC check,\nand the final score.",fontsize=13,color=MUTED,linespacing=1.6)
    fig.text(.075,.197,"Dot size shows intensity. Peak3D panels show actual graph and basin assignments.",fontsize=10.5,color=MUTED)
    for y in [.587,.302]:
        for x0,x1 in [(.285,.35),(.60,.66)]:
            fig.add_artist(FancyArrowPatch((x0,y),(x1,y),transform=fig.transFigure,
                           arrowstyle="-|>",mutation_scale=18,lw=1.5,color=MUTED))
    METADATA["comparison"]={"trace_lane":"Conceptual architecture; no competitor executed",
                            "peak3d_lane":"Computed local graph and merged basins",
                            "sources":["https://doi.org/10.1186/1471-2105-9-504",
                                       "https://doi.org/10.1038/s41467-023-39889-1",
                                       "https://doi.org/10.1038/s41467-025-60640-5"]}
    save(fig,"00b_trace_based_vs_peak3d")


def figure_cloud():
    x = np.arange(73)
    a = 1 + 100*np.exp(-.5*((x-24)/6)**2) + 72*np.exp(-.5*((x-49)/5)**2)
    a[24] *= .78
    b = 1 + 83*np.exp(-.5*((x-35)/6)**2)
    # Deliberately broad, synthetic split-centroid clouds make the basin geometry visible.
    # These are actual input points, not display jitter added to precomputed trace labels.
    delta = np.linspace(-.0014,.0014,15)
    weight = np.exp(-.5*(delta/.00065)**2)
    masses,values = [],[]
    rng=np.random.default_rng(2026)
    for s in x:
        drift=.00006*np.sin(s*.4)
        masses.append(np.concatenate([400+delta+drift+rng.uniform(-.000025,.000025,len(delta)),
                                      400.006+delta-drift+rng.uniform(-.000025,.000025,len(delta))]))
        values.append(np.concatenate([a[s]*weight/2.9,b[s]*weight/2.9]))
    cloud = from_scans("illustrative_dense_centroid_cloud",x/60,masses,values)
    g = graph(cloud, params())
    assert len(g["roots"]) == 4 and g["n"] == 3, (len(g["roots"]), g["n"])
    fig = canvas(1, "A basin is a cloud of points that leads to one apex",
                 "Each dot is a centroid. Colour shows basin membership; a star marks its apex.",
                 "BASIN = points sharing an apex.     APEX = the endpoint of their uphill paths.",
                 "Illustrative synthetic clouds with broadened m/z spread • memberships computed by Peak3D • faint patches = projected footprints")
    titles = ["1   Start with a point cloud", "2   Assign points to apexes", "3   Merge the shallow split"]
    captions = ["Positions: retention time and m/z.\nHeight: effective intensity.",
                "Four initial basins.\nThe shallow dip creates an extra apex.",
                "Three surviving basins.\nThe weak split joins its neighbour."]
    s = cloud.scan_index()
    mz = (cloud.mz-400)*1000
    height=g["eff"]
    visible=height>=2
    final_colors=[TEAL,PURPLE,"#438CCB"]
    initial_colors={int(g["initial"][p]):final_colors[k] for k,p in enumerate(g["survivors"])}
    for k in np.unique(g["initial"]):
        if int(k) not in initial_colors:
            initial_colors[int(k)]=ORANGE
    axes=[]
    for j in range(3):
        ax = fig.add_axes([.018+j*.32, .285, .305, .465], projection="3d",computed_zorder=False)
        axes.append(ax)
        ax.view_init(elev=29, azim=-62)
        ax.set_box_aspect((1.35,1,1.12))
        if j == 0:
            ax.scatter(s[visible],mz[visible],height[visible],c="#7292AA",s=9,alpha=.65,
                       edgecolors="none",depthshade=False,zorder=3)
        else:
            labels = g["initial"] if j == 1 else g["lab"]
            for k in np.unique(labels):
                sel = (labels == k)&visible
                color=initial_colors[int(k)] if j==1 else final_colors[k]
                xy=np.column_stack([s[sel],mz[sel]])
                hull=xy[ConvexHull(xy).vertices]
                footprint=Poly3DCollection([np.column_stack([hull,np.zeros(len(hull))])],
                                           facecolor=to_rgba(color,.12),edgecolor=to_rgba(color,.55),
                                           linewidth=.9,zorder=1)
                ax.add_collection3d(footprint)
                ax.scatter(s[sel],mz[sel],height[sel],color=color,s=10,alpha=.7,
                           edgecolors="none",depthshade=False,zorder=3)
            roots = g["roots"] if j == 1 else g["survivors"]
            ax.scatter(s[roots],mz[roots],height[roots],marker="*",s=245,
                       color=NAVY,edgecolor="white",linewidth=1,depthshade=False,zorder=10)
            for p in roots:
                ax.plot([s[p],s[p]],[mz[p],mz[p]],[0,height[p]],ls=":",lw=.8,color=NAVY,alpha=.4,zorder=2)
        ax.set(xlim=(4,68),ylim=(-2,8),zlim=(0,120),xticks=[12,36,60],
               yticks=[0,6],zticks=[0,50,100])
        ax.set_xlabel("Retention time (scans)", fontsize=10, labelpad=5)
        ax.set_ylabel("m/z", fontsize=10, labelpad=5)
        ax.set_yticklabels(["400.000", "400.006"], fontsize=9)
        ax.set_zlabel("Effective intensity", fontsize=10, labelpad=5)
        ax.tick_params(labelsize=9, pad=0)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis._axinfo["grid"]["color"] = to_rgba(LINE,.45)
        fig.text(.045+j*.32,.77,titles[j],fontsize=16,weight="bold")
        fig.text(.045+j*.32,.215,captions[j],fontsize=12,color=MUTED,linespacing=1.5)
    # Direct labels are projected from actual data positions, not manually guessed in 2D.
    def label3d(ax,text,xyz,where,color=NAVY):
        xx,yy,_=proj3d.proj_transform(*xyz,ax.get_proj())
        ax.annotate(text,xy=(xx,yy),xytext=where,textcoords="axes fraction",
                    fontsize=12,weight="bold",color=color,ha="center",va="center",
                    bbox=dict(boxstyle="round,pad=.25",fc="white",ec="none",alpha=.94),
                    arrowprops=dict(arrowstyle="->",color=color,lw=1.4),zorder=20)
    p=g["survivors"][0]
    label3d(axes[2],"APEX",(s[p],mz[p],height[p]),(.16,.92))
    target=np.flatnonzero((g["lab"]==0)&visible)
    p=target[np.argmin((s[target]-16)**2+((mz[target]+.6)*3)**2)]
    label3d(axes[2],"BASIN",(s[p],mz[p],height[p]),(.07,.43),TEAL)
    fig.text(.50,.169,"Same colour = one basin     •     ★ = apex     •     Very low-intensity points omitted for clarity",
             fontsize=11,color=MUTED,ha="center")
    METADATA["cloud"] = {"initial_basins": len(g["roots"]), "surviving_basins": g["n"],
                         "points": cloud.n_points,"visible_points":int(visible.sum()),
                         "display_height":"effective intensity", "synthetic_split_pieces_per_mass_per_scan":15,
                         "footprints":"Convex hulls of visible-point projections, not fitted peak bounds"}
    cloud.save_npz(OUT / "synthetic_centroids.npz")
    save(fig, "01_centroids_to_basins")


def figure_links():
    y1 = np.array([4,9,19,36,60,88,68,43,23,11,4.], dtype=float)
    y2 = np.array([3,7,15,29,48,63,80,56,29,12,4.], dtype=float)
    masses, intens = [], []
    for s in range(11):
        if s == 5:
            masses.append([400,400.0015,400.012]); intens.append([y1[s],22,y2[s]])
        else:
            masses.append([400+.00012*np.sin(s),400.012+.00012*np.cos(s)])
            intens.append([y1[s],y2[s]])
    cloud = from_scans("link_example", np.arange(11)/60, masses, intens)
    g = graph(cloud, params())
    assert len(g["roots"]) == 2
    p_split = int(cloud.off[5]+1)
    p_main = int(cloud.off[5])
    assert g["parent"][p_split] == p_main and g["eff"][p_main] == 110
    fig = canvas(2, "Each centroid follows a path to one apex",
                 "Arrows show the actual parent links. This is a top-down view of the point cloud.",
                 "A basin is the set of centroids whose uphill paths end at the same apex.")
    ax = fig.add_axes([.075,.30,.56,.43])
    s = cloud.scan_index()
    yy = (cloud.mz-400)*1000
    colors = np.where(cloud.mz < 400.006, TEAL, PURPLE)
    sizes = 140+cloud.inten*4.5
    for p,q in enumerate(g["parent"]):
        if p != q:
            ax.add_patch(FancyArrowPatch((s[p],yy[p]),(s[q],yy[q]), arrowstyle="-|>",
                                        mutation_scale=14, lw=1.7, color=colors[p],
                                        shrinkA=10, shrinkB=12, zorder=1))
    ax.scatter(s,yy,s=sizes,c=colors,edgecolors="white",linewidths=1.4,zorder=3)
    for p in range(cloud.n_points):
        if p != p_split:
            ax.text(s[p],yy[p],str(int(cloud.inten[p])),ha="center",va="center",
                    color="white",fontsize=9,weight="bold",zorder=4)
    ax.annotate("split piece: 22", (5,1.5), (6.4,4.4), fontsize=11, color=TEAL,
                arrowprops=dict(arrowstyle="-",color=TEAL,connectionstyle="angle3"))
    for p in g["roots"]:
        ax.scatter(s[p], yy[p], s=sizes[p]+180, facecolors="none", edgecolors=NAVY, lw=1.7,zorder=2)
        dy = -2.8 if yy[p] < 6 else 2.4
        ax.text(s[p],yy[p]+dy,"apex",ha="center",fontsize=12,weight="bold")
    ax.set(xlim=(-.6,10.6),ylim=(-4.7,17),xticks=range(11),yticks=[0,12])
    ax.set_yticklabels(["400.000","400.012"])
    neat(ax,ylabel="m/z")
    ax.grid(False)
    fig.text(.075,.225,"Circle size and number = raw centroid intensity (a.u.).   Outlined circles = roots.",
             fontsize=11,color=MUTED)
    items = [
        ("01  Find eligible neighbours", "Use the pair's m/z tolerance in nearby\nscans; link same-scan split centroids too."),
        ("02  Choose the highest neighbour", "Move to a higher effective intensity.\nIf none is higher, this point is an apex."),
        ("03  Follow the arrows", "All points reaching the same root\nbelong to one initial basin."),
    ]
    for i,(h,t) in enumerate(items):
        y = .72-i*.155
        fig.text(.68,y,h,fontsize=15,weight="bold")
        fig.text(.68,y-.058,t,fontsize=12,color=MUTED,linespacing=1.5)
    fig.text(.68,.205,"Split example: 88 + 22 = 110\neffective height; the stronger piece is the root.",
             fontsize=11,color=TEAL,linespacing=1.5)
    METADATA["link_example"] = {"initial_basins":len(g["roots"]),"split_effective_height":110}
    save(fig,"02_uphill_links")


def figure_persistence():
    x = np.arange(51)
    fig = canvas(3, "Should two neighbouring basins be one peak?",
                 "Persistence measures how far the lower apex rises above the connecting valley (saddle).",
                 "In these examples, a shallow dip merges; a deep, noise-resolved valley keeps two peaks.",
                 "Synthetic worked examples • default fractional merge rule • spacing and noise also affect the decision")
    meta = []
    for j,saddle in enumerate([60,15]):
        y = np.interp(x,[0,18,25,32,50],[1,100,saddle,70,1])
        cloud = cloud_from_traces([y],[400])
        g = graph(cloud,params())
        assert len(g["roots"]) == 2 and g["n"] == j+1
        ax = fig.add_axes([.075+j*.48,.35,.39,.38])
        for k in np.unique(g["initial"]):
            sel = g["initial"]==k
            ax.fill_between(x,y,where=sel,color=PALETTE[k],alpha=.17,interpolate=True)
        ax.plot(x,y,color=NAVY,lw=2.4)
        ax.scatter([18,32],[100,70],color=[TEAL,ORANGE],s=80,zorder=4)
        ax.scatter([25],[saddle],color=NAVY,s=36,zorder=4)
        ax.hlines([70,saddle],25,40,linestyles="dotted",colors=MUTED,lw=1)
        ax.annotate("",(39,70),(39,saddle),arrowprops=dict(arrowstyle="<->",color=ORANGE,lw=2))
        ax.text(40.2,(70+saddle)/2,f"Δ = {70-saddle}",va="center",fontsize=12,color=ORANGE,weight="bold")
        ax.text(32,77,"lower apex = 70",ha="center",fontsize=11,color=MUTED)
        ax.annotate(f"saddle = {saddle}",(25,saddle),(7,35 if j==0 else 48),fontsize=11,color=MUTED,
                    arrowprops=dict(arrowstyle="-",color=MUTED))
        ax.set(xlim=(0,51),ylim=(0,117),xticks=[0,10,20,30,40,50],yticks=[0,50,100])
        neat(ax)
        fig.text(.075+j*.48,.772,"Shallow dip" if j==0 else "Deep valley",fontsize=18,weight="bold")
        color = TEAL if j==0 else PURPLE
        fig.text(.075+j*.48,.257,"MERGE → one candidate" if j==0 else "RETAIN → two candidates",
                 fontsize=19,color=color,weight="bold")
        fig.text(.075+j*.48,.208,f"Δ = {70-saddle} {'<' if j==0 else '≥'} threshold 35",fontsize=14,color=MUTED)
        meta.append({"saddle":saddle,"persistence":70-saddle,"threshold":35,"basins_after":g["n"]})
    fig.text(.50,.175,"Here: threshold = max(0.5 × lower apex, 3 × local noise) = 35",ha="center",fontsize=11,color=MUTED)
    METADATA["persistence_examples"] = meta
    save(fig,"03_persistence_merging")


def figure_validation():
    s = np.arange(161)
    traces = [3+90*np.exp(-.5*((s-80)/8)**2), 8+.42*s, 20+1.2*np.cos((s-80)*.22)]
    cloud = cloud_from_traces(traces,[400,450,500])
    df = pd.DataFrame({"mz":[400.,450.,500.],"sig_apex":[.5]*3,"scan_apex":[80]*3,
                       "scan_lo":[70]*3,"scan_hi":[90]*3,"noise":[3.]*3,
                       "height":[float(y[80]) for y in traces]})
    P = params()
    E = eic_check(df,cloud,P)
    assert list(E.eic_class) == ["ok","noapex","lowsnr"], E.to_dict()
    fig = canvas(4,"A 3D candidate must also look like a chromatographic peak",
                 "After basin-level gates, inspect the extracted ion chromatogram (EIC) around each candidate.",
                 "3D basins propose candidates; chromatogram checks help reject fragments and noise.",
                 "Synthetic candidates passed to the actual EIC check • these are check outcomes, not final feature calls")
    labels = ["Localised peak", "Fragment of a rising signal", "Weak ripple on background"]
    outcomes = ["PASS chromatogram check", "REJECT: no local apex", "REJECT: too little prominence"]
    notes = ["A maximum lies inside the bounds\nand rises clearly above the noise.",
             "The signal keeps rising through\nthe candidate's entire interval.",
             "A maximum exists, but its prominence\nis below 3 × the noise reference."]
    kern = K.gaussian_kernel(P.eic_smooth_fwhm*P.fwhm_scans)
    for j,y in enumerate(traces):
        color = TEAL if j==0 else RED
        left = .065+j*.318
        ax = fig.add_axes([left,.405,.265,.315])
        smooth = np.convolve(y,kern,mode="same")
        ax.axvspan(70,90,color=TEAL if j==0 else ORANGE,alpha=.12,zorder=0)
        ax.plot(s,y,color="#A8B8C8",lw=1.4,label="Raw EIC")
        ax.plot(s,smooth,color=NAVY,lw=2.4,label="Smoothed EIC")
        ax.axvline(70,color=LINE,ls="--"); ax.axvline(90,color=LINE,ls="--")
        ax.set(xlim=(45,115),ylim=(0,105),xticks=[50,70,90,110],yticks=[0,50,100])
        neat(ax,ylabel="Intensity (a.u.)" if j==0 else "")
        if j==0:
            ax.legend(frameon=False,fontsize=10,loc="upper left")
        fig.text(left,.762,labels[j],fontsize=16,weight="bold")
        fig.text(left,.315,outcomes[j],fontsize=14,weight="bold",color=color)
        fig.text(left,.251,notes[j],fontsize=12,color=MUTED,linespacing=1.5)
    fig.text(.065,.183,"Shaded interval = candidate bounds. Other checks cover scan support, width, m/z consistency, background and score.",
             fontsize=11,color=MUTED)
    METADATA["eic_check"] = E.to_dict(orient="records")
    save(fig,"04_chromatogram_validation")


def main():
    figure_what_is_3d()
    figure_comparison()
    figure_cloud()
    figure_links()
    figure_persistence()
    figure_validation()
    with PdfPages(OUT / "peak3d_explained.pdf") as pdf:
        for _,fig in FIGS:
            pdf.savefig(fig)
    with PdfPages(OUT / "peak3d_3d_and_comparison.pdf") as pdf:
        for _,fig in FIGS[:2]:
            pdf.savefig(fig)
    # Contact sheet for all six slides, plus a larger preview of the two opening figures.
    fig,axs = plt.subplots(3,2,figsize=(16,13.5),layout="constrained")
    for ax,(stem,_) in zip(axs.flat,FIGS):
        ax.imshow(plt.imread(OUT/f"{stem}.png")); ax.axis("off")
    fig.savefig(OUT/"preview.png",dpi=115)
    fig,axs = plt.subplots(2,1,figsize=(16,18),layout="constrained")
    for ax,(stem,_) in zip(axs.flat,FIGS[:2]):
        ax.imshow(plt.imread(OUT/f"{stem}.png")); ax.axis("off")
    fig.savefig(OUT/"comparison_preview.png",dpi=100)
    METADATA["source_sha256"] = {
        name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ["peak3d/kernels.py","peak3d/pick.py","peak3d/estimate.py","peak3d/features.py", "presentation/peak3d/render_figures.py"]}
    METADATA["scope"] = "Synthetic kernel demonstrations; not benchmark results or a full end-to-end run."
    (OUT/"verification.json").write_text(json.dumps(METADATA,indent=2)+"\n")
    bundle_files = ["README.md", "render_figures.py", "render.sbatch", "verification.json",
                    "synthetic_centroids.npz", "synthetic_3d_graph.npz", "peak3d_explained.pdf", "preview.png",
                    "peak3d_3d_and_comparison.pdf", "comparison_preview.png"]
    bundle_files += [f"{stem}.{ext}" for stem,_ in FIGS for ext in ("png","svg","pdf")]
    with zipfile.ZipFile(OUT/"Peak3D_figures.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in bundle_files:
            bundle.write(OUT/name, arcname=f"peak3d_figures/{name}")
    print(json.dumps(METADATA,indent=2))
    plt.close("all")


if __name__ == "__main__":
    main()
