"""Render a slide-ready schematic using synthetic LC-MS signals."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import Bbox

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})
navy, gray = '#18334B', '#657888'
colors = ['#128C91', '#D77A26', '#7962B3']
times = np.linspace(0, 6, 301)
masses = [180.065, 255.233, 310.202]
centers = [1.6, 3.1, 4.6]
widths = [.18, .25, .21]
heights = [1.0, .78, .9]
rng = np.random.default_rng(12)
signals = [h*np.exp(-.5*((times-t)/w)**2) for h,t,w in zip(heights,centers,widths)]

fig = plt.figure(figsize=(16, 6.2), facecolor='white')
fig.text(.045, .925, 'From raw LC–MS data to measurable features',
         fontsize=25, weight='bold', color=navy)
positions = [(.055, .285, .255, .40), (.39, .285, .255, .40)]
for x, n, heading in [(.055, '01', 'Raw LC–MS signals'),
                       (.39, '02', 'Detect & integrate peaks'),
                       (.72, '03', 'Feature table')]:
    fig.text(x, .79, n, fontsize=13, weight='bold', color=colors[0])
    fig.text(x+.027, .79, heading, fontsize=17, weight='bold', color=navy)

ax = fig.add_axes(positions[0])
bg_t = rng.uniform(0, 6, 190)
bg_m = rng.uniform(140, 345, 190)
ax.scatter(bg_t, bg_m, s=4, color='#D4DEE4', alpha=.7, linewidths=0)
for i, (mass, sig, color) in enumerate(zip(masses, signals, colors)):
    mask = sig > .018
    tx = times[mask]
    y = mass + rng.normal(0, .45, len(tx))
    ax.scatter(tx, y, s=8+sig[mask]*80, c=color, alpha=.78, linewidths=0)
    ax.text(centers[i], mass+14, f'F{i+1}', ha='center', color=color, weight='bold')
ax.set(xlim=(0, 6), ylim=(130, 350), xlabel='Retention time (min)', ylabel='m/z')
ax.set_xticks([0,2,4,6]); ax.set_yticks([150,200,250,300,350])
for spine in ['top','right']: ax.spines[spine].set_visible(False)
ax.tick_params(colors=gray)
ax.xaxis.label.set_color(navy); ax.yaxis.label.set_color(navy)

ax2 = fig.add_axes(positions[1])
for i, (sig, color, center, width, mass) in enumerate(zip(signals, colors, centers, widths, masses)):
    offset = (2-i)*1.35
    observed = np.maximum(0, sig+rng.normal(.018,.014,len(times)))
    mask = abs(times-center) <= 3*width
    ax2.plot(times, observed+offset, color=color, lw=1.55)
    ax2.fill_between(times, offset, observed+offset, where=mask, color=color, alpha=.22)
    for bound in [center-3*width, center+3*width]:
        ax2.plot([bound,bound], [offset,offset+.5], color=color, lw=1, ls=':')
    ax2.plot(center, sig.max()+offset+.018, 'o', color=color, ms=5)
    ax2.text(.06, offset+.65, f'F{i+1}  |  m/z {mass:.3f}', fontsize=10, color=color)
ax2.set(xlim=(0,6), ylim=(-.12,4.05), xlabel='Retention time (min)',
        ylabel='Intensity (offset for clarity)')
ax2.set_xticks([0,2,4,6]); ax2.set_yticks([])
for spine in ['top','right']: ax2.spines[spine].set_visible(False)
ax2.tick_params(colors=gray)
ax2.xaxis.label.set_color(navy); ax2.yaxis.label.set_color(navy)

axt = fig.add_axes([.715,.345,.26,.30]); axt.axis('off')
areas = [np.trapezoid(s[abs(times-c)<=3*w], times[abs(times-c)<=3*w])
         if hasattr(np,'trapezoid') else np.trapz(s[abs(times-c)<=3*w], times[abs(times-c)<=3*w])
         for s,c,w in zip(signals,centers,widths)]
rows = [[f'F{i+1}', f'{m:.3f}', f'{c:.2f}', f'{a:.3f}']
        for i,(m,c,a) in enumerate(zip(masses,centers,areas))]
table = axt.table(cellText=rows, colLabels=['Feature','m/z','RT (min)','Area*'],
                  cellLoc='center', bbox=[0,0,1,1], colWidths=[.21,.30,.27,.22])
table.auto_set_font_size(False); table.set_fontsize(11)
for (r,c), cell in table.get_celld().items():
    cell.set_edgecolor('white'); cell.set_linewidth(3)
    cell.set_facecolor('#EEF3F6' if r else navy)
    cell.get_text().set_color(navy if r else 'white')
    if not r: cell.get_text().set_weight('bold')
    if r and c==0:
        cell.get_text().set_color(colors[r-1]); cell.get_text().set_weight('bold')
fig.text(.72,.30,'*Illustrative integrated signal (a.u.·min)', fontsize=9, color=gray)
for x in [.34,.674]:
    fig.add_artist(FancyArrowPatch((x-.012,.50),(x+.015,.50), transform=fig.transFigure,
                                  arrowstyle='-|>', mutation_scale=22, lw=1.8, color=gray))
for x, text in [(.055,'Signals across m/z and retention time'),
                (.39,'Locate peak boundaries and quantify area'),
                (.72,'Summarize each detected feature')]:
    fig.text(x,.16,text,fontsize=12,color=navy)
fig.text(.055,.055,'Illustrative data • One sample • A feature does not necessarily represent a unique metabolite',
         fontsize=10, color=gray)
if '--panels-only' not in sys.argv:
    for ext in ['svg','pdf','png']:
        fig.savefig(OUT/f'feature_detection_workflow.{ext}', dpi=220, facecolor='white')

# Export each panel from the original vector artists, preserving every data point.
# Hide neighboring artists so each export is a genuinely independent figure.
panel_specs = [
    ('01_raw_lcms_signals', ax, .005, '01', 'Raw LC–MS signals',
     'Signals across m/z and retention time'),
    ('02_peak_detection', ax2, .345, '02', 'Detect & integrate peaks',
     'Locate peak boundaries and quantify area'),
    ('03_feature_table', axt, .69, '03', 'Feature table',
     'Summarize each detected feature'),
]
for artist in fig.artists:
    artist.set_visible(False)
for name, selected_ax, left, number, heading, caption in panel_specs:
    for panel_ax in fig.axes:
        panel_ax.set_visible(panel_ax is selected_ax)
    keep = {number, heading, caption}
    if selected_ax is axt:
        keep.add('*Illustrative integrated signal (a.u.·min)')
    for label in fig.texts:
        label.set_visible(label.get_text() in keep)
    footer = fig.text(left+.035, .075, 'Illustrative data • One sample',
                      fontsize=9, color=gray)
    bounds = Bbox.from_extents(left*16, .04*6.2, (left+.325)*16, .86*6.2)
    for ext in ['svg', 'pdf', 'png']:
        fig.savefig(OUT/f'{name}.{ext}', dpi=300, facecolor='white', bbox_inches=bounds)
    footer.remove()
plt.close(fig)
print('Created separate workflow panels as SVG, PDF, and PNG in', OUT)
