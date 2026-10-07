import sys
from pyteomics import mzml
for f in sys.argv[1:]:
    n=0; low=[]
    for s in mzml.read(f):
        if s.get('ms level')!=1: continue
        n+=1
        i=s['intensity array']
        if len(i)==0 or i.max()<50000:
            low.append(round(float(s['scanList']['scan'][0]['scan start time']),2))
    print(f.split('/')[-1], f"{len(low)}/{n} MS1 scans empty after 50k cut", "RT(min):", low[:6], "..." if len(low)>6 else "")
