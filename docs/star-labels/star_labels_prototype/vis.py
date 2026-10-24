import numpy as np, pickle
from cat import load
from model import project
tm=np.load('tm.npy').astype(np.int32)
p=pickle.load(open('p3.pkl','rb'))
C=load(6.0)
RA=np.array([c['ra'] for c in C]); DE=np.array([c['dec'] for c in C]); MAG=np.array([c['mag'] for c in C])
F=np.arange(0,1440,2)
px,py,Z,alt=project(p,RA[:,None],DE[:,None],F[None,:])
inb=(Z>0)&(px>=4)&(px<1916)&(py>=4)&(py<1076)
# local tm window min |tm - n|
vis=np.zeros(px.shape,bool)
for i in range(len(C)):
    idx=np.nonzero(inb[i])[0]
    for k in idx:
        x=int(round(px[i,k]));y=int(round(py[i,k]))
        w=tm[y-3:y+4,x-3:x+4]
        n=F[k]
        vis[i,k]=((w>=n-10)&(w<=n+4)).any()
out=[]
for i in range(len(C)):
    v=vis[i].astype(float)
    if v.sum()<10: continue
    # 1-second windows (15 samples) with >=70% visible
    win=np.convolve(v,np.ones(15)/15,mode='valid')
    good=np.nonzero(win>=0.7)[0]
    if len(good)==0: continue
    a=F[good[0]]; b=F[good[-1]+14]
    frac=v[good[0]:good[-1]+15].mean()
    out.append(dict(i=i,start=int(a),end=int(b),frac=float(frac),nvis=int(v.sum()),**C[i]))
print(len(out))
pickle.dump(out,open('vis.pkl','wb'))
for o in sorted(out,key=lambda o:o['mag']):
    print(f"{o['name'] or '-':18s} {o['des']:10s} {o['mag']:5.2f} {o['start']:5d} {o['end']:5d} {o['frac']:.2f} {o['nvis']}")
