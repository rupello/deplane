import numpy as np, pickle
from scipy.optimize import least_squares
from cat import load
from model import project
tracks=pickle.load(open('tracks.pkl','rb'))
def snap(n):
    P=[]
    for i,t in enumerate(tracks):
        if t[0,0]<=n<=t[-1,0]:
            x=np.interp(n,t[:,0],t[:,1]); y=np.interp(n,t[:,0],t[:,2])
            P.append((x,y,np.median(t[:,3])*np.median(t[:,4]),i))
    return np.array(P)
P=snap(600); P=P[np.argsort(-P[:,2])]
C=load(6.5); byname={c['name']:c for c in C if c['name']}
ids={'Betelgeuse':int(P[0,3]),'Rigel':int(P[1,3]),'Bellatrix':int(P[2,3]),'Alnilam':int(P[3,3])}
def data(m):
    T=[];X=[];Y=[];RA=[];DE=[]
    for nm,ti in m.items():
        t=tracks[ti]; c=byname[nm] if isinstance(nm,str) else nm
        T+=list(t[:,0]);X+=list(t[:,1]);Y+=list(t[:,2]);RA+=[c['ra']]*len(t);DE+=[c['dec']]*len(t)
    return [np.array(a) for a in (T,X,Y,RA,DE)]
T,X,Y,RA,DE=data(ids)
def resid(p,T,X,Y,RA,DE,free_dist=True):
    q=p.copy()
    if not free_dist: q[8:]=0
    px,py,Z,_=project(q,RA,DE,T)
    r=np.concatenate([px-X,py-Y]); r[np.concatenate([Z,Z])<=0]=1e4
    return r
best=None
for th0 in range(0,360,15):
  for az0 in [70,90,110,130]:
    for alt0 in [15,30]:
        p0=np.array([th0,0.02,az0,alt0,0,2200,960,540,0,0.])
        lo=[-720,0.001,0,-10,-60,800,700,300,-1,-1];hi=[720,0.2,360,80,60,6000,1220,780,1,1]
        try:
            r=least_squares(lambda p:resid(p,T,X,Y,RA,DE,False),p0,bounds=(lo,hi),loss='soft_l1',f_scale=5)
        except Exception as e: continue
        rms=np.sqrt(np.mean(r.fun**2))
        if best is None or rms<best[0]: best=(rms,r.x)
print(best[0],best[1].round(4))
pickle.dump(best[1],open('p1.pkl','wb'))
