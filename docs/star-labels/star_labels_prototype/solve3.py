import numpy as np, pickle
from scipy.optimize import least_squares
from cat import load
import model
from model import project
tracks=pickle.load(open('tracks.pkl','rb'))
p=pickle.load(open('p2.pkl','rb'))
C=load(6.0)
RAc=np.array([c['ra'] for c in C]); DEc=np.array([c['dec'] for c in C]); MAG=np.array([c['mag'] for c in C])
def match(p):
    M=[]
    for ti,t in enumerate(tracks):
        s=max(1,len(t)//6); tt=t[::s,0]; xx=t[::s,1]; yy=t[::s,2]
        px,py,Z,_=project(p,RAc[:,None],DEc[:,None],tt[None,:])
        d=np.hypot(px-xx,py-yy).mean(1); d[(Z<=0).any(1)]=1e9
        tol=np.where(MAG<4,10,4.5)
        ok=d<tol
        if not ok.any(): continue
        j=np.argmin(np.where(ok,d+MAG*0.3,1e9))
        others=np.delete(d,j)
        if d[j]<4.5 and np.sort(others)[0]<1.5*d[j] and MAG[j]>=4: continue
        M.append((ti,j,d[j]))
    return M
def resid(q,T,X,Y,RA,DE):
    px,py,Z,_=project(q,RA,DE,T)
    return np.concatenate([px-X,py-Y])
for it in range(3):
    M=match(p)
    T=[];X=[];Y=[];RA=[];DE=[]
    for ti,ci,_ in M:
        t=tracks[ti]; T+=list(t[:,0]);X+=list(t[:,1]);Y+=list(t[:,2]);RA+=[RAc[ci]]*len(t);DE+=[DEc[ci]]*len(t)
    T,X,Y,RA,DE=map(np.array,(T,X,Y,RA,DE))
    r=least_squares(resid,p,args=(T,X,Y,RA,DE),loss='soft_l1',f_scale=2)
    p=r.x
    res=r.fun.reshape(2,-1); rr=np.hypot(*res)
    print(it,len(M),len(set(m[1] for m in M)),'median',np.median(rr).round(2),'p90',np.percentile(rr,90).round(2))
print(p)
pickle.dump(p,open('p3.pkl','wb')); pickle.dump(M,open('matches3.pkl','wb'))
# residual by x region
for x0 in range(0,1920,320):
    s=(X>=x0)&(X<x0+320); print(x0, s.sum(), np.median(rr[s]).round(2) if s.any() else None)
