import numpy as np, pickle, itertools
from cat import load
tracks=pickle.load(open('tracks.pkl','rb'))
def snap(n):
    P=[]
    for i,t in enumerate(tracks):
        if t[0,0]<=n<=t[-1,0]:
            x=np.interp(n,t[:,0],t[:,1]); y=np.interp(n,t[:,0],t[:,2])
            P.append((x,y,np.median(t[:,3])*np.median(t[:,4]),i))
    return np.array(P)
N=600
P=snap(N); P=P[np.argsort(-P[:,2])]
print(len(P)); print(P[:15].round(1))
C=load(3.0)
ra=np.radians([c['ra'] for c in C]); de=np.radians([c['dec'] for c in C])
V=np.stack([np.cos(de)*np.cos(ra),np.cos(de)*np.sin(ra),np.sin(de)],1)
ang=np.degrees(np.arccos(np.clip(V@V.T,-1,1)))
def gnom(v,c):
    # tangent plane at unit vector c
    z=np.array([0,0,1.]); e=np.cross(z,c); e/=np.linalg.norm(e); nn=np.cross(c,e)
    w=v@c
    return np.stack([(v@e)/w,(v@nn)/w],1)
Dp=P[:,:2]
best=[]
top=P[:10,:2]
# pixel triangles
for i,j,k in itertools.combinations(range(10),3):
    pts=top[[i,j,k]]
    d=[np.hypot(*(pts[a]-pts[b])) for a,b in [(0,1),(1,2),(0,2)]]
    if min(d)<40: continue
    best.append((i,j,k))
print(len(best),'pixel tris')
# catalog: for each pixel tri, find catalog triples with matching ratios -> brute force via scale
res=[]
nC=len(C)
ang_r=ang.copy()
for (i,j,k) in best:
    pi=top[[i,j,k]]
    dij=np.hypot(*(pi[0]-pi[1])); djk=np.hypot(*(pi[1]-pi[2])); dik=np.hypot(*(pi[0]-pi[2]))
    for s in [26,29,32,35,38,41,44,48,52,57]: # px per degree guesses
        tol=0.08
        A=np.argwhere(np.abs(ang_r-dij/s)<tol*dij/s)
        for a,b in A:
            if a==b: continue
            cands=np.nonzero((np.abs(ang_r[b]-djk/s)<tol*djk/s)&(np.abs(ang_r[a]-dik/s)<tol*dik/s))[0]
            for c in cands:
                if c in (a,b): continue
                res.append((i,j,k,a,b,c,s))
print(len(res),'candidates')
# verify each: similarity on gnomonic plane
from numpy.linalg import lstsq
scores=[]
seen=set()
for (i,j,k,a,b,c,s) in res:
    key=(i,j,k,a,b,c)
    if key in seen: continue
    seen.add(key)
    cen=V[[a,b,c]].mean(0); cen/=np.linalg.norm(cen)
    near=np.nonzero(V@cen>np.cos(np.radians(55)))[0]
    g=gnom(V[near],cen)
    gt=gnom(V[[a,b,c]],cen); pt=top[[i,j,k]]
    for refl in [1,-1]:
        g2=gt*[1,refl]
        # similarity: [x y]=[[p -q],[q p]]g + t
        M=[];R=[]
        for (u,v),(x,y) in zip(g2,pt):
            M+= [[u,-v,1,0],[v,u,0,1]]; R+=[x,y]
        sol=lstsq(np.array(M),np.array(R),rcond=None)[0]
        p,q,tx,ty=sol
        resid=np.array(M)@sol-np.array(R)
        if np.abs(resid).max()>15: continue
        gg=g*[1,refl]
        X=np.stack([p*gg[:,0]-q*gg[:,1]+tx,q*gg[:,0]+p*gg[:,1]+ty],1)
        inb=(X[:,0]>0)&(X[:,0]<1920)&(X[:,1]>0)&(X[:,1]<1080)
        dd=np.hypot(X[inb,None,0]-Dp[None,:,0],X[inb,None,1]-Dp[None,:,1])
        nin=(dd.min(0)<20).sum()
        scores.append((nin,refl,(i,j,k,a,b,c),(p,q,tx,ty),cen))
scores.sort(key=lambda x:-x[0])
for sc in scores[:10]:
    i,j,k,a,b,c=sc[2]
    print(sc[0],sc[1],[C[a]['name'] or C[a]['des'],C[b]['name'] or C[b]['des'],C[c]['name'] or C[c]['des']],(i,j,k))
pickle.dump((scores[:20],N),open('solve0.pkl','wb'))
