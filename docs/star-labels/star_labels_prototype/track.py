import numpy as np, pickle
res=pickle.load(open('heads.pkl','rb'))
ks=sorted(res)
# estimate velocity field
vs=[]
for a,b in zip(ks[50:200],ks[51:201]):
    A=res[a];B=res[b]
    for p in A:
        d=B[:,:2]-p[:2]; r=np.hypot(*d.T)
        if r.min()<15: vs.append(d[r.argmin()])
vs=np.array(vs); print(np.median(vs,axis=0), len(vs))
v0=np.median(vs,axis=0)
# link
tracks=[]; active=[]
for k in ks:
    D=res[k]; used=set()
    new=[]
    for tr in active:
        last=tr[-1]; 
        vel=(np.array(tr[-1][1:3])-np.array(tr[-2][1:3]))/(tr[-1][0]-tr[-2][0])*5 if len(tr)>1 else v0
        pred=np.array(last[1:3])+vel*(k-last[0])/5
        if len(D)==0: 
            if k-last[0]<=20: new.append(tr)
            else: tracks.append(tr)
            continue
        r=np.hypot(*(D[:,:2]-pred).T); r[list(used)]=1e9
        j=r.argmin()
        if r[j]<6: tr.append((k,*D[j])); used.add(j); new.append(tr)
        elif k-last[0]<=20: new.append(tr)
        else: tracks.append(tr)
    for j in range(len(D)):
        if j not in used: new.append([(k,*D[j])])
    active=new
tracks+=active
tracks=[np.array(t) for t in tracks if len(t)>=8]
# keep tracks moving consistently
good=[]
for t in tracks:
    disp=t[-1,1:3]-t[0,1:3]; dt=t[-1,0]-t[0,0]
    v=disp/dt*5
    if np.dot(v,v0)/np.dot(v0,v0)>0.6 and np.hypot(*disp)>40: good.append(t)
pickle.dump(good,open('tracks.pkl','wb'))
print(len(tracks),len(good))
for t in sorted(good,key=lambda t:-t[:,3].max())[:30]:
    print(int(t[0,0]),int(t[-1,0]),t[0,1:3].round(),t[-1,1:3].round(),t[:,3].max())
