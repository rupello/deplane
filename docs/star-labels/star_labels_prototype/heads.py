import numpy as np, cv2, pickle
tm=np.load('tm.npy'); mx=np.load('mx.npy')
H=4
res={}
for n in range(H,1440-H,5):
    m=((tm>=n-H)&(tm<=n+H)).astype(np.uint8)
    m=cv2.dilate(m,np.ones((3,3),np.uint8))
    cnt,lab,st,cen=cv2.connectedComponentsWithStats(m)
    out=[]
    for i in range(1,cnt):
        if st[i,4]<12: continue
        ys,xs=np.nonzero(lab[st[i,1]:st[i,1]+st[i,3],st[i,0]:st[i,0]+st[i,2]]==i)
        ys=ys+st[i,1]; xs=xs+st[i,0]
        b=mx[ys,xs].astype(float)
        out.append((xs.mean(),ys.mean(),b.max(),len(xs)))
    res[n]=np.array(out)
pickle.dump(res,open('heads.pkl','wb'))
print({k:len(v) for k,v in list(res.items())[::24]})
