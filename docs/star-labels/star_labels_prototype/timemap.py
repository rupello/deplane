import cv2, numpy as np, sys
cap=cv2.VideoCapture(sys.argv[1])
T=int(sys.argv[2]) if len(sys.argv)>2 else 10
tm=None; mx=None; n=0
ker=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
while True:
    ok,f=cap.read()
    if not ok: break
    g=f.max(axis=2)
    th=cv2.morphologyEx(g,cv2.MORPH_TOPHAT,ker)
    if tm is None:
        tm=np.full(g.shape,-1,np.int16); mx=np.zeros(g.shape,np.uint8); base=th.copy()
    lit=(th.astype(int)-base>T)&(tm<0)
    tm[lit]=n
    mx=np.maximum(mx,th)
    n+=1
np.save('tm.npy',tm); np.save('mx.npy',mx)
v=np.zeros(tm.shape,np.uint8); v[tm>=0]=(50+tm[tm>=0]*200//n).astype(np.uint8)
cv2.imwrite('tm.png',cv2.applyColorMap(v,cv2.COLORMAP_TURBO)*(tm>=0)[...,None])
print(n,(tm>=0).sum())
