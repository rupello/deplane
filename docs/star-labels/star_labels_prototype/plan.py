import numpy as np, pickle
from model import project
from PIL import ImageFont
p=pickle.load(open('p3.pkl','rb'))
V=pickle.load(open('vis.pkl','rb'))
tracks=pickle.load(open('tracks.pkl','rb')); M=pickle.load(open('matches3.pkl','rb'))
IAU={'Sirius','Rigel','Procyon','Betelgeuse','Bellatrix','Alnilam','Alnitak','Mirzam','Saiph','Mintaka','Arneb','Hatysa','Cursa','Nihal','Gomeisa','Zaurak','Tabit','Alzirr','Meissa','Rana','Tarf','Angetenar','Ran','Sceptrum','Azha','Beid','Keid','Zibal'}
SUP=str.maketrans('0123456789','⁰¹²³⁴⁵⁶⁷⁸⁹')
CONST={'Ori':'ORION','CMa':'CANIS MAJOR','CMi':'CANIS MINOR','Lep':'LEPUS','Eri':'ERIDANUS','Mon':'MONOCEROS','Tau':'TAURUS','Cet':'CETUS','Gem':'GEMINI','Cnc':'CANCER'}
FPS=30; NF=1440
# per-star offset correction from matched tracks
from cat import load
C=load(6.0)
off={}
for ti,ci,_ in M:
    t=tracks[ti]
    px,py,_,_=project(p,np.full(len(t),C[ci]['ra']),np.full(len(t),C[ci]['dec']),t[:,0])
    off.setdefault(ci,[]).append(np.stack([t[:,1]-px,t[:,2]-py],1))
off={k:np.median(np.concatenate(v),0) for k,v in off.items()}
# select
sel=[o for o in V if o['des'] and ((o['mag']<=4.75 and o['frac']>=0.85 and o['nvis']>=40) or (o['mag']<2.5 and o['frac']>=0.6) or o['name']=='Trapezium')]
sel.sort(key=lambda o:o['mag'])
F=np.arange(NF)
kept=[]
for o in sel:
    x,y,_,_=project(p,o['ra'],o['dec'],F)
    o['x']=x+off.get(o['i'],[0,0])[0]; o['y']=y+off.get(o['i'],[0,0])[1]
    dup=False
    for k in kept:
        a,b=max(o['start'],k['start']),min(o['end'],k['end'])
        if b>a and np.median(np.hypot(o['x'][a:b]-k['x'][a:b],o['y'][a:b]-k['y'][a:b]))<10: dup=True
    if not dup: kept.append(o)
for o in kept:
    d=o['des'].split(' ')
    bay=((d[0][0]+d[0][1:].translate(SUP)) if d[0][0] in 'αβγδεζηθικλμνξοπρστυφχψω' else d[0])+' '+d[1] if len(d)==2 else o['des']
    if o['name']=='Trapezium': o['label']='Orion Nebula'; o['sub']='M42'
    elif o['name'] in IAU: o['label']=o['name']; o['sub']=bay
    else: o['label']=bay; o['sub']=''
    o['con']=d[1] if len(d)==2 else ''
print(len(kept))
# scheduling
fontB=lambda s: ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-SemiBold.otf',s)
def size(m): return 30 if m<1 else 26 if m<2.3 else 22 if m<3.6 else 18
FIN,FOUT=12,45
def hold(m): return 90 if m<2.3 else 70 if m<3.6 else 55
def bbox(o,f,side):
    x,y=o['x'][f],o['y'][f]
    w=o['w']; h=o['h']
    if side==0: return (x+14,y+4,x+14+w,y+4+h)
    return (x-14-w,y-4-h,x-14,y-4)
for o in kept:
    s=size(o['mag']); fb=fontB(s); fs=fontB(int(s*0.62))
    w=fb.getlength(o['label'])+(fs.getlength('  '+o['sub']) if o['sub'] else 0)
    o['w']=w+4; o['h']=s*1.15; o['fs']=s
# start-of-video stars: stagger by brightness
early=sorted([o for o in kept if o['start']<=6],key=lambda o:o['mag'])
kept=[o for o in kept if not (o['start']<=6 and o['mag']>4.4)]
early=[o for o in early if o['mag']<=4.4]
for k,o in enumerate(early): o['start']=10+k*20
order=sorted(kept,key=lambda o:(o['start'],o['mag']))
placed=[]
def overl(A,B): return not(A[2]<B[0] or B[2]<A[0] or A[3]<B[1] or B[3]<A[1])
for o in order:
    life=FIN+hold(o['mag'])+FOUT
    ok=False
    for delay in range(0,121,10):
        s0=o['start']+delay
        if s0+life>=NF: s0=max(0,NF-1-life) if delay==0 else None
        if s0 is None: break
        if s0>o['end']-20 and delay>0: break
        for side in (0,1):
            fr=range(s0,min(NF,s0+life),6)
            bad=False
            for f in fr:
                b=bbox(o,f,side)
                if b[0]<4 or b[2]>1916 or b[1]<4 or b[3]>1076: bad=True;break
                for q in placed:
                    if q['t0']<=f<q['t0']+q['life'] and overl(b,bbox(q,f,q['side'])): bad=True;break
                if bad: break
            if not bad:
                o['t0']=s0;o['side']=side;o['life']=life;ok=True;break
        if ok: break
    if ok: placed.append(o)
    elif o['mag']<2.5:
        o['t0']=o['start'];o['side']=0;o['life']=life;placed.append(o)
print(len(placed),'placed of',len(kept))
for o in sorted(placed,key=lambda o:o['t0']): print(round(o['t0']/30,1),o['label'],o['sub'],o['mag'],o['side'])
pickle.dump(placed,open('plan.pkl','wb'))
