import numpy as np, pickle, subprocess, sys
from PIL import Image, ImageDraw, ImageFont
from model import project
P=pickle.load(open('plan.pkl','rb'))
SRC=sys.argv[1]; OUT=sys.argv[2]
only=set(int(a) for a in sys.argv[3].split(',')) if len(sys.argv)>3 else None
W,H,NF=1920,1080,1440
FIN,FOUT=12,45
FB='/usr/share/fonts/opentype/inter/Inter-SemiBold.otf'; FR='/usr/share/fonts/opentype/inter/Inter-Regular.otf'
FM='/usr/share/fonts/opentype/inter/Inter-Medium.otf'
fc={}
def font(path,s):
    k=(path,s)
    if k not in fc: fc[k]=ImageFont.truetype(path,s)
    return fc[k]
def alpha_at(o,f):
    t=f-o['t0']
    if t<0 or t>=o['life']: return 0
    if t<FIN: return t/FIN
    if t>o['life']-FOUT: return (o['life']-t)/FOUT
    return 1.0
# constellation captions
CN={'Ori':'ORION','CMa':'CANIS MAJOR','CMi':'CANIS MINOR','Lep':'LEPUS','Eri':'ERIDANUS','Mon':'MONOCEROS','Tau':'TAURUS'}
caps=[]
for c,nm in CN.items():
    mem=[o for o in P if o['con']==c]
    if not mem: continue
    if c=='Ori':
        key=[o for o in mem if o['label'] in ('Betelgeuse','Rigel','Bellatrix','Alnilam','Alnitak','Mintaka','Saiph')]
        t0=max(o['start'] for o in key if o['label']!='Saiph')+45
    else:
        b=min(mem,key=lambda o:o['mag']); t0=b['t0']+30
    caps.append(dict(name=nm,mem=mem,t0=t0,life=30+120+60))
def cap_pos(cp,f):
    pts=[(o['x'][f],o['y'][f]) for o in cp['mem'] if o['start']<=f<=o['end'] and 0<o['x'][f]<W and 0<o['y'][f]<H]
    if not pts: return None
    a=np.array(pts); return a.mean(0)
def cap_alpha(cp,f):
    t=f-cp['t0']
    if t<0 or t>=cp['life']: return 0
    if t<30: return t/30
    if t>cp['life']-60: return (cp['life']-t)/60
    return 1.0
# caption placement avoiding labels
def lab_box(o,f):
    x,y=o['x'][f],o['y'][f]; w=o['w']; h=o['h']
    if o['side']==0: return (x-12,y-12,x+14+w,y+4+h)
    return (x-14-w,y-4-h,x+12,y+12)
def tracked_w(fnt,txt,tr=7):
    return sum(fnt.getlength(ch)+(tr*2.2 if ch==' ' else tr) for ch in txt)-tr
CF=font(FM,32)
placedcaps=[]
for cp in caps:
    f0=min(cp['t0'],NF-1); p0=cap_pos(cp,f0)
    if p0 is None: cp['ok']=False; continue
    cp['ok']=True
    ms=[o for o in cp['mem'] if o['start']<=f0<=o['end']]
    dx=np.mean([o['x']-o['x'][f0] for o in ms],0); dy=np.mean([o['y']-o['y'][f0] for o in ms],0)
    cw=tracked_w(CF,cp['name']); ch=40
    best=None
    for ox in range(-400,401,40):
        for oy in range(-320,321,40):
            sc=np.hypot(ox,oy)*0.02; bad=False
            for f in range(cp['t0'],min(NF,cp['t0']+cp['life']),8):
                cx=p0[0]+ox+dx[f]; cy=p0[1]+oy+dy[f]
                B=(cx-cw/2,cy-ch/2,cx+cw/2,cy+ch/2)
                if B[0]<8 or B[2]>W-8 or B[1]<8 or B[3]>720: bad=True;break
                for o in P:
                    if o['t0']<=f<o['t0']+o['life']:
                        L=lab_box(o,f)
                        if not(L[2]<B[0] or B[2]<L[0] or L[3]<B[1] or B[3]<L[1]): sc+=1
                for q in placedcaps:
                    if q['t0']<=f<q['t0']+q['life']:
                        Q=(q['x'][f]-q['cw']/2,q['y'][f]-20,q['x'][f]+q['cw']/2,q['y'][f]+20)
                        if not(Q[2]<B[0] or B[2]<Q[0] or Q[3]<B[1] or B[3]<Q[1]): sc+=5
            if not bad and (best is None or sc<best[0]): best=(sc,ox,oy)
    if best is None: cp['ok']=False; continue
    cp['x']=p0[0]+best[1]+dx; cp['y']=p0[1]+best[2]+dy; cp['cw']=cw
    placedcaps.append(cp); print(cp['name'],round(cp['t0']/30,1),best)
def draw_frame(img,f):
    layer=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(layer)
    for cp in caps:
        if not cp['ok']: continue
        a=cap_alpha(cp,f)
        if a<=0: continue
        x=cp['x'][f]-cp['cw']/2; y=cp['y'][f]-20
        for chh in cp['name']:
            d.text((x,y),chh,font=CF,fill=(170,200,255,int(110*a)))
            x+=CF.getlength(chh)+(7*2.2 if chh==' ' else 7)
    for o in P:
        a=alpha_at(o,f)
        if a<=0: continue
        x,y=o['x'][f],o['y'][f]
        s=o['fs']; fb=font(FB,s); fs=font(FR,int(s*0.62))
        main=(255,244,225) if o['mag']<1 else (240,240,240)
        ca=int(255*a*(1.0 if o['mag']<3.6 else 0.85))
        r=7 if o['mag']>=2.3 else 10
        d.ellipse((x-r,y-r,x+r,y+r),outline=(255,220,140,int(ca*0.8)),width=2)
        w=fb.getlength(o['label']); ws=fs.getlength('  '+o['sub']) if o['sub'] else 0
        if o['side']==0: tx,ty=x+14,y+4
        else: tx,ty=x-14-(w+ws+4),y-4-s*1.15
        d.text((tx,ty),o['label'],font=fb,fill=main+(ca,),stroke_width=2,stroke_fill=(0,0,0,int(ca*0.55)))
        if o['sub']:
            d.text((tx+w,ty+s*0.30),'  '+o['sub'],font=fs,fill=(190,205,225,int(ca*0.9)),stroke_width=2,stroke_fill=(0,0,0,int(ca*0.5)))
    return Image.alpha_composite(img.convert('RGBA'),layer).convert('RGB')
if only:
    for f in sorted(only):
        raw=subprocess.run(['ffmpeg','-v','error','-ss',str(f/30),'-i',SRC,'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True).stdout
        img=Image.frombytes('RGB',(W,H),raw); draw_frame(img,f).save(f'{OUT}_{f}.png')
    sys.exit()
dec=subprocess.Popen(['ffmpeg','-v','error','-i',SRC,'-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
enc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','30','-i','-','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',OUT],stdin=subprocess.PIPE)
f=0
while True:
    raw=dec.stdout.read(W*H*3)
    if len(raw)<W*H*3: break
    img=Image.frombytes('RGB',(W,H),raw)
    enc.stdin.write(draw_frame(img,f).tobytes()); f+=1
enc.stdin.close(); enc.wait(); print('frames',f)
