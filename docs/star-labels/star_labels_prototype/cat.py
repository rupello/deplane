import json, numpy as np
import os
# 'data' folder of the d3-celestial npm package (npm pack d3-celestial@0.7.35 && tar xzf d3-celestial-0.7.35.tgz)
D=os.environ.get('D3C_DATA','package/data/')
s=json.load(open(D+'stars.6.json'))['features']
names=json.load(open(D+'starnames.json'))
GREEK={'α':'alpha','β':'beta','γ':'gamma','δ':'delta','ε':'epsilon','ζ':'zeta','η':'eta','θ':'theta','ι':'iota','κ':'kappa','λ':'lambda','μ':'mu','ν':'nu','ξ':'xi','ο':'omicron','π':'pi','ρ':'rho','σ':'sigma','τ':'tau','υ':'upsilon','φ':'phi','χ':'chi','ψ':'psi','ω':'omega'}
def load(maxmag=6):
    out=[]
    for f in s:
        m=f['properties']['mag']
        if m>maxmag: continue
        ra,dec=f['geometry']['coordinates']
        nm=names.get(str(f['id']),{})
        lab=nm.get('name','') 
        des=''
        if nm.get('bayer'): des=nm['bayer']+' '+nm.get('c','')
        elif nm.get('flam'): des=nm['flam']+' '+nm.get('c','')
        out.append(dict(hip=f['id'],ra=ra%360,dec=dec,mag=m,bv=f['properties'].get('bv'),name=lab,des=des))
    return out
