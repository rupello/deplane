import numpy as np
LAT=np.radians(39.05)
def radec2v(ra,dec):
    ra=np.radians(ra);dec=np.radians(dec)
    return ra,dec
def refr(alt):  # alt radians -> apparent alt (Bennett/Saemundsson)
    a=np.degrees(alt)
    R=1.02/np.tan(np.radians(a+10.3/(a+5.11)))/60.0
    R=np.where(a>-1.5,R,0)
    return alt+np.radians(R)
def project(p,ra,dec,t,use_refr=True):
    th0,om,az0,alt0,roll,f,cx,cy,k1,k2=p
    ra=np.radians(ra);dec=np.radians(dec)
    H=np.radians(th0)+np.radians(om)*t-ra
    sa=np.sin(dec)*np.sin(LAT)+np.cos(dec)*np.cos(LAT)*np.cos(H)
    alt=np.arcsin(np.clip(sa,-1,1))
    az=np.arctan2(-np.sin(H)*np.cos(dec), np.sin(dec)*np.cos(LAT)-np.cos(dec)*np.sin(LAT)*np.cos(H))
    if use_refr: alt=refr(alt)
    # ENU vector
    E=np.cos(alt)*np.sin(az); N=np.cos(alt)*np.cos(az); U=np.sin(alt)
    a0=np.radians(az0); e0=np.radians(alt0); r0=np.radians(roll)
    fwd=np.array([np.cos(e0)*np.sin(a0),np.cos(e0)*np.cos(a0),np.sin(e0)])
    right=np.array([np.cos(a0),-np.sin(a0),0.])
    up=np.cross(right,fwd)
    r2=np.cos(r0)*right+np.sin(r0)*up; u2=-np.sin(r0)*right+np.cos(r0)*up
    V=np.stack([E,N,U],-1)
    X=V@r2; Y=V@u2; Z=V@fwd
    x=X/Z; y=Y/Z
    rr=x*x+y*y; d=1+k1*rr+k2*rr*rr
    px=cx+f*x*d; py=cy-f*y*d
    return px,py,Z,np.degrees(alt)
