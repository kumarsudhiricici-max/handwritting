import cv2
import numpy as np
from scipy.stats import skew, kurtosis

FEATURE_NAMES = [
    "ink_density","aspect_ratio","bbox_fill_ratio","contour_area_ratio",
    "perimeter_norm","centroid_offset","radial_mean","radial_std",
    "radial_cv","radial_range","radial_slope","angular_coverage",
    "turning_mean","turning_std","turning_abs_mean","width_mean",
    "width_std","width_cv","x_spread","y_spread","x_skew","y_skew",
    "x_kurtosis","y_kurtosis"
]

def to_gray(image):
    if image is None:
        raise ValueError("Empty image")
    if len(image.shape)==3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()

def make_mask(image):
    gray = cv2.GaussianBlur(to_gray(image),(5,5),0)
    a = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    b = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # Choose a foreground fraction closest to a typical sparse drawing.
    mask=min([a,b], key=lambda m: abs(np.mean(m>0)-0.15))
    k=np.ones((3,3),np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k)
    return mask

def largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask,8)
    if n<=1: return mask
    idx=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
    out=np.zeros_like(mask); out[labels==idx]=255
    return out

def skeletonize(mask):
    img=mask.copy(); skel=np.zeros_like(img)
    kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    for _ in range(max(mask.shape)):
        eroded=cv2.erode(img,kernel)
        opened=cv2.dilate(eroded,kernel)
        skel=cv2.bitwise_or(skel,cv2.subtract(img,opened))
        img=eroded
        if cv2.countNonZero(img)==0: break
    return skel

def extract_features(image):
    mask=largest_component(make_mask(image))
    ys,xs=np.where(mask>0)
    if len(xs)<50: raise ValueError("Insufficient foreground pixels")

    x0,x1=xs.min(),xs.max(); y0,y1=ys.min(),ys.max()
    bw=max(1,x1-x0+1); bh=max(1,y1-y0+1)
    crop=mask[y0:y1+1,x0:x1+1]

    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    c=max(contours,key=cv2.contourArea)
    area=float(cv2.contourArea(c)); peri=float(cv2.arcLength(c,True))

    cx,cy=xs.mean(),ys.mean()
    center=np.array([(x0+x1)/2,(y0+y1)/2])
    centroid_offset=float(np.linalg.norm(np.array([cx,cy])-center)/(np.hypot(bw,bh)+1e-9))

    sk=skeletonize(crop)
    sy,sx=np.where(sk>0)
    if len(sx)<20: sx,sy=xs-x0,ys-y0
    sx=sx.astype(float); sy=sy.astype(float)

    ccx,ccy=sx.mean(),sy.mean()
    dx=sx-ccx; dy=sy-ccy
    r=np.sqrt(dx*dx+dy*dy)+1e-8
    ang=np.unwrap(np.arctan2(dy,dx))
    order=np.argsort(ang); ang=ang[order]; r=r[order]
    r=r/(np.hypot(bw,bh)+1e-9)

    if len(r)>2:
        slope=float(np.polyfit(ang,r,1)[0])
    else: slope=0.0

    # Turning statistics along a deterministic spatial ordering.
    o=np.argsort(sy*(crop.shape[1]+1)+sx)
    px,py=sx[o],sy[o]
    if len(px)>=3:
        v1=np.c_[np.diff(px[:-1]),np.diff(py[:-1])]
        v2=np.c_[np.diff(px[1:]),np.diff(py[1:])]
        n1=np.linalg.norm(v1,axis=1)+1e-8
        n2=np.linalg.norm(v2,axis=1)+1e-8
        turns=np.arccos(np.clip(np.sum(v1*v2,axis=1)/(n1*n2),-1,1))
    else: turns=np.array([0.0])

    dist=cv2.distanceTransform(crop,cv2.DIST_L2,5)
    widths=2*dist[sk>0]; widths=widths[widths>0]
    if len(widths)==0: widths=np.array([1.0])

    xn=(sx-sx.min())/max(1,sx.max()-sx.min())
    yn=(sy-sy.min())/max(1,sy.max()-sy.min())

    def safe_stat(fn,a):
        try: return float(fn(a,bias=False))
        except: return 0.0

    feat=[
        float(np.mean(crop>0)), bw/bh,
        float(np.mean(crop>0)), area/(bw*bh+1e-9),
        peri/(2*(bw+bh)+1e-9), centroid_offset,
        float(np.mean(r)),float(np.std(r)),
        float(np.std(r)/(np.mean(r)+1e-9)),float(np.ptp(r)),slope,
        float((ang.max()-ang.min())/(2*np.pi)) if len(ang) else 0.0,
        float(np.mean(turns)),float(np.std(turns)),float(np.mean(np.abs(turns))),
        float(np.mean(widths)),float(np.std(widths)),
        float(np.std(widths)/(np.mean(widths)+1e-9)),
        float(np.std(xn)),float(np.std(yn)),
        safe_stat(skew,xn),safe_stat(skew,yn),
        safe_stat(kurtosis,xn),safe_stat(kurtosis,yn)
    ]
    return np.asarray(feat,dtype=np.float64), mask
