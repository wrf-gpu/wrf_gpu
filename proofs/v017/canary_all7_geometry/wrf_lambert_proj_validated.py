import numpy as np

# WRF Lambert Conformal projection (matches module_llxy / map_utils set_lc, llij_lc, ijll_lc)
# Reference: WRFV4 share/module_llxy.F
RE_M = 6370000.0   # WRF default earth radius (a)
DEG2RAD = np.pi/180.0
RAD2DEG = 180.0/np.pi

class LCC:
    def __init__(self, truelat1, truelat2, stand_lon, ref_lat, ref_lon, dx, knowni=None, knownj=None, e_we=None, e_sn=None):
        # ref point is grid CENTER (CEN_LAT/CEN_LON) at (knowni,knownj). For MOAD WRF uses center.
        self.truelat1=truelat1; self.truelat2=truelat2; self.stdlon=stand_lon
        self.dx=dx
        # cone factor
        if abs(truelat1-truelat2) > 0.1:
            self.cone = (np.log(np.cos(truelat1*DEG2RAD)) - np.log(np.cos(truelat2*DEG2RAD))) / \
                        (np.log(np.tan((90.0-abs(truelat1))*DEG2RAD*0.5)) - np.log(np.tan((90.0-abs(truelat2))*DEG2RAD*0.5)))
        else:
            self.cone = np.sin(abs(truelat1)*DEG2RAD)
        self.hemi = 1.0 if truelat1>=0 else -1.0
        # rebydx
        self.rebydx = RE_M/dx
        # compute pole point so that (ref_lat,ref_lon) -> (knowni,knownj)
        deltalon1 = ref_lon - stand_lon
        if deltalon1>180: deltalon1-=360
        if deltalon1<-180: deltalon1+=360
        tl1r = truelat1*DEG2RAD
        ctl1r = np.cos(tl1r)
        rsw = self.rebydx*ctl1r/self.cone * (np.tan((90.0*self.hemi-ref_lat)*DEG2RAD/2.0) /
                                             np.tan((90.0*self.hemi-truelat1)*DEG2RAD/2.0))**self.cone
        arg = self.cone*(deltalon1*DEG2RAD)
        # knowni/knownj: grid center
        self.knowni = knowni; self.knownj = knownj
        self.polei = self.hemi*knowni - self.hemi*rsw*np.sin(arg)
        self.polej = self.hemi*knownj + rsw*np.cos(arg)

    def latlon_to_ij(self, lat, lon):
        deltalon = lon - self.stdlon
        if deltalon>180: deltalon-=360
        if deltalon<-180: deltalon+=360
        tl1r = self.truelat1*DEG2RAD
        ctl1r=np.cos(tl1r)
        rm = self.rebydx*ctl1r/self.cone*(np.tan((90.0*self.hemi-lat)*DEG2RAD/2.0)/
                                          np.tan((90.0*self.hemi-self.truelat1)*DEG2RAD/2.0))**self.cone
        arg = self.cone*(deltalon*DEG2RAD)
        i = self.polei + self.hemi*rm*np.sin(arg)
        j = self.polej - rm*np.cos(arg)
        i = self.hemi*i; j=self.hemi*j
        return i, j

    def ij_to_latlon(self, i, j):
        chi1 = (90.0 - self.hemi*self.truelat1)*DEG2RAD
        chi2 = (90.0 - self.hemi*self.truelat2)*DEG2RAD
        inew = self.hemi*i; jnew=self.hemi*j
        xx = inew - self.polei
        yy = self.polej - jnew
        r2 = xx*xx+yy*yy
        r = np.sqrt(r2)/self.rebydx
        if r2==0:
            lat = self.hemi*90.0; lon=self.stdlon
        else:
            lon = self.stdlon + RAD2DEG*np.arctan2(self.hemi*xx, yy)/self.cone
            lon = np.mod(lon+360.0, 360.0)
            if abs(chi1-chi2) < 1e-12:
                chi = 2.0*np.arctan((r/np.tan(chi1))**(1.0/self.cone) * np.tan(chi1*0.5))
            else:
                chi = 2.0*np.arctan((r*self.cone/np.sin(chi1))**(1.0/self.cone) * np.tan(chi1*0.5))
            lat = (90.0 - chi*RAD2DEG)*self.hemi
            if lon>180: lon-=360
            if lon<-180: lon+=360
        return lat, lon

if __name__=="__main__":
    # Validate against operational d01: e_we=94,e_sn=60 => mass dims 93x59; center at ((94)/2,(60)/2)? 
    # WRF MOAD center = (e_we/2, e_sn/2) in the dot grid; map_utils uses (real(e_we)/2., real(e_sn)/2.)? 
    # geogrid sets ref at grid center: knowni=(e_we+1)/2, knownj=(e_sn+1)/2 (dot-point center)
    ewe,esn=94,60
    p=LCC(25.0,30.0,-16.4,28.3,-16.4,9000.0,knowni=(ewe+1)/2.0,knownj=(esn+1)/2.0)
    # mass point (1,1) center is at i=1.5,j=1.5 in dot index; geo_em XLAT_M[0,0] = mass point (1,1)
    # mass point (m,n) located at dot (m+0.5, n+0.5)
    lat,lon=p.ij_to_latlon(1.5,1.5)
    print(f"d01 SW mass(1,1): comp={lat:.4f},{lon:.4f}  truth=25.8882,-20.5422")
    lat,lon=p.ij_to_latlon(93+0.5,59+0.5)
    print(f"d01 NE mass(93,59): comp={lat:.4f},{lon:.4f}  truth=30.5834,-12.0758")
