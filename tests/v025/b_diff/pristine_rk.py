"""Unmodified pristine WRF RK/prep/finish operators, C-callable savepoints."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

WRF=Path("<USER_HOME>/src/wrf_pristine/WRF")
ROUTINES=("small_step_prep","small_step_finish","rk_addtend_dry",
          "horizontal_pressure_gradient","coriolis","curvature","calc_mu_uv","calc_mu_uv_1","calc_p_rho_phi")
FIELDS3=("u_1","u_2","v_1","v_2","w_1","w_2","t_1","t_2","ph_1","ph_2",
         "u_save","v_save","w_save","t_save","ph_save","ww","ww_save","ww1",
         "c2a","pb","p","alt","al","php","cqu","cqv","ru","rv","rw",
         "ru_tend","rv_tend","rw_tend","ph_tend","t_tend","ru_tendf","rv_tendf",
         "rw_tendf","ph_tendf","t_tendf","h_diabatic","alb","phb","t",
         "q0","qv","qc","qr","qi","qs","qg")
FIELDS2=("mub","mu_1","mu_2","muu","muus","muv","muvs","mut","muts","mudf",
         "mu_save","mu_tend","mu_tendf","msftx","msfty","msfux","msfuy",
         "msfvx","msfvx_inv","msfvy","f","e","sina","cosa","xlat")
VECTORS=("c1h","c2h","c1f","c2f","c3h","c4h","c3f","c4f","fnm","fnp","rdnw","fzm","fzp","znu","znw","dnw","rdn")
SCALARS=("rdx","rdy","cf1","cf2","cf3","cfn","cfn1","dts","p0","t0","ptop")


def build(directory, top_lid=False):
    directory.mkdir(parents=True,exist_ok=True)
    sources={n:(WRF/"dyn_em"/n).read_text() for n in
             ("module_small_step_em.F","module_em.F","module_big_step_utilities_em.F")}
    routines={}
    for name in ROUTINES:
        src=sources["module_small_step_em.F" if name.startswith("small_step") else
                    "module_em.F" if name=="rk_addtend_dry" else "module_big_step_utilities_em.F"]
        routines[name]=re.search(r"(?ims)^\s*SUBROUTINE "+name+r"\b.*?^\s*END SUBROUTINE "+name+r"\b",src).group(0)
    flags=sorted(set(re.findall(r"config_flags%(\w+)","\n".join(routines.values()))))
    config="\n".join(("integer :: map_proj=3" if f=="map_proj" else f"logical :: {f}=.false.") for f in flags)
    macros=sources["module_big_step_utilities_em.F"].split("MODULE module_big_step_utilities_em")[0]
    header=macros+"module wrf_rk_oracle\nuse iso_c_binding\nimplicit none\nreal,parameter :: r_d=287.,rvovrd=461.6/287.,cpovcv=1004.5/717.5,cvpm=-717.5/1004.5,p1000mb=100000.\nreal,parameter :: reradius=1./6370000.,degrad=3.141592653589793238/180.\ninteger,parameter :: PARAM_FIRST_SCALAR=2,P_QV=2\ntype grid_config_rec_type\n"+config+"\nend type\ncontains\n"
    wrapper=f'''subroutine oracle(nx,ny,nz,phase,rk,f3,f2,v,s) bind(c)
integer(c_int),value :: nx,ny,nz,phase,rk
real(c_float) :: f3(nx+3,nz+1,ny+3,{len(FIELDS3)}),f2(nx+3,ny+3,{len(FIELDS2)})
real(c_float) :: v(nz+1,{len(VECTORS)}),s({len(SCALARS)})
type(grid_config_rec_type) :: config_flags
config_flags%specified=.true.
select case(phase)
'''
    bounds=dict(ids="1",ide="nx+1",jds="1",jde="ny+1",kds="1",kde="nz+1",ims="0",ime="nx+2",
                jms="0",jme="ny+2",kms="1",kme="nz+1",its="1",ite="nx+1",jts="1",jte="ny+1",
                kts="1",kte="nz+1",ips="1",ipe="nx+1",jps="1",jpe="ny+1",kps="1",kpe="nz+1",
                config_flags="config_flags",rk_step="rk",rk_order="3",number_of_small_timesteps="1",
                non_hydrostatic=".true.",top_lid=".true." if top_lid else ".false.",n_moist="7",use_theta_m="1",hypsometric_opt="rk",
                moist=f"f3(:,:,:,{FIELDS3.index('q0')+1}:{FIELDS3.index('qg')+1})")
    for phase,name in enumerate(ROUTINES,1):
        signature=re.sub(r"![^\n]*","",routines[name][:routines[name].index(")")+1]).replace("&","")
        names=[s.strip().lower() for s in signature[signature.index("(")+1:-1].split(",")]
        actual=[]
        for arg in names:
            if name=="calc_mu_uv_1" and arg=="mu":arg="muts"
            arg={"u":"u_2","v":"v_2","w":"w_2","ph":"ph_2","mu":"mu_2","c1":"c1h","c2":"c2h","t":"t_2"}.get(arg,arg)
            if name=="calc_mu_uv_1":arg={"muu":"muus","muv":"muvs"}.get(arg,arg)
            if arg in FIELDS3:actual.append(f"f3(:,:,:,{FIELDS3.index(arg)+1})")
            elif arg in FIELDS2:actual.append(f"f2(:,:,{FIELDS2.index(arg)+1})")
            elif arg in VECTORS:actual.append(f"v(:,{VECTORS.index(arg)+1})")
            elif arg in SCALARS:actual.append(f"s({SCALARS.index(arg)+1})")
            else:actual.append(bounds[arg])
        wrapper+=f"case({phase})\ncall {name}( &\n"+", &\n".join(actual)+")\n"
    wrapper+="end select\nend subroutine\nend module\n"
    massv=(WRF/'frame/libmassv.F').read_text()
    vspow=re.search(r"(?ims)^\s*subroutine vspow\b.*?^\s*end\s*$",massv).group(0)
    fatal='subroutine wrf_error_fatal(message)\ncharacter(len=*) :: message\nerror stop "invalid oracle hypsometric option"\nend subroutine\n'
    path=directory/"oracle.f90";path.write_text(header+"\n".join(routines.values())+"\n"+wrapper+vspow+'\n'+fatal)
    command=[os.environ.get("FC","<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"),"-O2","-cpp","-shared","-fPIC","-ffree-line-length-none","-ffp-contract=off","-fcheck=bounds","-J",str(directory),str(path),"-o",str(directory/"oracle.so")]
    cp=subprocess.run(command,capture_output=True,text=True)
    if cp.returncode:raise RuntimeError(cp.stderr)
    manifest=dict(source_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()},
        routine_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in routines.items()},
        vspow_sha256=hashlib.sha256(vspow.encode()).hexdigest(),
        caller=str(WRF/"dyn_em/solve_em.F"),caller_sha256=hashlib.sha256((WRF/"dyn_em/solve_em.F").read_bytes()).hexdigest(),
        REAL_bytes=4,command=command,limitations="specified/nested, map_proj3, source nonpolar operators; caller fields mapped explicitly in probe_rk.py")
    (directory/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return directory/"oracle.so"


class Oracle:
    def __init__(self,path,shape):
        self.nz,self.ny,self.nx=shape
        nz,ny,nx=shape
        self.f3=np.zeros((nx+3,nz+1,ny+3,len(FIELDS3)),np.float32,order="F")
        self.f2=np.ones((nx+3,ny+3,len(FIELDS2)),np.float32,order="F")
        self.v=np.zeros((nz+1,len(VECTORS)),np.float32,order="F")
        self.s=np.zeros(len(SCALARS),np.float32)
        self.library=ctypes.CDLL(str(path));self.fn=self.library.oracle
        self.fn.argtypes=[ctypes.c_int]*5+[ctypes.c_void_p]*4;self.fn.restype=None

    def set(self,name,value):
        value=np.asarray(value,np.float32)
        nz,ny,nx=self.nz,self.ny,self.nx
        if name in FIELDS3:
            k,y,x=value.shape
            padded=np.pad(value,((0,nz+1-k),(1,ny+2-y),(1,nx+2-x)),mode="edge")
            self.f3[:,:,:,FIELDS3.index(name)]=padded.transpose(2,0,1)
        elif name in FIELDS2:
            y,x=value.shape
            self.f2[:,:,FIELDS2.index(name)]=np.pad(value,((1,ny+2-y),(1,nx+2-x)),mode="edge").T
        elif name in VECTORS:
            self.v[:,VECTORS.index(name)]=np.pad(value.reshape(-1),(0,nz+1-value.size),mode="edge")
        else:self.s[SCALARS.index(name)]=value

    def field(self,name,shape):
        if name in FIELDS3:
            k,y,x=shape
            return self.f3[1:x+1,:k,1:y+1,FIELDS3.index(name)].transpose(1,2,0).copy()
        y,x=shape
        return self.f2[1:x+1,1:y+1,FIELDS2.index(name)].T.copy()

    def run(self,name,rk):
        self.fn(self.nx,self.ny,self.nz,ROUTINES.index(name)+1,rk,
                *[v.ctypes.data for v in (self.f3,self.f2,self.v,self.s)])
