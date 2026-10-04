"""Compile unchanged pristine WRF acoustic routines with a C-callable adapter.

Only module dependencies (configuration storage and g) are supplied by a small
stub. Numerical subroutine bodies are copied verbatim and hashed. Inputs are
real stage fixtures; results are independent Fortran binary32 savepoints.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

WRF=Path("<USER_HOME>/src/wrf_pristine/WRF")
FIELDS3=("u","ru_tend","v","rv_tend","p","pb","ph","php","alt","al",
         "cqu","cqv","ww","ww_1","u_1","v_1","uam","vam","wwam","t",
         "t_1","t_ave","ft","w","rw_tend","w_save","t_2ave","ph_1","phb",
         "ph_tend","c2a","cqw","alb","a","alpha","gamma","pm1")
FIELDS2=("mu","mut","muave","muts","muu","muv","mudf","msfux","msfuy",
         "msfvx","msfvx_inv","msfvy","msftx","msfty","mu_tend","ht")
VECTORS=("c1h","c2h","c1f","c2f","c3h","c4h","c3f","c4f","znu","rdn","rdnw","dnw","fnm","fnp")
SCALARS=("rdx","rdy","dts","cf1","cf2","cf3","emdiv","epssm","t0","smdiv","g","dampcoef","zdamp")
FUNCTIONS=("calc_coef_w","advance_uv","advance_mu_t","advance_w","calc_p_rho")


def build(directory,*,contraction="off",native=False):
    directory.mkdir(parents=True,exist_ok=True)
    source=(WRF/"dyn_em/module_small_step_em.F").read_text()
    routines={name:re.search(r"(?ims)^SUBROUTINE "+name+r"\b.*?^END SUBROUTINE "+name+r"\b",source).group(0)
              for name in FUNCTIONS}
    flags=sorted(set(re.findall(r"config_flags%(\w+)","\n".join(routines.values()))))
    config=[]
    for name in flags:
        typ="integer" if name in ("phi_adv_z","damp_opt") else "real" if name in ("dampcoef","zdamp") else "logical"
        default="0" if typ=="integer" else "0." if typ=="real" else ".false."
        config.append(f"{typ} :: {name}={default}")
    header="module b_core_wrf\nuse iso_c_binding\nimplicit none\nreal, parameter :: g=9.81\n" + \
           "type grid_config_rec_type\n"+"\n".join(config)+"\nend type\ncontains\n"
    wrapper="""subroutine wrf_oracle(nx,ny,nz,phase,f3,f2,vec,s,flags) bind(c)
integer(c_int),value :: nx,ny,nz,phase
real(c_float),intent(inout) :: f3(nx+3,nz+1,ny+3,N3),f2(nx+3,ny+3,N2)
real(c_float),intent(inout) :: vec(nz+1,NV),s(NS)
integer(c_int),intent(in) :: flags(4)
type(grid_config_rec_type) :: config_flags
config_flags%specified=flags(1)/=0
config_flags%nested=flags(2)/=0
config_flags%periodic_x=flags(3)/=0
config_flags%damp_opt=flags(4)
config_flags%phi_adv_z=1
config_flags%dampcoef=s(IDAMP)
config_flags%zdamp=s(IZDAMP)
select case(phase)
""".replace("N3",str(len(FIELDS3))).replace("N2",str(len(FIELDS2))).replace("NV",str(len(VECTORS))).replace("NS",str(len(SCALARS))).replace("IDAMP",str(SCALARS.index("dampcoef")+1)).replace("IZDAMP",str(SCALARS.index("zdamp")+1))
    bounds=dict(ids="1",ide="nx+1",jds="1",jde="ny+1",kds="1",kde="nz+1",
                ims="0",ime="nx+2",jms="0",jme="ny+2",kms="1",kme="nz+1",
                its="1",ite="nx+1",jts="1",jte="ny+1",kts="1",kte="nz+1",
                config_flags="config_flags",step="1",spec_zone="1",non_hydrostatic=".true.",top_lid=".false.")
    for phase,name in enumerate(FUNCTIONS,1):
        signature=routines[name][:routines[name].index(")")+1]
        signature=re.sub(r"![^\n]*","",signature).replace("&","")
        names=[x.strip().lower() for x in signature[signature.index("(")+1:-1].split(",")]
        actual=[]
        for arg in names:
            arg={"t_2":"t","mu1":"mu"}.get(arg,arg)
            # Bind the literal solve_em caller, not just formal-name matches.
            # Both routines share t_2save; pressure consumes live grid%muts.
            if name=="advance_mu_t" and arg=="t_ave":arg="t_2ave"
            if name=="calc_p_rho" and arg=="mut":arg="muts"
            if arg in FIELDS3:
                actual.append(f"f3(:,:,:,{FIELDS3.index(arg)+1})")
            elif arg in FIELDS2:
                actual.append(f"f2(:,:,{FIELDS2.index(arg)+1})")
            elif arg in VECTORS:
                actual.append(f"vec(:,{VECTORS.index(arg)+1})")
            elif arg in SCALARS:
                actual.append(f"s({SCALARS.index(arg)+1})")
            else:
                actual.append(bounds[arg])
        wrapper+=f"case({phase})\ncall {name}( &\n"+", &\n".join(actual)+")\n"
    wrapper+="end select\nend subroutine\nend module\n"
    text=header+"\n\n".join(routines.values())+"\n"+wrapper
    (directory/"oracle.f90").write_text(text)
    cmd=[os.environ.get("FC","<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"),"-O2","-fPIC","-shared","-ffree-line-length-none","-fcheck=bounds",
         "-ffp-contract="+contraction,*(["-march=native"] if native else []),
         "-J",str(directory),str(directory/"oracle.f90"),"-o",str(directory/"oracle.so")]
    subprocess.run(cmd,check=True,capture_output=True,text=True)
    meta=dict(source=str(WRF/"dyn_em/module_small_step_em.F"),source_sha256=hashlib.sha256(source.encode()).hexdigest(),
              routine_sha256={name:hashlib.sha256(text.encode()).hexdigest() for name,text in routines.items()},
              compile=cmd,real_kind_bytes=4,config_stub=flags,
              wrf_commit=subprocess.check_output(["git","-C",str(WRF),"rev-parse","HEAD"],text=True).strip())
    (directory/"manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    return directory/"oracle.so"


class Oracle:
    def __init__(self,path,state,cfg):
        self.cfg=cfg
        self.nz,self.ny,self.nx=state.theta.shape
        nx,ny,nz=self.nx,self.ny,self.nz
        self.f3=np.ones((nx+3,nz+1,ny+3,len(FIELDS3)),dtype=np.float32,order="F")
        self.f2=np.ones((nx+3,ny+3,len(FIELDS2)),dtype=np.float32,order="F")
        self.vec=np.zeros((nz+1,len(VECTORS)),dtype=np.float32,order="F")
        mapping={"ru_tend":"u_tend","rv_tend":"v_tend","pb":"p_base","php":"php_stage",
                 "uam":"ru_m","vam":"rv_m","wwam":"ww_m","t":"theta_coupled_work",
                 "t_1":"theta_1","t_ave":"theta_coupled_work","ft":"theta_tend","rw_tend":"rw_tend_pg_buoy"}
        for name in FIELDS3:
            value=getattr(state,mapping.get(name,name),None)
            if value is None:
                value=np.zeros((nz+1,ny,nx),np.float32)
            value=np.asarray(value,np.float32)
            k,y,x=value.shape
            # Domain-edge halos are explicit repeated inputs. WRF specified/
            # nested active bounds avoid consuming them in numerical updates.
            a=np.pad(value,((0,nz+1-k),(1,ny+2-y),(1,nx+2-x)),mode="edge")
            self.f3[:,:,:,FIELDS3.index(name)]=a.transpose(2,0,1)
        for name in FIELDS2:
            value=getattr(state,"mu_work" if name=="mu" else name)
            a=np.asarray(value,np.float32)
            y,x=a.shape
            a=np.pad(a,((1,ny+2-y),(1,nx+2-x)),mode="edge")
            self.f2[:,:,FIELDS2.index(name)]=a.T
        for name in VECTORS:
            value=getattr(state,name,None)
            if value is None:
                value=getattr(state,name.replace("3","1").replace("4","2"),None)
            if value is not None:
                a=np.asarray(value,np.float32).reshape(-1)
                self.vec[:,VECTORS.index(name)]=np.pad(a,(0,nz+1-len(a)),mode="edge")
        values=dict(rdx=1./cfg.dx,rdy=1./cfg.dy,dts=cfg.dt,cf1=float(state.cf1),cf2=float(state.cf2),
                    cf3=float(state.cf3),emdiv=.01,epssm=cfg.epssm,t0=300.,smdiv=.1,g=9.81,
                    dampcoef=cfg.dampcoef,zdamp=cfg.zdamp)
        self.s=np.asarray([values[name] for name in SCALARS],np.float32)
        self.flags=np.asarray([cfg.specified,cfg.nested,cfg.periodic_x,cfg.damp_opt],np.int32)
        self.library=ctypes.CDLL(str(path))
        self.fn=self.library.wrf_oracle
        ptr=ctypes.c_void_p
        self.fn.argtypes=[ctypes.c_int]*4+[ptr]*5
        self.fn.restype=None

    def run(self,phase):
        self.fn(self.nx,self.ny,self.nz,phase,*[a.ctypes.data for a in (self.f3,self.f2,self.vec,self.s,self.flags)])

    def field(self,name):
        nz,ny,nx=self.nz,self.ny,self.nx
        if name in FIELDS3:
            a=self.f3[:,:,:,FIELDS3.index(name)].transpose(1,2,0)
            k=nz+1 if name in ("w","ph","ww","ww_1","a","alpha","gamma","ph_1","phb","rw_tend") else nz
            return a[:k,1:ny+1+int(name in ("v","v_1")),1:nx+1+int(name in ("u","u_1"))].copy()
        return self.f2[:,:,FIELDS2.index(name)].T[1:ny+1,1:nx+1].copy()


if __name__=="__main__":
    print(build(Path("<USER_HOME>/wrf_gpu2_lanes/b-core/phase1/pristine")))
