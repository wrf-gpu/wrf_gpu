"""Pure-host WRF stochastic binding; this module stays outside physics imports."""
from dataclasses import dataclass, replace
import math
from numbers import Integral

VARIABLES = "WTPRSQO"
_GROUPS = ("sppt", "rand_pert", "spp_conv", "spp_pbl", "spp_lsm")
_STD = (.5,.03,.3,.15,.3)
_CUTOFF = (2.,3.,3.,2.,3.)
_LENGTH = (150000.,500000.,150000.,700000.,50000.)
_TAU = (21600.,21600.,21600.,21600.,86400.)
SEED_DEFAULTS = (811,53,17,171,217,317)
SWITCHES = ("skebs","sppt","rand_perturb","spp","spp_conv","spp_pbl","spp_lsm","stoch_force_opt")
SEED_KEYS = ("iseed_skebs","iseed_sppt","iseed_rand_pert","iseed_spp_conv","iseed_spp_pbl","iseed_spp_lsm")
_FORCT_DEFAULTS = (("kminforct",1),("lminforct",1),("kmaxforct",1000000),("lmaxforct",1000000))
_INTEGER_KEYS = frozenset((*SWITCHES,*SEED_KEYS,"nens","multi_perturb","num_pert_3d",
    "skebs_vertstruc","stoch_vertstruc_opt","sppt_vertstruc","rand_pert_vertstruc",
    "vertstruc_spp_conv","vertstruc_spp_pbl","vertstruc_spp_lsm",
    *(key for key,_ in _FORCT_DEFAULTS),"kminforc","lminforc","kmaxforc","lmaxforc"))
BOUND_KEYS = frozenset((*SWITCHES,*SEED_KEYS,"nens","hrrr_cycling","stoch_vertstruc_opt","skebs_vertstruc",
    "sppt_vertstruc","rand_pert_vertstruc","vertstruc_spp_conv","vertstruc_spp_pbl","vertstruc_spp_lsm",
    "tot_backscat_psi","tot_backscat_t","ztau_psi","ztau_t","rexponent_psi","rexponent_t",
    "kminforct","kmaxforct","lminforct","lmaxforct",
    *(prefix+group for group in _GROUPS for prefix in
      ("gridpt_stddev_","stddev_cutoff_","lengthscale_","timescale_"))))
# These controls do not enter original WRF's seven primary pattern consumers.
# INITIALIZE_STOCH passes the *T bounds to BOTH SKEBS setup calls.
PASSIVE_KEYS = frozenset(("kminforc","kmaxforc","lminforc","lmaxforc",
    "zsigma2_eps","zsigma2_eta","spdt","num_pert_3d"))
EXTENSION_KEYS = frozenset(("multi_perturb","perturb_bdy",
    *("pert_"+group for group in ("farms","deng","mynn","noah","thom","cld3")),
    *("pert_"+group+"_"+field for group,fields in {
        "farms":("albedo","aod","angexp","aerasy","qv","qc","qs"),
        "deng":("qv","qc","t","w"),"mynn":("qv","qc","t","qke"),
        "noah":("qv","t","smois","tslb"),"thom":("qv","qc","qi","qs","ni"),
        "cld3":("qv","t")}.items() for field in fields)))


@dataclass(frozen=True)
class PatternConfig:
    variable: str
    vertical: int
    stddev: float
    cutoff: float
    lengthscale: float
    timescale: float
    backscatter: float = 1e-6
    decorrelation: float = 10800.
    exponent: float = -1.83
    kmin: int = 1
    lmin: int = 1
    kmax: int = 1000000
    lmax: int = 1000000

    def setup_kwargs(self):
        return {key:value for key,value in vars(self).items() if key != "cutoff"}


@dataclass(frozen=True, eq=False)
class StochasticConfig:
    root: bool
    patterns: tuple
    seeds: tuple
    nens: int
    produce: tuple
    apply_pbl: bool
    # Host-only init count: med_nest_initial (share/mediation_integrate.F:845)
    # re-calls start_domain(parent) once per d01 child opened at t0, after
    # d01's tiles exist, so INITIALIZE_STOCH re-seeds and spins RAND ten more
    # times per child. Initial carry values only; never a step-program switch.
    # Set only by the nested loader from the executed tree (with_executed_d01_children).
    nest_init_rounds: int = 0

    def _layout(self):
        # All varying numerical controls live in device carry after init. Only
        # these static switches affect the step; seeds/members reuse executables.
        return (self.root,tuple(None if p is None else (p.variable,p.vertical) for p in self.patterns),
                self.produce,self.apply_pbl)

    def __hash__(self):
        return hash(self._layout())

    def __eq__(self, other):
        return isinstance(other,StochasticConfig) and self._layout() == other._layout()


def _integer4(key,value):
    if isinstance(value,bool) or not isinstance(value,Integral) or not -(2**31) <= value < 2**31:
        raise ValueError(f"&stoch {key} must be a WRF INTEGER(4)")
    return int(value)


def bind_stochastic(wrf_namelist, domain):
    group = {str(k).lower():v for k,v in (wrf_namelist.get("stoch") or {}).items()}
    # Registry INTEGER controls must be validated before int()/bool() can
    # truncate fractions or accept quoted numbers, including child entries.
    for key in _INTEGER_KEYS.intersection(group):
        raw=group[key]
        for item in raw if isinstance(raw,(list,tuple)) else (raw,):
            if item is not None:
                _integer4(key,item)
    number = int(str(domain).removeprefix("d"))
    def value(key,default, *, global_value=False):
        result = group.get(key,default)
        if isinstance(result,(tuple,list)):
            index = 0 if global_value else number-1
            result = result[index] if index < len(result) and result[index] is not None else default
        elif not global_value and number != 1:
            # A scalar Fortran namelist assignment sets the FIRST array entry.
            result = default
        return default if result is None else result
    def integer(key,default):
        return _integer4(key,value(key,default,global_value=True))
    def any_on(key):
        val = group.get(key,0)
        return any(v not in (0,None,False) for v in val) if isinstance(val,(list,tuple)) else bool(val)
    for key in (*SWITCHES,"stoch_vertstruc_opt"):
        values = group.get(key,0)
        values = values if isinstance(values,(list,tuple)) else (values,)
        if any(v not in (None,0,1) for v in values):
            raise ValueError(f"&stoch {key} supports WRF values0/1 only")
    skebs = any_on("skebs") or any_on("stoch_force_opt")
    sppt, rand = any_on("sppt"), any_on("rand_perturb")
    # check_a_mundo sets ALL three arrays to one when any domain sets spp.
    spp = any_on("spp")
    selected = (skebs,skebs,sppt,rand, bool(value("spp_conv",0)) or spp,
                bool(value("spp_pbl",0)) or spp, bool(value("spp_lsm",0)) or spp)
    spp_on = spp or any(any_on(key) for key in ("spp_conv","spp_pbl","spp_lsm"))
    allocated = (*selected[:4],spp_on,spp_on,spp_on)
    if not any(allocated):
        return None
    # Original module_check_a_mundo.F:1155-1200 fatals custom *forct bounds
    # whenever a Gaussian scheme is selected in ANY domain. SKEBS-only is valid.
    if sppt or rand or spp_on:
        for key,default in _FORCT_DEFAULTS:
            if integer(key,default)!=default:
                raise ValueError(f"&stoch {key}: custom spectral bounds are SKEBS-only; "
                                 f"WRF requires {key}={default} with SPPT, RAND_PERTURB or SPP")
    unknown = {k for k in group if k.startswith("pert_") or k in {"multi_perturb","perturb_bdy"}}
    if any(value(k,0) for k in unknown):
        raise ValueError("multi-perturb and boundary-perturbation extensions are not qualified")
    if value("hrrr_cycling",True,global_value=True) is not True:
        raise ValueError("stochastic restart currently requires hrrr_cycling=.true.")
    patterns = []
    for index, (variable,on) in enumerate(zip(VARIABLES,allocated)):
        if not on:
            patterns.append(None)
            continue
        if index < 2:
            suffix = "psi" if variable == "W" else "t"
            vertical = integer("skebs_vertstruc",0)
            if any_on("stoch_vertstruc_opt"):
                vertical = 1
            pattern = PatternConfig(variable,vertical,float(value("gridpt_stddev_sppt",.5)),
                float(value("stddev_cutoff_sppt",2.)),150000.,21600.,
                float(value("tot_backscat_"+suffix,1e-5 if variable == "W" else 1e-6)),
                float(value("ztau_"+suffix,10800.,global_value=True)),
                float(value("rexponent_"+suffix,-1.83,global_value=True)),
                *[integer(k,d) for k,d in _FORCT_DEFAULTS])
        else:
            pos = index-2
            suffix = _GROUPS[pos]
            vertical_key = suffix+"_vertstruc" if pos < 2 else "vertstruc_"+suffix
            pattern = PatternConfig(variable,integer(vertical_key,0),
                float(value("gridpt_stddev_"+suffix,_STD[pos])),float(value("stddev_cutoff_"+suffix,_CUTOFF[pos])),
                float(value("lengthscale_"+suffix,_LENGTH[pos])),float(value("timescale_"+suffix,_TAU[pos])))
        if pattern.vertical not in (0,1) or not math.isfinite(pattern.cutoff) or pattern.cutoff < 0:
            raise ValueError("stochastic vertical structure must be0/1; cutoff must be finite/nonnegative")
        if (any(not math.isfinite(x) for x in (pattern.stddev,pattern.lengthscale,pattern.timescale,
                pattern.backscatter,pattern.decorrelation,pattern.exponent))
                or min(pattern.stddev,pattern.lengthscale,pattern.backscatter) < 0
                or min(pattern.timescale,pattern.decorrelation) <= 0):
            raise ValueError(f"&stoch {variable}: invalid finite/positive stochastic scales")
        patterns.append(pattern)
    seeds = tuple(integer(key,default) for key,default in zip(SEED_KEYS,SEED_DEFAULTS))
    # nest_init_rounds stays 0 here: the namelist alone cannot know which nests the run opens (a d01-only
    # run may use a max_dom=3 namelist). The nested loader sets it from the EXECUTED d01 edges.
    return StochasticConfig(number==1,tuple(patterns),seeds,integer("nens",1),selected,selected[5],0)


def restart_controls(config):
    """JSON controls a stochastic restart is bound to (critic FIX v50-stoch finding 5).

    The step program is value-blind (seeds, nens and scales live in carry), so restart identity must name
    them explicitly: a resume with another nens/seed/scale would silently continue the saved streams.
    nest_init_rounds is initial-value only and is not part of a continuation.
    """
    if config is None:
        return None
    return {"root": bool(config.root), "nens": int(config.nens), "seeds": [int(s) for s in config.seeds],
            "produce": [bool(x) for x in config.produce], "apply_pbl": bool(config.apply_pbl),
            "patterns": [None if p is None else {k: (v if isinstance(v, str) else float(v) if isinstance(v, float) else int(v))
                                                 for k, v in sorted(vars(p).items())} for p in config.patterns]}


def restart_controls_digest(config):
    import hashlib, json
    controls = restart_controls(config)
    return None if controls is None else hashlib.sha256(
        json.dumps(controls, sort_keys=True, allow_nan=False).encode()).hexdigest()


def with_executed_d01_children(config, domain, children):
    """d01 config whose RAND nest-init count equals the d01 children the run actually opens."""
    if config is None:
        return None
    rounds = len(tuple(children)) if str(domain) == "d01" else 0
    return config if rounds == config.nest_init_rounds else replace(config, nest_init_rounds=rounds)


def _domain_list(group,key,max_dom,default=None):
    # Missing max_domains entries keep their Registry default in WRF; unknown
    # here (None) so that comparisons fail closed instead of guessing.
    raw = group.get(key)
    raw = list(raw) if isinstance(raw,(list,tuple)) else [raw]
    return [raw[i] if i < len(raw) and raw[i] is not None else default for i in range(max_dom)]


def _max_dom(namelist):
    value = (namelist.get("domains") or {}).get("max_dom",1)
    return int(value[0] if isinstance(value,(list,tuple)) else value)


def d01_children(namelist):
    """Domain numbers whose parent_id is 1 (WRF opens each with med_nest_initial)."""
    max_dom = _max_dom(namelist)
    parents = _domain_list(namelist.get("domains") or {},"parent_id",max_dom,0)  # Registry default 0
    return tuple(i+1 for i in range(1,max_dom) if int(parents[i]) == 1)


_CLOCK = ("start_year","start_month","start_day","start_hour","start_minute","start_second")


def _nest_refusals(namelist):
    """Nested layouts whose original-WRF stochastic behaviour the port does not reproduce."""
    max_dom = _max_dom(namelist)
    if max_dom < 2:
        return []
    errors = []
    clock = namelist.get("time_control") or {}
    starts = list(zip(*(_domain_list(clock,key,max_dom) for key in _CLOCK)))
    for child in d01_children(namelist):
        if starts[child-1] != starts[0]:
            # A late d01 child re-runs INITIALIZE_STOCH on d01 mid-run
            # (start_em.F:322-328 with hrrr_cycling, module_stoch.F:145-160):
            # every d01 stream is re-seeded from iseed_* at that time.
            errors.append(f"d{child:02d}: a d01 child that opens after the d01 start re-seeds every d01 "
                          "stochastic stream in original WRF; nest start times must equal d01")
    levels = _domain_list(namelist.get("domains") or {},"e_vert",max_dom)
    if len(set(levels)) > 1:
        # check_a_mundo.F:1242-1250 sizes every domain's patterns by e_vert(1).
        errors.append("stochastic patterns use e_vert(1) levels on every domain; nests with a different "
                      "e_vert are not qualified")
    return errors


def stochastic_nesting_refusals(namelist, names, *, feedback, batch_size):
    """Stochastic + nesting combinations the port does not reproduce (critic FIX v50-stoch finding 4).

    Original WRF feeds child stochastic-forced fields back only through the regular feedback of the
    forced state, and the port's stochastic carry is qualified for one-way nesting at B=1 only: the
    two-way feedback path and the batched (B>1) domain tree never apply the parent pattern force-down
    or the lead-zero child phase. Refused before loading/JAX when any executed domain is active.
    """
    active = False
    for name in names:
        try:
            active = active or bind_stochastic(namelist, name) is not None
        except (ValueError, TypeError, OverflowError):
            return []  # reported by stochastic_refusals
    if not active:
        return []
    errors = []
    if len(tuple(names)) > 1 and feedback:
        errors.append("stochastic physics with two-way nest feedback is not qualified (use feedback=0)")
    if int(batch_size) > 1:
        errors.append("stochastic physics with batched members (B>1) is not qualified; run members as separate B=1 runs")
    return errors


def stochastic_refusals(namelist,domains=None):
    """Validate before importing JAX; no silently ignored stochastic controls."""
    group = {str(k).lower():v for k,v in (namelist.get("stoch") or {}).items()}
    errors=[]
    for key,raw in group.items():
        values = raw if isinstance(raw,(list,tuple)) else (raw,)
        if key in EXTENSION_KEYS:
            if any(v not in (0,False,None) for v in values):
                errors.append(f"&stoch {key}: table-based multi-perturbation/boundary extensions are not qualified")
        elif key not in BOUND_KEYS | PASSIVE_KEYS:
            errors.append(f"&stoch {key}: unknown or unbound WRF stochastic control")
    if domains is None:
        max_dom = (namelist.get("domains") or {}).get("max_dom",1)
        if isinstance(max_dom,(list,tuple)):
            max_dom = max_dom[0]
        domains = tuple(f"d{i:02d}" for i in range(1,int(max_dom)+1))
    active = False
    for domain in domains:
        try:
            cfg = bind_stochastic(namelist,domain)
        except (ValueError,TypeError,OverflowError) as exc:
            errors.append(f"{domain}: {exc}")
            continue
        if cfg is None:
            continue
        active = True
        physics = namelist.get("physics") or {}
        index = int(domain.removeprefix("d"))-1
        def physical(key,default):
            v=physics.get(key,default)
            if isinstance(v,(list,tuple)):
                v=v[index] if index<len(v) and v[index] is not None else default
            return int(v)
        if cfg.patterns[2] is not None and (physical("bl_pbl_physics",5) not in (0,5)
                or physical("cu_physics",1) not in (0,1)
                or (physical("bl_pbl_physics",5)==0 and physical("gwd_opt",0)!=0)):
            errors.append(f"{domain}: SPPT requires raw WRF RK tendencies (PBL0/5, CU0/1; GWDO with MYNN)")
        if cfg.apply_pbl and physical("gwd_opt",0)==3:
            errors.append(f"{domain}: SPP with GSL GWDO (gwd_opt=3) is not qualified")
    if active:
        try:
            errors.extend(_nest_refusals(namelist))
        except (ValueError,TypeError,IndexError) as exc:
            errors.append(f"nested stochastic layout unreadable: {exc}")
    return errors
