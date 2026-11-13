"""Fail-closed exact stochastic restart validation shared by both formats."""
from collections.abc import Mapping
import numpy as np

SEED_NAMES = ("ISEEDARR_SKEBS", "ISEEDARR_SPPT", "ISEEDARR_RAND_PERTURB",
              "ISEEDARRAY_SPP_CONV", "ISEEDARRAY_SPP_PBL", "ISEEDARRAY_SPP_LSM")
PATTERN_FIELDS = ("RU_TENDF_STOCH", "RV_TENDF_STOCH", "RT_TENDF_STOCH", "RSTOCH", "RAND_PERT",
                  "PATTERN_SPP_CONV", "PATTERN_SPP_PBL", "PATTERN_SPP_LSM")
CLIP_FIELDS = tuple("STOCH_CLIP_"+v for v in "WTPRSQO")
SPECTRAL_FIELDS = tuple(f'SPFORC{part}{i}' for i in (2,3,4) for part in ('C','S'))
HISTORY_FIELDS = (*PATTERN_FIELDS,*SEED_NAMES,*CLIP_FIELDS,*SPECTRAL_FIELDS)


def history_specs(factory):
    z = "num_stoch_levels_stag"
    specs = {}
    for name in PATTERN_FIELDS:
        xstag,ybase = name == "RU_TENDF_STOCH", name == "RV_TENDF_STOCH"
        dims = ("Time",z,"south_north_stag" if ybase else "south_north",
                "west_east_stag" if xstag else "west_east")
        unit = "m s-2" if name.startswith(("RU_","RV_")) else "K s-1" if name.startswith("RT_") else "1"
        specs[name] = factory(name,dims,"XYZ",name.replace("_"," "),unit,
                              stagger="X" if xstag else "Y" if ybase else "")
    specs.update({name:factory(name,("Time",),"0", "Cumulative pre-cap stochastic events "+name[-1],
                               "count",dtype="i4") for name in CLIP_FIELDS})
    for name in SPECTRAL_FIELDS:
        specs[name]=factory(name,('Time','south_north_stag','west_east_stag'),'XY ',
            ('real' if name[6]=='C' else 'imag.')+'  spect. coeff. of randomn perturbation field',
            '',stagger='XY',coordinates='XLONG XLAT XTIME')
    return specs


def history_fields(carry, nz, ny, nx):
    """Host output boundary only; broadcast thin fields without allocating a copy."""
    result = seed_arrays(carry)
    mapping = ((0,0),(0,1),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0))
    for name,(index,component) in zip(PATTERN_FIELDS,mapping):
        pattern = carry.patterns[index]
        if pattern is None:
            continue
        value = np.asarray(pattern.field[component] if index == 0 else pattern.field)
        value = np.broadcast_to(value,(nz+1,*value.shape[1:]))
        result[name] = value[:,:ny+int(name == "RV_TENDF_STOCH"),:nx+int(name == "RU_TENDF_STOCH")]
    result.update({name:np.asarray(carry.clip_counts)[index] for index,name in enumerate(CLIP_FIELDS)})
    # module_stoch.F SETUP_RAND_PERTURB('S'/'Q'/'O') retains the real
    # SPFORCC2/3/4 and imaginary SPFORCS2/3/4 on both horizontal staggered
    # edges. Return device references; the writer's existing batch owns D2H.
    for suffix,index in ((2,4),(3,5),(4,6)):
        pattern=carry.patterns[index]
        if pattern is not None and pattern.cosine.size:
            result[f'SPFORCC{suffix}']=pattern.cosine
            result[f'SPFORCS{suffix}']=pattern.sine
        elif pattern is not None:
            # Registry stoch_param_perturb allocates these 'irh' arrays on EVERY
            # domain when spp_on; a child (or a non-producing root) never runs
            # SETUP_RAND_PERTURB, so original WRF writes their alloc value
            # initial_data_value=0 (frame/wrf_num_bytes_between.c). Zero-copy.
            zero=np.broadcast_to(np.float32(0),(ny+1,nx+1))
            result[f'SPFORCC{suffix}']=zero
            result[f'SPFORCS{suffix}']=zero
    return result


def validate_stochastic(carry, namelist=None, *, required=False):
    config = (namelist.get("stochastic_config") if isinstance(namelist,Mapping)
              else getattr(namelist,"stochastic_config",None))
    if carry is None:
        if required or config is not None:
            raise ValueError("restart lacks active stochastic spectral state/seeds (E78)")
        return
    from gpuwrf.physics.stochastic_driver import StochasticCarry
    if not isinstance(carry,StochasticCarry) or int(np.asarray(carry.schema)) != 1:
        raise ValueError("stochastic restart schema mismatch (E78)")
    if len(carry.patterns) != 7 or len(carry.seeds) != 6:
        raise ValueError("stochastic restart pattern/seed layout mismatch (E78)")
    clips = np.asarray(carry.clip_counts)
    if clips.shape != (7,) or clips.dtype != np.int32 or np.any(clips<0):
        raise ValueError("stochastic restart event counters invalid (E78)")
    expected_seeds = (carry.patterns[0] is not None or carry.patterns[1] is not None,
                      *(p is not None for p in carry.patterns[2:]))
    if tuple(s is not None for s in carry.seeds) != expected_seeds:
        raise ValueError("stochastic restart missing an active RNG stream (E78)")
    if config is not None:
        if tuple(p is not None for p in carry.patterns) != tuple(p is not None for p in config.patterns):
            raise ValueError("stochastic restart active patterns differ from namelist (E78)")
    for seed in carry.seeds:
        if seed is not None and (np.asarray(seed).shape != (8,) or np.asarray(seed).dtype != np.int32):
            raise ValueError("stochastic restart seed words must be eight int32 values (E78)")
    for index, p in enumerate(carry.patterns):
        if p is None:
            continue
        if p.sine.shape != p.cosine.shape or p.sine.shape != p.amplitude.shape:
            raise ValueError("stochastic restart coefficient shape mismatch (E78)")
        values = p.field if index == 0 else (p.field,)
        if (len(values)!=(2 if index==0 else 1) or any(np.ndim(v)!=3 for v in values)
                or any(np.shape(v)!=np.shape(values[0]) for v in values)
                or np.shape(p.alpha)!=() or np.shape(p.cutoff)!=()):
            raise ValueError("stochastic restart field/parameter rank mismatch (E78)")
        if config is not None:
            root = config.root and config.produce[index]
            if (root and p.sine.shape != values[0].shape[1:]) or (not root and p.sine.shape != (0,0)):
                raise ValueError("stochastic restart root/child coefficient layout differs (E78)")
            if bool(config.patterns[index].vertical) != (values[0].shape[0]>1):
                raise ValueError("stochastic restart vertical structure differs (E78)")
        arrays = (p.sine,p.cosine,p.amplitude,p.alpha,p.cutoff,
                  *(p.field if index == 0 else (p.field,)))
        if any(np.asarray(a).dtype != np.float32 or not np.isfinite(np.asarray(a)).all() for a in arrays):
            raise ValueError("stochastic restart contains nonfinite/non-REAL fields (E78)")
    # Shared vertical plans are essential live state too. A child or a root
    # with only constant-vertical producers legitimately has no such plan.
    vertical_shapes=[np.shape(p.field[0] if index==0 else p.field)
        for index,p in enumerate(carry.patterns) if p is not None and p.sine.shape!=(0,0)
        and np.shape(p.field[0] if index==0 else p.field)[0]>1]
    if vertical_shapes:
        shape=vertical_shapes[0]
        if (carry.vertical is None or len(carry.vertical)!=3
                or any(s!=shape for s in vertical_shapes)
                or tuple(np.shape(a) for a in carry.vertical)!=(shape,shape,(shape[0],))):
            raise ValueError("stochastic restart missing/mismatched vertical plan (E78)")
        if any(np.asarray(a).dtype!=np.float32 or not np.isfinite(np.asarray(a)).all()
               for a in carry.vertical):
            raise ValueError("stochastic restart nonfinite/non-REAL vertical plan (E78)")
    elif carry.vertical is not None:
        raise ValueError("stochastic restart unexpected vertical plan (E78)")


def seed_arrays(carry):
    return {name:seed for name,seed in zip(SEED_NAMES,carry.seeds) if seed is not None}
