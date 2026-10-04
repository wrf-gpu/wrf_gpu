"""Fused physics attribution must retain ambiguity and caller provenance."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('b_phys_kernel_map', Path(__file__).with_name('hlo_kernel_map.py'))
mapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mapper)


def test_generic_math_frame_resolves_to_calling_scheme():
    tabs = ({1: '/jax/numpy.py', 2: '/src/gpuwrf/physics/mynn_edmf.py'}, {},
            {1: {'file_name_id': '1'}, 2: {'file_name_id': '2'}},
            {1: {'file_location_id': '1', 'parent_frame_id': '2'},
             2: {'file_location_id': '2', 'parent_frame_id': '0'}})
    assert mapper.metadata_family({'frame': 1, 'line': ''}, tabs) == 'MYNN'


def test_exact_fusion_names_and_mixed_family_disclosure(tmp_path):
    path = tmp_path / 'module.after_optimizations.txt'
    path.write_text('''HloModule sample

%fused_computation.1 (a: f32[4]) -> f32[4] {
  %a = f32[4] parameter(0)
  ROOT %x = f32[4] add(%a, %a), metadata={source_file="/src/gpuwrf/physics/mynn_edmf.py"}
}

%fused_computation.2 (a: f32[4]) -> f32[4] {
  %a = f32[4] parameter(0)
  %x = f32[4] add(%a, %a), metadata={source_file="/src/gpuwrf/physics/mynn_edmf.py"}
  ROOT %y = f32[4] multiply(%x, %a), metadata={source_file="/src/gpuwrf/physics/thompson_column.py"}
}

ENTRY %main (a: f32[4]) -> f32[4] {
  %a = f32[4] parameter(0)
  %loop_multiply_fusion.63 = f32[4] fusion(%a), kind=kLoop, calls=%fused_computation.1
  ROOT %loop_add_fusion.7 = f32[4] fusion(%a), kind=kLoop, calls=%fused_computation.2
}
''')
    result = mapper.extract(path)
    assert result['kernels']['loop_multiply_fusion_63'] == 'MYNN'
    assert result['kernels']['loop_add_fusion_7'] == 'MIXED'
