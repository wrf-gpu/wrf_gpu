"""Full HloModuleProto hashing, ignoring only the documented replay module ID.

Schema: https://github.com/openxla/xla/blob/main/xla/service/hlo.proto
HloModuleProto.id = 5 is ignored on replay. All other bytes, including literal
values, instruction/computation IDs, backend config and source locations stay.
"""
import hashlib


def fields(proto):
    def varint(pos):
        value = shift = 0
        while True:
            byte = proto[pos]
            pos += 1
            value |= (byte & 127) << shift
            if not byte & 128:
                return value, pos
            shift += 7
            if shift > 63:
                raise ValueError('invalid protobuf varint')
    pos = 0
    while pos < len(proto):
        start = pos
        tag, pos = varint(pos)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, pos = varint(pos)
        elif wire in (1, 5):
            pos += 8 if wire == 1 else 4
            value = None
        elif wire == 2:
            size, pos = varint(pos)
            value = memoryview(proto)[pos:pos + size]
            pos += size
        else:
            raise ValueError(f'unsupported protobuf wire type {wire}')
        assert number and pos <= len(proto)
        yield number, wire, value, memoryview(proto)[start:pos]


def digest(proto):
    h = hashlib.sha256()
    module_ids = []
    for number, wire, value, raw in fields(proto):
        if number == 5:
            assert wire == 0
            module_ids.append(value)
        else:
            h.update(raw)
    assert len(module_ids) == 1
    return h.hexdigest(), module_ids[0]
