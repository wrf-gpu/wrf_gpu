"""Semantic host-only negative control: remove the input read-only mount."""


def pytest_collection_modifyitems(items):
    for item in items:
        if item.name != 'test_read_only_mount_prevents_actual_child_write':
            continue
        module = item.module
        original = module.control.model_command

        def without_readonly(*args, **kwargs):
            command = original(*args, **kwargs)
            index = next(i for i, value in enumerate(command)
                         if value == '--readonly-dir' and command[i + 1] == str(module.control.READ_ONLY_ROOT))
            return command[:index] + command[index + 2:]

        module.control.model_command = without_readonly
