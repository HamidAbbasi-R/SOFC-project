def import_module(module_name):
    import importlib.util
    import sys
    import os

    file_path = f'c:\\Users\\{os.getlogin()}\\OneDrive - The University of Manchester\\SOFC\\Codes\\{module_name}.py'
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module