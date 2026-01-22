import importlib.util
import os

if os.name == "nt":
    spec = importlib.util.find_spec("paddle")
    if spec and spec.submodule_search_locations:
        libs_dir = os.path.join(spec.submodule_search_locations[0], "libs")
        if os.path.isdir(libs_dir):
            os.add_dll_directory(libs_dir)
