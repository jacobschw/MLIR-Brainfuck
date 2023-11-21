# Autogenerated from C:/Projects/2023/info-ba/Sourcen/mlir_bf/test/lit.site.cfg.py.in
# Do not edit!

# Allow generated file to be relocatable.
import os
import platform
def path(p):
    if not p: return ''
    # Follows lit.util.abs_path_preserve_drive, which cannot be imported here.
    if platform.system() == 'Windows':
        return os.path.abspath(os.path.join(os.path.dirname(__file__), p))
    else:
        return os.path.realpath(os.path.join(os.path.dirname(__file__), p))


config.llvm_tools_dir = lit_config.substitute("C:/llvm-project/llvm-project/build/%(build_mode)s/bin")
config.mlir_obj_dir = "C:/Projects/2023/info-ba/Sourcen/mlir_bf"
config.enable_bindings_python = 0
config.Bf_obj_root = "C:/Projects/2023/info-ba/Sourcen/mlir_bf"
config.llvm_shlib_ext = ".dll"

import lit.llvm
lit.llvm.initialize(lit_config, config)

# Let the main config do the real work.
lit_config.load_config(config, "C:/Projects/2023/info-ba/Sourcen/mlir_bf/test/lit.cfg.py")