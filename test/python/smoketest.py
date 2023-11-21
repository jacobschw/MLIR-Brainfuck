# RUN: %python %s | FileCheck %s

from mlir_Bf.ir import *
from mlir_Bf.dialects import builtin as builtin_d, Bf as Bf_d

with Context():
    Bf_d.register_dialect()
    module = Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = Bf.foo %0 : i32
    """
    )
    # CHECK: %[[C:.*]] = arith.constant 2 : i32
    # CHECK: Bf.foo %[[C]] : i32
    print(str(module))
