// RUN: python C:\Projects\2023\info-ba\Sourcen\mlir_bf\tools\bf-to-mlir_bf.py translate bf-to-mlir-bf --output-path="" %s > %t
// RUN: Bf-opt %t --bf-to-llvm > %t1
// RUN: Bf-translate %t1 --mlir-to-llvmir > %t2
// RUN: clang %t2 > %t3
// RUN: %t3 > %t4
// RUN: FileCheck %t4 < hello_world.sem


>+++++++++[<++++++++>-]<.>+++++++[<++++>-]<+.+++++++..+++.[-]
>++++++++[<++++>-] <.>+++++++++++[<++++++++>-]<-.--------.+++
.------.--------.[-]>++++++++[<++++>- ]<+.[-]++++++++++.