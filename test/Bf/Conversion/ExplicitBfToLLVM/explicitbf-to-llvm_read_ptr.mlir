// RUN: Bf-opt %s --explicitbf-to-llvm > %t
// RUN: FileCheck %s < %t

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// The script is designed to make adding checks to
// a test case fast, it is *not* designed to be authoritative
// about what constitutes a good test! The CHECK should be
// minimized and named to reflect the test intent.


// CHECK-LABEL:   llvm.mlir.global private @bf_ptr(0 : index) {addr_space = 0 : i32} : i64
// CHECK:         %[[VAL_0:.*]] = llvm.mlir.addressof @bf_ptr : !llvm.ptr<i64>
// CHECK:         %[[VAL_1:.*]] = llvm.load %[[VAL_0]] : !llvm.ptr<i64>

bf_pointer.ptr @bf_ptr = 0 : (index) -> ()

bf_pointer.read_ptr {name = @bf_ptr} : () -> index 