// RUN: Bf-opt %s --explicitbf-to-llvm > %t
// RUN: FileCheck %s < %t

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// The script is designed to make adding checks to
// a test case fast, it is *not* designed to be authoritative
// about what constitutes a good test! The CHECK should be
// minimized and named to reflect the test intent.


// CHECK-LABEL:   llvm.mlir.global private @bf_ptr(0 : index) {addr_space = 0 : i32} : i64
// CHECK:         %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[VAL_1:.*]] = llvm.mlir.addressof @bf_ptr : !llvm.ptr<i64>
// CHECK:         %[[VAL_2:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         llvm.store %[[VAL_2]], %[[VAL_1]] : !llvm.ptr<i64>

bf_pointer.ptr @bf_ptr = 0 : (index) -> ()

%value = index.constant 1

bf_pointer.write_ptr %value {name = @bf_ptr} : (index) -> ()