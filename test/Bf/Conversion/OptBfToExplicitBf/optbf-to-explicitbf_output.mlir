// RUN: Bf-opt %s --optbf-to-explicitbf > %t
// RUN: FileCheck %s < %t

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// The script is designed to make adding checks to
// a test case fast, it is *not* designed to be authoritative
// about what constitutes a good test! The CHECK should be
// minimized and named to reflect the test intent.


// CHECK-LABEL:   bf_pointer.ptr @bf_ptr = 0 : () -> ()
// CHECK:         memref.global "private" @bf_memory : memref<30000xi8>
// CHECK:         func.func private @getchar() -> i32
// CHECK:         func.func private @putchar(i32) -> i32
// CHECK:         %[[VAL_0:.*]] = memref.get_global @bf_memory : memref<30000xi8>
// CHECK:         %[[VAL_1:.*]] = bf_pointer.read_ptr {name = @bf_ptr} : () -> index
// CHECK:         %[[VAL_2:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_1]]] : memref<30000xi8>
// CHECK:         %[[VAL_3:.*]] = llvm.sext %[[VAL_2]] : i8 to i32
// CHECK:         %[[VAL_4:.*]] = func.call @putchar(%[[VAL_3]]) : (i32) -> i32

bf_pointer.ptr @bf_ptr = 0 : () -> ()
memref.global "private" @bf_memory : memref<30000xi8>
func.func private @getchar() -> i32
func.func private @putchar(i32) -> i32

Bf.output