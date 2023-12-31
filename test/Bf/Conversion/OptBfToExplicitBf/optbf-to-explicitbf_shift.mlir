// RUN: Bf-opt %s --optbf-to-explicitbf > %t
// RUN: FileCheck %s < %t

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// The script is designed to make adding checks to
// a test case fast, it is *not* designed to be authoritative
// about what constitutes a good test! The CHECK should be
// minimized and named to reflect the test intent.


// CHECK-LABEL:   bf_pointer.ptr @bf_ptr = 0 : () -> ()
// CHECK:         %[[VAL_0:.*]] = index.constant 0
// CHECK:         %[[VAL_1:.*]] = bf_pointer.read_ptr {name = @bf_ptr} : () -> index
// CHECK:         %[[VAL_2:.*]] = index.constant 1
// CHECK:         %[[VAL_3:.*]] = index.sub %[[VAL_1]], %[[VAL_2]]
// CHECK:         bf_pointer.write_ptr %[[VAL_3]] {name = @bf_ptr} : (index) -> ()
// CHECK:         %[[VAL_4:.*]] = bf_pointer.read_ptr {name = @bf_ptr} : () -> index
// CHECK:         %[[VAL_5:.*]] = index.constant 1
// CHECK:         %[[VAL_6:.*]] = index.add %[[VAL_4]], %[[VAL_5]]
// CHECK:         bf_pointer.write_ptr %[[VAL_6]] {name = @bf_ptr} : (index) -> ()

bf_pointer.ptr @bf_ptr = 0 : (index) -> () 
%0 = index.constant 0

bf_red.shift {value = -1 : si32}

//---

bf_red.shift {value = 1 : si32}