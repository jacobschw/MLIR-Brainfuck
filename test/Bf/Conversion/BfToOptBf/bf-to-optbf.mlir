// RUN: Bf-opt %s --bf-to-optbf > %t
// RUN: FileCheck %s < %t

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// The script is designed to make adding checks to
// a test case fast, it is *not* designed to be authoritative
// about what constitutes a good test! The CHECK should be
// minimized and named to reflect the test intent.


// CHECK-LABEL:   bf_red.shift {value = 1 : si32}
// CHECK:         bf_red.shift {value = -1 : si32}
// CHECK:         bf_red.increment {amount = 1 : si8}
// CHECK:         bf_red.increment {amount = -1 : si8}

Bf.shift_right

//---

Bf.shift_left

//---

Bf.increment

//---

Bf.decrement