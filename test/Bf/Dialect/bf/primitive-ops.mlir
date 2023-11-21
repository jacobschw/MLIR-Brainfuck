// RUN: Bf-opt %s | Bf-opt | FileCheck %s


// CHECK: Bf.increment
Bf.increment

// -----

// CHECK: Bf.decrement
Bf.decrement

// -----

// CHECK: Bf.shift_right
Bf.shift_right

// -----

// CHECK: Bf.shift_left
Bf.shift_left

// -----

// CHECK: Bf.input
Bf.input

// -----

// CHECK: Bf.output
Bf.output


