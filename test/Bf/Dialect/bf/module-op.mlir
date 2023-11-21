// RUN: Bf-opt %s | Bf-opt | FileCheck %s


// CHECK: Bf.module {
Bf.module {
}

// -----

// CHECK: Bf.module {
Bf.module {
    // CHECK-NEXT: Bf.increment
    Bf.increment
}
