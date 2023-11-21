// RUN: Bf-opt %s | Bf-opt | FileCheck %s

// CHECK: Bf.loop {
Bf.loop {

}

// CHECK: Bf.loop {
Bf.loop {
    // CHECK-NEXT: Bf.decrement
    Bf.decrement
}
