// RUN: Bf-opt %s | Bf-opt | FileCheck %s

builtin.module {

    bf_pointer.ptr @bf_ptr = 0 : (index) -> () 

    // CHECK: %[[VAL_2:.*]] = bf_pointer.read_ptr
    %2 = bf_pointer.read_ptr {name = @bf_ptr} : () -> (index)

    // -----

    // CHECK: bf_pointer.write_ptr
    bf_pointer.write_ptr %2 {name = @bf_ptr}: (index) -> ()


}
