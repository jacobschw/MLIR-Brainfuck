// RUN: Bf-opt %s | Bf-opt | FileCheck %s

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// CHECK-LABEL:   bf_red.increment {amount = 1 : si8}
// CHECK:         bf_red.shift {value = 1 : si32}
// CHECK:         bf_red.increment {amount = -1 : si8}
// CHECK:         bf_red.shift {value = -1 : si32}

builtin.module {    

    bf_red.increment {amount = 1 : si8}

    bf_red.shift {value = 1 : si32}

    bf_red.increment {amount = -1 : si8}

    bf_red.shift {value = -1 : si32}

}