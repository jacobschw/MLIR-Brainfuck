#ifndef BF_RED_OPS
#define BF_RED_OPS

include "Bf/Dialect/bf_red/IR/BfRedBase.td"

def Bf_Increment : BfRed_Op<"increment"> {
    let summary = "Increments the current cell by amount.";

    let arguments = (ins SI8Attr:$amount);

    let assemblyFormat = "attr-dict";
}

def Bf_Shift : BfRed_Op<"shift"> {
    let summary = "Shift the pointer by value to the right if value > 0 and to the left else.";

    let arguments = (ins SI32Attr:$value);

    let assemblyFormat = "attr-dict";
}


#endif // BF_RED_OPS