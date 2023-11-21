
#ifndef Bf_OPS
#define Bf_OPS

include "Bf/Dialect/bf/IR/BfDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

def Bf_Increment : Bf_PrimitiveOp<"increment"> {
    let summary = "Increments the current cell.";
    let assemblyFormat = "attr-dict";
}

def Bf_Decrement : Bf_PrimitiveOp<"decrement"> {
    let summary = "Decrements the current cell.";
    let assemblyFormat = "attr-dict";
}

def Bf_ShiftRight : Bf_PrimitiveOp<"shift_right"> {
    let summary = "Shift the pointer right.";
    let assemblyFormat = "attr-dict";
}

def Bf_ShiftLeft : Bf_PrimitiveOp<"shift_left"> {
    let summary = "Shift the pointer left.";
    let assemblyFormat = "attr-dict";
}

def Bf_Output : Bf_PrimitiveOp<"output"> {
    let summary = "Outputs the current cell in ASCI format.";
    let assemblyFormat = "attr-dict";
}


def Bf_Input : Bf_PrimitiveOp<"input"> {
    let summary = "Input an ASCI value and write it to the current cell.";
    let assemblyFormat = "attr-dict";
}

def Bf_Module : Bf_ComposedOp<"module", [
    IsolatedFromAbove, DeclareOpInterfaceMethods<RegionKindInterface>
    ] # GraphRegionNoTerminator.traits> {
    let summary = "A Brainfuck Module serving as root.";
    
    let regions = (region SizedRegion<1>:$body); 

    let assemblyFormat = "attr-dict-with-keyword $body";
}

def Bf_Loop : Bf_ComposedOp<"loop", [DeclareOpInterfaceMethods<RegionKindInterface>] 
    # GraphRegionNoTerminator.traits> {
    let summary = "Brainfuck loop in respect to the Brainfuck loop condition: current cell != 0.";
    
    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = "attr-dict-with-keyword $body";
}

#endif // Bf_OPS
