#ifndef BF_POINTER_OPS
#define BF_POINTER_OPS

include "Bf/Dialect/bf_pointer/IR/BfPointerBase.td"
include "mlir/IR/SymbolInterfaces.td"

def Bf_ptr : BfPointer_Op<"ptr", [Symbol]> {
    let summary = "Define the Brainfuck pointer as a global variable.";
    
    let arguments = (ins SymbolNameAttr:$sym_name,
        DefaultValuedAttr<IndexAttr, "0">:$initial_value
    );

    let assemblyFormat = "$sym_name `=` $initial_value `:` functional-type(operands, results) attr-dict";
}

def Bf_read_ptr : BfPointer_Op<"read_ptr", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Read the current value of the Brainfuck pointer of index type.";

    let arguments = (ins FlatSymbolRefAttr:$name);
    let results = (outs Index:$ptr);

    let assemblyFormat = "attr-dict `:` functional-type(operands, results)";
}

def Bf_write_ptr : BfPointer_Op<"write_ptr", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Write a new value of index type to the Brainfuck pointer.";

    let arguments = (ins FlatSymbolRefAttr:$name,
        Index:$nptr 
    );

    let assemblyFormat = "$nptr attr-dict `:` functional-type(operands, results)";
}

#endif // BF_POINTER_OPS