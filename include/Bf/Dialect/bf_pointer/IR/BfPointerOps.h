
#ifndef BF_POINTER_OPS_H
#define BF_POINTER_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/SymbolTable.h"

#define GET_OP_CLASSES
#include "Bf/Dialect/bf_pointer/IR/BfPointerOps.h.inc"

#endif // BF_POINTER_OPS_H