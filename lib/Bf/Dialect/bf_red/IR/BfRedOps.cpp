#include "Bf/Dialect/bf_red/IR/BfRedOps.h"
#include "Bf/Dialect/bf_red/IR/BfRed.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"


#define GET_OP_CLASSES
#include "Bf/Dialect/bf_red/IR/BfRedOps.cpp.inc"