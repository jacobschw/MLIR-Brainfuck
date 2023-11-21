
#ifndef BF__RED_OPS_H
#define BF__RED_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"


#define GET_OP_CLASSES
#include "Bf/Dialect/bf_red/IR/BfRedOps.h.inc"

#endif // 