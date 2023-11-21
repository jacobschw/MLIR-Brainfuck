
#ifndef Bf_BfOPS_H
#define Bf_BfOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/IR/RegionKindInterface.h"

#define GET_OP_CLASSES
#include "Bf/Dialect/bf/IR/BfOps.h.inc"

#endif // Bf_BfOPS_H
