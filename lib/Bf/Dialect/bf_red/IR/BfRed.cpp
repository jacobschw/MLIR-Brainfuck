#include "Bf/Dialect/bf_red/IR/BfRed.h"
#include "Bf/Dialect/bf_red/IR/BfRedOps.h"

using namespace mlir;
using namespace mlir::bf_red;

#include "Bf/Dialect/bf_red/IR/BfRedOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// BfRed dialect.
//===----------------------------------------------------------------------===//

void BfRedDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Bf/Dialect/bf_red/IR/BfRedOps.cpp.inc"
      >();
}