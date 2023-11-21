
#include "Bf/Dialect/bf/IR/BfDialect.h"
#include "Bf/Dialect/bf/IR/BfOps.h"

using namespace mlir;
using namespace mlir::bf;

#include "Bf/Dialect/bf/IR/BfOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Bf dialect.
//===----------------------------------------------------------------------===//

void BfDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Bf/Dialect/bf/IR/BfOps.cpp.inc"
      >();
}
