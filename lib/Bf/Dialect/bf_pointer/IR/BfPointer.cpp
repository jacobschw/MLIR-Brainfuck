#include "Bf/Dialect/bf_pointer/IR/BfPointer.h"
#include "Bf/Dialect/bf_pointer/IR/BfPointerOps.h"

using namespace mlir;
using namespace mlir::bf_pointer;

#include "Bf/Dialect/bf_pointer/IR/BfPointerOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// BfPointer dialect.
//===----------------------------------------------------------------------===//

void BfPointerDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Bf/Dialect/bf_pointer/IR/BfPointerOps.cpp.inc"
      >();
}