#include "Bf/Dialect/bf/IR/BfOps.h"
#include "Bf/Dialect/bf/IR/BfDialect.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/IR/RegionKindInterface.h"

//===----------------------------------------------------------------------===//
// Loop
//===----------------------------------------------------------------------===//

using namespace mlir::bf;

mlir::RegionKind Loop::getRegionKind(unsigned index) {
  return mlir::RegionKind::SSACFG;
}

//===----------------------------------------------------------------------===//
// Module
//===----------------------------------------------------------------------===//

using namespace mlir::bf;

mlir::RegionKind Module::getRegionKind(unsigned index) {
  return mlir::RegionKind::SSACFG;
}

#define GET_OP_CLASSES
#include "Bf/Dialect/bf/IR/BfOps.cpp.inc"

