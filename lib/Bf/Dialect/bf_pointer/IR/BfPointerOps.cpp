#include "Bf/Dialect/bf_pointer/IR/BfPointerOps.h"
#include "Bf/Dialect/bf_pointer/IR/BfPointer.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/SymbolTable.h"

//===----------------------------------------------------------------------===//
// read_ptr
//===----------------------------------------------------------------------===//

using namespace mlir::bf_pointer;

mlir::LogicalResult
read_ptr::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verify that the result type is same as the type of the referenced
  // memref.global op.
  auto bf_ptr =
      symbolTable.lookupNearestSymbolFrom<ptr>(*this, getNameAttr());
  if (!bf_ptr)
    return emitOpError("'")
           << getName() << "' does not reference a valid global bf ptr";

           
  return success();
}

//===----------------------------------------------------------------------===//
// write_ptr
//===----------------------------------------------------------------------===//

using namespace mlir::bf_pointer;

mlir::LogicalResult
write_ptr::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verify that the result type is same as the type of the referenced
  // memref.global op.
  auto bf_ptr =
      symbolTable.lookupNearestSymbolFrom<ptr>(*this, getNameAttr());
  if (!bf_ptr)
    return emitOpError("'")
           << getName() << "' does not reference a valid global bf ptr";

           
  return success();
}
#define GET_OP_CLASSES
#include "Bf/Dialect/bf_pointer/IR/BfPointerOps.cpp.inc"