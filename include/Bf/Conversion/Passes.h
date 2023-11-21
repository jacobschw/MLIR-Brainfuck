#ifndef BF_CONVERSION_H_
#define BF_CONVERSION_H_

#include "Bf/Dialect/bf/IR/BfDialect.h"
#include "Bf/Dialect/bf/IR/BfOps.h"

#include "Bf/Conversion/BfToOptBf/BfToOptBf.h"
#include "Bf/Conversion/BfOptToExplicitBf/BfOptToExplicitBf.h"
#include "Bf/Conversion/ExplicitBfToLLVM/ExplicitBfToLLVM.h"

namespace mlir {
namespace bf {
namespace lowerings {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Bf/Conversion/Passes.h.inc"

}  // namespace lowerings
}  // namespace Bf
}  // namespace mlir

#endif  // BF_CONVERSION_H_