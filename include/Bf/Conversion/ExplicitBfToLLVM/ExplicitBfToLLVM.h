#ifndef CONVERSION_BF_TO_LLVM_
#define CONVERSION_BF_TO_LLVM_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace bf {
namespace lowerings {

#define GEN_PASS_DECL_EXPLICITBFTOLLVM
#include "Bf/Conversion/Passes.h.inc"


} // namespace lowerings
} // namespace Bf
} // namespace mlir



#endif //CONVERSION_BF_TO_DIALECT_MIX_