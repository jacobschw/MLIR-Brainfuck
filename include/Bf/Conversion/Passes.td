#ifndef BF_CONVERSION_TD
#define BF_CONVERSION_TD

include "mlir/Pass/PassBase.td"

//===
// BfToOptBf
//===

def BfToOptBf : Pass<"bf-to-optbf"> {
  let summary = "Lowering of bf to optbf.";

  let description = [{
    This lowering reduces 
      'Bf.increment' and 'Bf.decrement' to 'bf_red.increment {amount = +-1 : si8}'
    and 
      'Bf.shift_right' and 'Bf_red.shift_left' to 'bf_red.shift {value = +-1 : si32}'.
  }];

  let dependentDialects = ["mlir::bf_red::BfRedDialect"];
}

//===
// BfToExplicitBf
//===

def BfOptToExplicitBf : Pass <"optbf-to-explicitbf"> {
  let summary = "Lowering of the OpfBt to ExplicitBf.";

  let description = [{
    The --optbf-to-explicitbf pass lowers OpfBf to ExplicitBf. This makes explicit the use of Brainfuck memory, pointer and the console.
    The Braifuck memory is modelled as a <30000, i8> shaped memref. The pointer is modelled by the bf_pointer dialect of the MLIR Brainfuck
    project. The console is modelled by declaration and application of the C language functions getchar and putchar.
  }];

  let dependentDialects = ["mlir::LLVM::LLVMDialect", "mlir::func::FuncDialect", "mlir::bf_red::BfRedDialect", "mlir::bf_pointer::BfPointerDialect", "mlir::memref::MemRefDialect", "mlir::index::IndexDialect", "mlir::arith::ArithDialect", "mlir::scf::SCFDialect","mlir::BuiltinDialect"];
}

//===
// ExplicitBfToLLVM
//===

def ExplicitBfToLLVM : Pass<"explicitbf-to-llvm"> {
  let summary = "Lowering of the ExplicitBf to llvm.";

  let description = [{
    The --explicitbf-to-llvm pass lowers to ExplicitBf the llvm dialect. This includes custom conversion of the bf_pointer operations.
    Conversion of all other constructs of the dialects used in ExplicitBf is enabled by rewrite patterns contained in the MLIR ecosystem.
  }];

  let dependentDialects = ["mlir::bf_red::BfRedDialect", "mlir::func::FuncDialect", "mlir::bf_pointer::BfPointerDialect", 
    "mlir::memref::MemRefDialect", 
    "mlir::index::IndexDialect", "mlir::arith::ArithDialect", 
    "mlir::scf::SCFDialect", "mlir::cf::ControlFlowDialect",
    "mlir::BuiltinDialect", "mlir::LLVM::LLVMDialect"];
}


#endif // BF_CONVERSION_TD