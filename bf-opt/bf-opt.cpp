//===- Bf-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Bf/Dialect/bf/IR/BfDialect.h"
#include "Bf/Dialect/bf_pointer/IR/BfPointer.h"
#include "Bf/Dialect/bf_red/IR/BfRed.h"
#include "Bf/Conversion/Passes.h"

void bfToLLVMPipeline(mlir::OpPassManager &manager) {
  manager.addPass(mlir::bf::lowerings::createBfToOptBf());
  manager.addPass(mlir::bf::lowerings::createBfOptToExplicitBf());
  manager.addPass(mlir::bf::lowerings::createExplicitBfToLLVM());
}

int main(int argc, char **argv) {
  mlir::bf::lowerings::registerPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::bf::BfDialect,
                  mlir::bf_pointer::BfPointerDialect,
                  mlir::bf_red::BfRedDialect,
                  mlir::arith::ArithDialect, 
                  mlir::func::FuncDialect,
                  mlir::scf::SCFDialect, 
                  mlir::memref::MemRefDialect,
                  mlir::LLVM::LLVMDialect,
                  mlir::index::IndexDialect>();

  mlir::PassPipelineRegistration<>("bf-to-llvm",
                             "Run passes to lower the Bf dialect to LLVM",
                             bfToLLVMPipeline);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Bf optimizer driver\n", registry));
}
