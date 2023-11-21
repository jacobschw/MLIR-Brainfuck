#include "Bf/Conversion/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "Bf/Dialect/bf_red/IR/BfRed.h"
#include "Bf/Dialect/bf_red/IR/BfRedOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "Bf/Dialect/bf_pointer/IR/BfPointer.h"
#include "Bf/Dialect/bf_pointer/IR/BfPointerOps.h"
#include "Bf/Dialect/bf_red/IR/BfRed.h"
#include "Bf/Dialect/bf_red/IR/BfRedOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir {
namespace bf {
namespace lowerings {
#define GEN_PASS_DEF_EXPLICITBFTOLLVM
#include "Bf/Conversion/Passes.h.inc"

// Lower the 'bf_pointer.ptr' operation to 'llvm.mlir.global'.
struct PtrLowering : public mlir::ConversionPattern {
       PtrLowering(mlir::MLIRContext *ctx)
        : mlir::ConversionPattern(mlir::bf_pointer::ptr::getOperationName(), 1, ctx) {}
    
    /// Match and rewrite the given `bf_pointer.ptr` operation.
    mlir::LogicalResult matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const override {
        auto ptrOp = llvm::dyn_cast<mlir::bf_pointer::ptr>(op);
        if (!ptrOp) {
            return failure();
        }

        auto loc = op->getLoc();

        auto init = ptrOp.getInitialValueAttr();
        LLVMTypeConverter typeConverter(getContext());

        auto res = rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(ptrOp, 
            typeConverter.getIndexType(), 
            false, 
            mlir::LLVM::linkage::Linkage(0), 
            ptrOp.getSymName(), 
            rewriter.getIndexAttr(init.getValue().getZExtValue()));

        return success();
    };  
};

// Lower the 'bf_pointer.read_ptr' operation to 'llvm.mlir.adressof' and 'llvm_load'.
struct ReadPtrLowering : public mlir::ConversionPattern {
       ReadPtrLowering(mlir::MLIRContext *ctx)
        : mlir::ConversionPattern(mlir::bf_pointer::read_ptr::getOperationName(), 1, ctx) {}
    
    /// Match and rewrite the given `bf_pointer.read_ptr` operation.
    mlir::LogicalResult matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const override {
        auto readPtrOp = llvm::dyn_cast<mlir::bf_pointer::read_ptr>(op);
        if (!readPtrOp) {
            return failure();
        }

        auto loc = op->getLoc();

        LLVMTypeConverter typeConverter(getContext());

        Value address = rewriter.create<mlir::LLVM::AddressOfOp>(loc, LLVM::LLVMPointerType::get(typeConverter.getIndexType()), readPtrOp.getName());

        auto load = rewriter.create<mlir::LLVM::LoadOp>(loc, address);

        rewriter.replaceOp(op, load);

        return success();
    };  
};

// Lower the 'bf_pointer.write_ptr' operation to 'llvm.mlir.adressof' and 'llvm_store'.
struct WritePtrLowering : public mlir::ConversionPattern {
       WritePtrLowering(mlir::MLIRContext *ctx)
        : mlir::ConversionPattern(mlir::bf_pointer::write_ptr::getOperationName(), 1, ctx) {}
    
    /// Match and rewrite the given `bf_pointer.write_ptr` operation.
    mlir::LogicalResult matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const override {
        auto writePtrOp = llvm::dyn_cast<mlir::bf_pointer::write_ptr>(op);
        if (!writePtrOp) {
            return failure();
        }

        auto loc = op->getLoc();

        LLVMTypeConverter typeConverter(getContext());

        auto convertedType =  typeConverter.getIndexType();

        Value address = rewriter.create<mlir::LLVM::AddressOfOp>(loc, LLVM::LLVMPointerType::get(convertedType), writePtrOp.getName());

        Value index = writePtrOp.getNptr();
        Value casted_index = rewriter.create<mlir::arith::IndexCastOp>(loc, mlir::TypeRange(convertedType), index);

        auto store = rewriter.create<mlir::LLVM::StoreOp>(loc, casted_index, address);

        rewriter.replaceOp(op, store);

        return success();
    };  
};

struct ExplicitBfToLLVM : impl::ExplicitBfToLLVMBase<ExplicitBfToLLVM> {
    using ExplicitBfToLLVMBase::ExplicitBfToLLVMBase;
 
    void runOnOperation() {
        mlir::ConversionTarget target(getContext());

        target.addLegalDialect<mlir::LLVM::LLVMDialect>();

        target.addIllegalDialect<mlir::bf::BfDialect, mlir::bf_red::BfRedDialect, mlir::bf_pointer::BfPointerDialect, 
            mlir::memref::MemRefDialect,
            mlir::index::IndexDialect, mlir::arith::ArithDialect,
            mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect, 
            mlir::BuiltinDialect, mlir::func::FuncDialect>();

        target.addLegalOp<mlir::ModuleOp>();

        mlir::RewritePatternSet patterns(&getContext());

        populateSCFToControlFlowConversionPatterns(patterns);

        // During this lowering, we will also be lowering the MemRef types, that are
        // currently being operated on, to a representation in LLVM. To perform this
        // conversion we use a TypeConverter as part of the lowering. This converter
        // details how one type maps to another. This is necessary now that we will be
        // doing more complicated lowerings, involving loop region arguments.
        LLVMTypeConverter typeConverter(&getContext());

        mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
        mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

        mlir::index::populateIndexToLLVMConversionPatterns(typeConverter, patterns);
        mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
        populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);

        patterns.add<PtrLowering, ReadPtrLowering, WritePtrLowering>(&getContext());

        if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
            signalPassFailure();
    };
};

} // namespac lowerings
} // namespace Bf
} // namespac mlir