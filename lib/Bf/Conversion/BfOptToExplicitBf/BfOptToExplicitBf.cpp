#include "Bf/Conversion/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "Bf/Dialect/bf_pointer/IR/BfPointer.h"
#include "Bf/Dialect/bf_pointer/IR/BfPointerOps.h"
#include "Bf/Dialect/bf_red/IR/BfRed.h"
#include "Bf/Dialect/bf_red/IR/BfRedOps.h"

namespace mlir {
namespace bf {
namespace lowerings {
#define GEN_PASS_DEF_BFOPTTOEXPLICITBF
#include "Bf/Conversion/Passes.h.inc"


/// Lower `bf.module` to `builtin.module`, `memref.global` and `something with a symbol`.
struct BfModuleLowering : public mlir::ConversionPattern {
    BfModuleLowering(mlir::MLIRContext *ctx)
        : mlir::ConversionPattern(mlir::bf::Module::getOperationName(), 1, ctx) {}
    
    /// Match and rewrite the given `bf_pointer.read_memory` operation.
    mlir::LogicalResult matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const override {
        auto bfModuleOp = llvm::dyn_cast<mlir::bf::Module>(op);
        if (!bfModuleOp) {
            return failure();
        }

        auto loc = op->getLoc();
        auto integerType = rewriter.getI8Type();

        // create the brainfuck pointer global
        auto bfPtr = rewriter.create<mlir::bf_pointer::ptr>(loc, rewriter.getStringAttr("bf_ptr"), rewriter.getIndexAttr(0));


        // create the brainfuck memory global
        auto memrefType = MemRefType::get(30000, integerType);
        auto memory = rewriter.create<mlir::memref::GlobalOp>(loc, "bf_memory", 
            rewriter.getStringAttr("private"), 
            memrefType, 
            Attribute(),
            false,
            IntegerAttr()
        );

        auto consoleType = rewriter.getI32Type();
        rewriter.create<mlir::func::FuncOp>(loc, "getchar", 
            rewriter.getFunctionType(mlir::TypeRange(), consoleType),
            rewriter.getStringAttr("private"),
            mlir::ArrayAttr(),
            mlir::ArrayAttr()
        );

        rewriter.create<mlir::func::FuncOp>(loc, "putchar", 
            rewriter.getFunctionType(consoleType, consoleType),
            rewriter.getStringAttr("private"),
            mlir::ArrayAttr(),
            mlir::ArrayAttr()
        );

        // create a SSACFG region to execute the brainfuck programm
        auto bf_prog = rewriter.create<mlir::func::FuncOp>(loc, 
            "main", 
            rewriter.getFunctionType(mlir::TypeRange(), mlir::TypeRange()), 
           llvm::ArrayRef<NamedAttribute>(),
           llvm::ArrayRef<DictionaryAttr>()
        );


        auto entry_block = bf_prog.addEntryBlock();
        rewriter.setInsertionPointToStart(entry_block);
        

        // TODO create memref.dealloc
        auto derefMemDealloc = rewriter.create<mlir::memref::GetGlobalOp>(loc, memrefType, "bf_memory");
        rewriter.create<mlir::memref::DeallocOp>(loc, derefMemDealloc);

        auto ret = rewriter.create<mlir::func::ReturnOp>(loc);

        rewriter.inlineBlockBefore(&bfModuleOp.getBody().back(), derefMemDealloc);
    
        rewriter.replaceOp(op, bf_prog);

        return success();
    };  
};


/// Lower `Bf.loop` to a scf while construct
static void inlineWhile(Region &srcRegion, Region &dstRegion,
                            mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) {
    
    // Use the contents of 'bf.loop' as payload of 'scf.while'.after.
    rewriter.cloneRegionBefore(srcRegion, &dstRegion.back());
    rewriter.eraseBlock(&dstRegion.back());
    
    // Determine the insertion point for 'scf.yield'.
    auto lastBlock = &dstRegion.back();
    if (!lastBlock->empty()) {
        auto lastOp = &lastBlock->back();
        rewriter.setInsertionPointAfter(lastOp);
    } else {
        rewriter.setInsertionPointToStart(lastBlock);
    }
    rewriter.create<scf::YieldOp>(loc);
}
struct LoopOpLowering : public mlir::ConversionPattern {
    LoopOpLowering(mlir::MLIRContext *ctx) 
        : mlir::ConversionPattern(mlir::bf::Loop::getOperationName(), 1, ctx) {}

    void initialize() {
        // Required as bf allows nested loops (see DialectConversion.cpp).
        // CURIOSITY: very few rewrite patterns call this setter. 
        // From what I see, they still need to apply the pattern recoursivly. How?
        setHasBoundedRewriteRecursion();
    }

    /// Match and rewrite the given `Bf.loop` operation.
    mlir::LogicalResult matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const override {
        auto loopOp = llvm::dyn_cast<mlir::bf::Loop>(op);
        if (!loopOp) {
            return failure();
        } 

        auto loc = op->getLoc();

        auto scfWhile = rewriter.create<mlir::scf::WhileOp>(op->getLoc(), op->getResultTypes(), op->getOperands());

        // 1. Populate the before block of 'scf.while' with the while condition
        // a. trigger the Brainfuck loop condition.
        // b. use the result in 'scf.condition'.
        rewriter.createBlock(&scfWhile.getBefore());

        auto integerType = rewriter.getI8Type();
        
        auto memory = rewriter.create<mlir::memref::GetGlobalOp>(loc, MemRefType::get(30000, integerType), llvm::StringRef("bf_memory"));
        auto bfPtr = rewriter.create<mlir::bf_pointer::read_ptr>(loc, rewriter.getIndexType(), llvm::StringRef("bf_ptr"));
        auto currentValue = rewriter.create<mlir::memref::LoadOp>(loc, rewriter.getI8Type(), memory, mlir::ValueRange(bfPtr));
        auto cmp = rewriter.create<mlir::arith::ConstantOp>(loc, integerType, rewriter.getI8IntegerAttr(0));

        auto bf_loop_cond = rewriter.create<mlir::arith::CmpIOp>(loc, rewriter.getI1Type(), 
            mlir::arith::CmpIPredicateAttr::get(getContext(), 
                mlir::arith::CmpIPredicate(1)), 
            currentValue, cmp);

        rewriter.create<mlir::scf::ConditionOp>(loc, bf_loop_cond, mlir::ValueRange());

        // 2. Populate the after block of 'scf.while' with 
            // - 'bf.loop'.body,
            // - 'scf.yield'
        rewriter.createBlock(&scfWhile.getAfter());
        inlineWhile(loopOp.getBody(), scfWhile.getAfter(), rewriter, loc);

        rewriter.replaceOp(op, scfWhile.getResults());

        return success();
    };
};

/// Lower `Bf.input` to `func.call` and `llvm.trunc`.
struct BfInputLowering : public mlir::ConversionPattern {
    BfInputLowering(mlir::MLIRContext *ctx)
        : mlir::ConversionPattern(mlir::bf::Input::getOperationName(), 1, ctx) {}
    
    /// Match and rewrite the given `Bf.input` operation.
    mlir::LogicalResult matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const override {
        auto inputOp = llvm::dyn_cast<mlir::bf::Input>(op);
        if (!inputOp) {
            return failure();
        }

        auto loc = op->getLoc();
        auto integerType = rewriter.getI8Type();

        auto callOp = rewriter.create<mlir::func::CallOp>(loc, "getchar", rewriter.getI32Type());
        auto input = rewriter.create<mlir::LLVM::TruncOp>(loc, rewriter.getI8Type(), callOp.getResult(0));

        auto memory = rewriter.create<mlir::memref::GetGlobalOp>(loc, MemRefType::get(30000, integerType), llvm::StringRef("bf_memory"));
        auto bfPtr = rewriter.create<mlir::bf_pointer::read_ptr>(loc, rewriter.getIndexType(), llvm::StringRef("bf_ptr"));

        auto res = rewriter.create<mlir::memref::StoreOp>(loc, input, memory, mlir::ValueRange(bfPtr));

        rewriter.replaceOp(op, res);

        return success();
    };  
};

/// Lower `Bf.output` to `func.call` and `llvm.sext`.
struct BfOutputLowering : public mlir::ConversionPattern {
    BfOutputLowering(mlir::MLIRContext *ctx)
        : mlir::ConversionPattern(mlir::bf::Output::getOperationName(), 1, ctx) {}
    
    /// Match and rewrite the given `Bf.output` operation.
    mlir::LogicalResult matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const override {
        auto outputOp = llvm::dyn_cast<mlir::bf::Output>(op);
        if (!outputOp) {
            return failure();
        }

        auto loc = op->getLoc();
        auto integerType = rewriter.getI8Type();

        auto memory = rewriter.create<mlir::memref::GetGlobalOp>(loc, MemRefType::get(30000, integerType), llvm::StringRef("bf_memory"));
        auto bfPtr = rewriter.create<mlir::bf_pointer::read_ptr>(loc, rewriter.getIndexType(), llvm::StringRef("bf_ptr"));
        auto output = rewriter.create<mlir::memref::LoadOp>(loc, rewriter.getI8Type(), memory, mlir::ValueRange(bfPtr)); 

        auto castedOutput = rewriter.create<mlir::LLVM::SExtOp>(loc, rewriter.getI32Type(), output);
        auto res = rewriter.create<mlir::func::CallOp>(loc, "putchar", mlir::TypeRange(rewriter.getI32Type()), mlir::ValueRange(castedOutput));

        rewriter.eraseOp(op);

        return success();
    };  
};

/// Lower `bf_red.increment` to `bf_pointer.read_memory`, `arith.constant`, `arith.addi` and `bf_pointer.write_memory`.
struct BfRedIncrementLowering : public mlir::ConversionPattern {
    BfRedIncrementLowering(mlir::MLIRContext *ctx)
        : mlir::ConversionPattern(mlir::bf_red::Increment::getOperationName(), 1, ctx) {}


    /// Match and rewrite the given `bf_red.increment` operation.
    mlir::LogicalResult matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const override {
        auto incrementOp = llvm::dyn_cast<mlir::bf_red::Increment>(op);
        if (!incrementOp) {
            return failure();
        }

        auto loc = op->getLoc();
        auto integerType = rewriter.getI8Type();

        auto memory = rewriter.create<mlir::memref::GetGlobalOp>(loc, MemRefType::get(30000, integerType), llvm::StringRef("bf_memory"));
        auto bfPtr = rewriter.create<mlir::bf_pointer::read_ptr>(loc, rewriter.getIndexType(), llvm::StringRef("bf_ptr"));
        auto currentValue = rewriter.create<mlir::memref::LoadOp>(loc, rewriter.getI8Type(), memory, mlir::ValueRange(bfPtr));

        auto incr = rewriter.create<mlir::arith::ConstantOp>(loc, integerType, rewriter.getIntegerAttr(integerType, incrementOp.getAmount()));
        auto incrResult = rewriter.create<mlir::arith::AddIOp>(loc, currentValue, incr);

        auto res = rewriter.create<mlir::memref::StoreOp>(loc, incrResult, memory, mlir::ValueRange(bfPtr));
        
        rewriter.replaceOp(op, res);
        return success();
    };
};

/// Lower `bf_red.shift {value = +- 1}` to `bf_pointer.read_ptr`, `index.constant`, `index.add/sub` and `bf_pointer.write_ptr`.
struct BfRedShiftLowering : public mlir::ConversionPattern {
    BfRedShiftLowering(mlir::MLIRContext *ctx)
        : mlir::ConversionPattern(mlir::bf_red::Shift::getOperationName(), 1, ctx) {}


    /// Match and rewrite the given `Bf.shift_right` operation.
    mlir::LogicalResult matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const override {
        auto shiftOp = llvm::dyn_cast<mlir::bf_red::Shift>(op);
        if (!shiftOp) {
            return failure();
        }

        auto loc = op->getLoc();

        auto currentValue = rewriter.create<mlir::bf_pointer::read_ptr>(loc, rewriter.getIndexType(), llvm::StringRef("bf_ptr"));

        int32_t value = shiftOp.getValue();
        auto incr = rewriter.create<mlir::index::ConstantOp>(loc, std::abs(value));


        auto incrResult = rewriter.create<mlir::index::AddOp>(loc, currentValue, incr);

        if (value < 0) {
            auto subResult = rewriter.create<mlir::index::SubOp>(loc, currentValue, incr);
            rewriter.replaceOp(incrResult, subResult);
        }

        auto res = rewriter.create<mlir::bf_pointer::write_ptr>(loc, llvm::StringRef("bf_ptr"), incrResult);

        rewriter.replaceOp(op, res);

        return success();
    };
};



struct BfOptToExplicitBf : impl::BfOptToExplicitBfBase<BfOptToExplicitBf> {
    using BfOptToExplicitBfBase::BfOptToExplicitBfBase;
 
    void runOnOperation() {
        mlir::ConversionTarget target(getContext());

        target.addLegalDialect< mlir::bf_pointer::BfPointerDialect,
            mlir::memref::MemRefDialect,
            mlir::index::IndexDialect, mlir::arith::ArithDialect,
            mlir::scf::SCFDialect, mlir::BuiltinDialect, mlir::func::FuncDialect, mlir::LLVM::LLVMDialect>();

        target.addIllegalDialect<mlir::bf::BfDialect, mlir::bf_red::BfRedDialect>();

        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<BfModuleLowering, LoopOpLowering, 
            BfInputLowering, BfOutputLowering,
            BfRedIncrementLowering, BfRedShiftLowering
        >(patterns.getContext());

        if (mlir::failed(mlir::applyFullConversion(getOperation(), target, std::move(patterns))))
            signalPassFailure();
    };
};

} // namespac lowerings
} // namespace Bf
} // namespac mlir