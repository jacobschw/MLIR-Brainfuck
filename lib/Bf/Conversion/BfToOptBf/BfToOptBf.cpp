#include "Bf/Conversion/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Bf/Dialect/bf_red/IR/BfRed.h"
#include "Bf/Dialect/bf_red/IR/BfRedOps.h"

namespace mlir {
namespace bf {
namespace lowerings {
#define GEN_PASS_DEF_BFTOOPTBF
#include "Bf/Conversion/Passes.h.inc"

/// Lower `Bf.increment` to `bf_red.increment {amount = 1 : i8}`.
struct BfIncrementToBfRedLowering : public mlir::ConversionPattern {
    BfIncrementToBfRedLowering(mlir::MLIRContext *ctx)
        : mlir::ConversionPattern(mlir::bf::Increment::getOperationName(), 1, ctx) {}


    /// Match and rewrite the given `Bf.increment` operation.
    mlir::LogicalResult matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const override {
        auto incrementOp = llvm::dyn_cast<mlir::bf::Increment>(op);
        if (!incrementOp) {
            return failure();
        }

        auto loc = op->getLoc();

        auto res = rewriter.create<mlir::bf_red::Increment>(loc, 1);

        rewriter.replaceOp(op, res);

        return success();
    };
};

/// Lower `Bf.decrement` to `bf_red.increment {amount = -1 : i8}`.
struct BfDecrementToBfRedLowering : public mlir::ConversionPattern {
    BfDecrementToBfRedLowering(mlir::MLIRContext *ctx)
        : mlir::ConversionPattern(mlir::bf::Decrement::getOperationName(), 1, ctx) {}


    /// Match and rewrite the given `Bf.decrement` operation.
    mlir::LogicalResult matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const override {
        auto decrementOp = llvm::dyn_cast<mlir::bf::Decrement>(op);
        if (!decrementOp) {
            return failure();
        }

        auto loc = op->getLoc();

        auto res = rewriter.create<mlir::bf_red::Increment>(loc, -1);

        rewriter.replaceOp(op, res);

        return success();
    };
};


/// Lower `Bf.shift_right` to `bf_red.shift`.
struct BfShiftRightToShiftLowering : public mlir::ConversionPattern {
    BfShiftRightToShiftLowering(mlir::MLIRContext *ctx)
        : mlir::ConversionPattern(mlir::bf::ShiftRight::getOperationName(), 1, ctx) {}


    /// Match and rewrite the given `Bf.shift_right` operation.
    mlir::LogicalResult matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const override {
        auto shiftRighttOp = llvm::dyn_cast<mlir::bf::ShiftRight>(op);
        if (!shiftRighttOp) {
            return failure();
        }

        auto loc = op->getLoc();

        auto res = rewriter.create<mlir::bf_red::Shift>(loc, rewriter.getSI32IntegerAttr(1));

        rewriter.replaceOp(op, res);

        return success();
    };
};

/// Lower `Bf.shift_left` to `bf_red.shift {value = -1 : si32}`.
struct BfShiftLeftToShiftLowering : public mlir::ConversionPattern {
    BfShiftLeftToShiftLowering(mlir::MLIRContext *ctx)
        : mlir::ConversionPattern(mlir::bf::ShiftLeft::getOperationName(), 1, ctx) {}


    /// Match and rewrite the given `Bf.shift_left` operation.
    mlir::LogicalResult matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const override {
        auto shiftLefttOp = llvm::dyn_cast<mlir::bf::ShiftLeft>(op);
        if (!shiftLefttOp) {
            return failure();
        }

        auto loc = op->getLoc();

        auto res = rewriter.create<mlir::bf_red::Shift>(loc, rewriter.getSI32IntegerAttr(-1));

        rewriter.replaceOp(op, res);

        return success();
    };
};

struct BfToOptBf : impl::BfToOptBfBase<BfToOptBf> {
    using BfToOptBfBase::BfToOptBfBase;
 
    void runOnOperation() {
        mlir::ConversionTarget target(getContext());

        target.addLegalDialect<mlir::bf_red::BfRedDialect>();

        target.addIllegalDialect<mlir::bf::BfDialect>();
        target.addLegalOp<mlir::bf::Loop, mlir::bf::Module, mlir::bf::Input, mlir::bf::Output>();

        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<BfShiftRightToShiftLowering, BfShiftLeftToShiftLowering, BfIncrementToBfRedLowering, BfDecrementToBfRedLowering>(&getContext());

        if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
            signalPassFailure();
    };
};

} // namespac lowerings
} // namespace Bf
} // namespac mlir