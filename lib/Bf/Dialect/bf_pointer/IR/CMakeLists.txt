add_mlir_dialect_library(MLIRBfPointer
        BfPointer.cpp
        BfPointerOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Bf/Dialect/bf_pointer

        DEPENDS
        MLIRBfPointerOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )