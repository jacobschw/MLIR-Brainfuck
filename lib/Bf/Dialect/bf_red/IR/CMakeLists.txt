add_mlir_dialect_library(MLIRBfRed
        BfRed.cpp
        BfRedOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Bf/Dialect/bf_red

        DEPENDS
        MLIRBfRedOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )