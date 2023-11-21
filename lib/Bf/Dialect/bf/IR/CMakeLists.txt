add_mlir_dialect_library(MLIRBf
        BfDialect.cpp
        BfOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Bf/Dialect/bf

        DEPENDS
        MLIRBfOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )