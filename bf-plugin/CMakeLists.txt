add_llvm_library(BfPlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        Bf-plugin.cpp

        DEPENDS
        MLIRBf
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIRBf
        )
target_include_directories(
        BfPlugin
        PRIVATE
        "${Bf_BINARY_DIR}/include"
)
