get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIROptLib
        MLIRBf
        MLIRBfPointer
        MLIRBfRed
        )
add_llvm_executable(Bf-opt Bf-opt.cpp)

llvm_update_compile_flags(Bf-opt)
target_link_libraries(Bf-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(Bf-opt)

