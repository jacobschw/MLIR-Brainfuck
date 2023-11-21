include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_Bf`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_Bf.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(BfPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT BfPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_Bf"
  TD_FILE dialects/BfOps.td
  SOURCES
    dialects/Bf.py
  DIALECT_NAME Bf)

declare_mlir_python_extension(BfPythonSources.Extension
  MODULE_NAME _BfDialects
  ADD_TO_PARENT BfPythonSources
  SOURCES
    BfExtension.cpp
  EMBED_CAPI_LINK_LIBS
    BfCAPI
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(BfPythonCAPI
  INSTALL_COMPONENT BfPythonModules
  INSTALL_DESTINATION python_packages/Bf/mlir_Bf/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/Bf/mlir_Bf/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    BfPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(BfPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/Bf/mlir_Bf"
  INSTALL_PREFIX "python_packages/Bf/mlir_Bf"
  DECLARED_SOURCES
    BfPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    BfPythonCAPI
  )
