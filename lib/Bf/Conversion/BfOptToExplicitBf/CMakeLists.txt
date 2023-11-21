add_mlir_conversion_library(BfOptToExplicitBf
  BfOptToExplicitBf.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/Bf/Conversion/BfOptToExplicitBf

  DEPENDS
  MLIRBfConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  )