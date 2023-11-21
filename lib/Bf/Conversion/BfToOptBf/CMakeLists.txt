add_mlir_conversion_library(BfToOptBf
  BfToOptBf.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/Bf/Conversion/BfToOptBf

  DEPENDS
  MLIRBfConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  )