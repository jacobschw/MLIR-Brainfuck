add_mlir_public_c_api_library(BfCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  MLIRBf
)
