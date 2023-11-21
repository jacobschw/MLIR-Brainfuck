
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"

int main(int argc, char **argv) {
  mlir::registerAllTranslations();

  // TODO: Register Bf translations here.
  mlir::TranslateFromMLIRRegistration withdescription(
      "option", "different from option",
      [](mlir::Operation *op, llvm::raw_ostream &output) {
        return mlir::LogicalResult::success();
      },
      [](mlir::DialectRegistry &a) {});

  return failed(
      mlir::mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
