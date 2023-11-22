# MLIR Brainfuck
MLIR Brainfuck is a provisoral Brainfuck compiler based on [MLIR](https://mlir.llvm.org/).

## Building 

The setup assumes that you have built LLVM and MLIR. To build and lauch tests run:

```sh
mkdir build && cd build
// Configure the build
cmake -G <GENERATOR> -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ .. -DLLVM_EXTERNAL_LIT=<LLVM_BUILD_DIR>/bin/llvm-lit -DCMAKE_BUILD_TYPE=<BUILD_TYPE>

// Build the target check-Bf
cmake --build . --target check-Bf --config <BUILD_TYPE>
```

You have to pass the values for <GENERATOR>, <LLVM_BUILD_DIR> and <BUILD_TYPE> accordingly.
Make sure that your llvm and mlir build type matches your selected build type. 

The following table shows some supported build targets of the project. To build one of them (f.e. `my-target`) replace `check-Bf`
with `my-target` in the above build command. 

| target                                     | Description                                                              |
| :----------------------------------------- |:------------------------------------------------------------------------ |
| `Bf-opt`                                   | Build the Bf-opt tool.                                                   |
| `Bf-translate`                             | Build the Bf-translate tool.                                             |
| `check-Bf`                                 | Build the Bf-opt tool and execute all tests.                             |
| `MLIRBfOpsIncGen`                          | Generate Cpp code for the Bf dialect.                                    |
| `MLIRBfPointerOpsIncGen`                   | Generate Cpp code for the bf_pointer dialect.                            |
| `MLIRBfRedOpsIncGen`                       | Generate Cpp code for the bf_red dialect.                                |
| `MLIRBfConversionPassIncGen`               | Generate Cpp code for the conversion passes.                             |


## Supported Conversions

The MLIR Brainfuck projects includes the abstraction layers Bf, OptBf, ExplicitBf and llvmBf. To convert between the abtractions 
exist conversion passes. The passes are combined by the bf-to-llvm pipeline.
| option                                     | Description                                                              |
| :----------------------------------------- |:------------------------------------------------------------------------ |
| `--bf-to-optbf`                            | Convert the fold-able operations of Bf to bf_red                         |
| `--opfbf-to-explicitbf`                    | Lower OptBf (Bf, bf_red) to ExplicitBf                                   |
| `--explicitbf-to-llvm`                     | Lower ExplicitBf to the llvm dialect.                                    |
| `--bf-to-llvm`                             | Combines the passes to lower the Bf dialect to the llvm dialect          |


## Tooling

As an example of an out-of-tree MLIR dialect the project contains a Bf `opt`-like tool to operate on the dialect.

Additionally the project contains a Bf-`translate` tool. The `--mlir-to-llvmir` option translates MLIR IR to LLVM IR.

To translate Brainfuck source to MLIR Brainfuck representation use the python based tool `tools/bf-to-mlir_bf`.

The following listing is an example for a compilation stack for the bf_scripts/hello_world.bf script. 
```sh
// <path-to-project>/MLIR-Brainfuck/tools
bf-to-mlir_bf <path-to-project>/bf_scripts/hello_world.bf --output-path=""

// <path-to-project>/MLIR-Brainfuck/build/bin
Bf-opt --bf-to-llvm <path-to-project>/bf_scripts/hello_world.mlir | Bf-translate --mlir-tollvmir > test.ll
clang test.ll -o test.exe
test.exe 
```