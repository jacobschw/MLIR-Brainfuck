# MLIR Brainfuck
MLIR Brainfuck is a provisoral Brainfuck compiler based on [MLIR](https://mlir.llvm.org/).

## Building 

The setup assumes that you have built LLVM and MLIR. To build and lauch tests run:

```sh
mkdir build && cd build
cmake -G <GENERATOR> -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ .. -DLLVM_EXTERNAL_LIT=<LLVM_BUILD_DIR>/bin/llvm-lit
cmake --build . --target check-Bf
```

You have to pass the values for <GENERATOR> and <LLVM_BUILD_DIR> accordingly.

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

As an example of an out-of-tree MLIR dialect(s) the project contains a Bf `opt`-like tool to operate on that dialect.

Additionally the project contains a Bf-`translate` tool. The --mlir-to-llvmir option translates MLIR IR to LLVM IR.

To translate Brainfuck source to MLIR Brainfuck representation use the python based tool bf-to-mlir_bf.

Example of a compilation stack for the MLIR-Brainfuck/bf_scripts/hello_world.bf script. 
```sh
// <path-to-project>/MLIR-Brainfuck/tools
bf-to-mlir_bf <path-to-project>/bf_scripts/hello_world.bf --output-path=""

// <path-to-project>/MLIR-Brainfuck/build/bin
Bf-opt --bf-to-llvm <path-to-project>/bf_scripts/hello_world.mlir | Bf-translate --mlir-tollvmir > test.ll
clang test.ll -o test.exe
test.exe 
```