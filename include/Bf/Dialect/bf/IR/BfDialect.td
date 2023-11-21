#ifndef Bf_DIALECT
#define Bf_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Bf dialect definition.
//===----------------------------------------------------------------------===//

def Bf_Dialect : Dialect {
    let name = "Bf";
    let summary = "An out-of-tree MLIR dialect for the Brainfuck language contained in the MLIR Brainfuck project .";
    let description = [{
        This dialect is an implementation of the Brainfuck Language in MLIR.
        See the following mapping of Brainfuck commands to Bf operations:

            Primitive commands map to operations without a region: 
                +           => Bf.increment
                -           => Bf.decrement
                .           => Bf.input
                ,           => Bf.output
                >           => Bf.shift_right
                <           => Bf.shift_left

            Composed commands map to operations with a region:
                [ .. ]      => Bf.loop
                ( .. , bf)  => Bf.module

        Implicitly, the operations are performed on a linear arrangement of 30000 cells; the cells are initialized to 0.
        To access a cell a pointer is used; the pointer is initialized to 0. The cell accessible by the pointer at a given moment 
        in exection time is called the current cell. 

        The Bf.loop is defined with respect to the Brainfuck loop condition
            current cell != 0 
    }];
    let cppNamespace = "::mlir::bf";

    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base Bf operation definition.
//===----------------------------------------------------------------------===//

class Bf_Op<string mnemonic, list<Trait> traits = []> :
        Op<Bf_Dialect, mnemonic, traits>;

class Bf_PrimitiveOp<string mnemonic, list<Trait> traits = []>:
        Bf_Op<mnemonic, traits>;

class Bf_ComposedOp<string mnemonic, list<Trait> traits = []> :
        Bf_Op<mnemonic, traits>;

#endif // Bf_DIALECT
