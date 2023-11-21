#ifndef BF_RED_BASE
#define BF_RED_BASE

include "mlir/IR/OpBase.td"

def BfRed_Dialect : Dialect {
    let name = "bf_red";

    let summary = "An out-of-tree MLIR dialect contained in the MLIR Brainfuck project to enable folding of the Brainfuck commands '+', '-', '>' and '<'";

    let description = [{
        The bf_red dialect is intended to provide an intermediate representation that enables 
        folding of the Brainfuck commands '+', '-', '>' and '<'. The mixin of the Bf and bf_red dialect 
        is called OptBf. Users can either map the related Brainfuck commands directly or the 
        associated Bf operations to bf_red: 

            +  => Bf.increment   => bf_red.increment {amount = 1 : si8}
            -  => Bf.decrement   => bf_red.increment {amount = -1 : si8}
            >  => Bf.shift_right => bf_red.shift     {value = 1 : si32}
            <  => Bf.shift_left  => bf_red.shift     {value = -1 : si32}

    }];

    let cppNamespace = "::mlir::bf_red";
}


class BfRed_Op<string mnemonic, list<Trait> traits = []> :
        Op<BfRed_Dialect, mnemonic, traits>;

#endif // BF_RED_BASE