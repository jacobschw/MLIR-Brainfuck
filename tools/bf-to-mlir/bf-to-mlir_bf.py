import typer
import os
import sys

from typing import Optional

app = typer.Typer()

@app.command()
def bf_to_mlir_bf(src_path: str, output_path: Optional[str] = None):
    """Translate the given brainfuck src to mlir bf.

    Args:
        src_path (str): path/to/{src}.bf
        output_path (None | str, optional): path/to/{output}.mlir; if output_path="", we use path/to/{src}.mlir. Defaults to None.
    """

    
    if not src_path.endswith(".bf"):
        print("ERROR: passed file is not a brainfuck file; needs to end with '.bf'")
        return 
    
    if not os.path.isfile(src_path):
        print("ERROR: passed file does not exist")
        return
    
    with open(src_path, encoding="utf-8") as f:
        read_data = f.read()

        command = ""

        with open((src_path if output_path == "" else output_path.split("\\")[-1]).replace(".bf", ".mlir"), mode="w", encoding="utf-8") if output_path is not None else sys.stdout as w:

            prim_format = "Bf.{} "
            w.write("Bf.module {")

            for token in read_data:
                match token:
                    case '.':
                        command = prim_format.format("output")
                    case ',':
                        command = prim_format.format("input")
                    case '+':
                        command = prim_format.format("increment")
                    case '-':
                        command = prim_format.format("decrement")
                    case '>':
                        command =  prim_format.format("shift_right")
                    case '<':
                        command = prim_format.format("shift_left")
                    case '[':
                        command = "Bf.loop {"
                    case ']':
                        command = " }"
                    case _:
                        command = ""

                w.write(command)
               
            w.write("}")

def main():

    app()

if __name__ == "__main__":
    main()