
import subprocess
import os
import timeit
from datetime import timedelta

import typer

app = typer.Typer()

@app.command()
def test_speed(src_path: str):
    """_summary_

    Args:
        src_path (str): _description_
    """
    
    if not os.path.isfile(src_path):
        print("ERROR: passed file does not exist")
        return
    
    to_be_speed_tested = src_path

    program_name = to_be_speed_tested.rsplit('\\', maxsplit=1)[-1]

    start = timeit.default_timer()
    subprocess.call(to_be_speed_tested)
    end = timeit.default_timer()


    print(f"\nExecution of {program_name} took {end - start} seconds")

def main():

    app()

if __name__ == "__main__":
    main()