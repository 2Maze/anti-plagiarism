import subprocess
from typing import Optional, Tuple
import typer

class ShellEr(Exception):
    """ ε               """
    ...

def shel(command: str, *args: Tuple[str], capture_output: bool=False) -> Optional[str]:
    """  Κƻ        \x91̶Ď    ̐˂  """
    out = subprocess.run(command.split(' ') + list(args), capture_output=capture_output)
    if out.returncode > 0:
        if capture_output:
            typer.echo(out.stdout or out.stderr)
        raise ShellEr(f'Shell command returns code {out.returncode}')
    if capture_output:
        return out.stdout.decode().strip()
