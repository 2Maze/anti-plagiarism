import subprocess
from typing import Optional, Tuple
import typer

class ShellError(Exception):
    """   ̥ŋ Ϭ ͑      """
    ...

def shell(command: str, *args: Tuple[str], cap: bool=False) -> Optional[str]:
    out = subprocess.run(command.split(' ') + list(args), capture_output=cap)
    if out.returncode > 0:
        if cap:
            typer.echo(out.stdout or out.stderr)
        raise ShellError(f'Shell command returns code {out.returncode}')
    if cap:
        return out.stdout.decode().strip()
