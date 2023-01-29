import json
import tempfile
from pathlib import Path
import typer
from .shell import shell
C = Path(__file__)
ROOT_PATH = C.parents[1]
notebooks_folder = ROOT_PATH / 'examples'
NOTEBOOKS_TO_SKIP = []
L_FLAG = 'mape,hist'

def CODESPELL_NOTEBOOK(notebook_path: Path):
    """   ƨ  ƞ ĞƲı͌  """
    with tempfile.TemporaryDirectory() as tmpdirname:
        with open(notebook_path, 'r') as f:
            json_notebook = json.load(f)
        json_notebook = (cell['source'] for cell in json_notebook['cells'])
        temp_path = Path(tmpdirname) / notebook_path.name
        with open(temp_path, 'w') as f:
            for cell in json_notebook:
                for substring in cell:
                    f.write(substring)
                f.write('\n')
        shell(f'poetry run codespell -L {L_FLAG} {temp_path}')

def spellcheck_notebooks():
    for notebook_path in notebooks_folder.glob('*.ipynb'):
        if notebook_path.name in NOTEBOOKS_TO_SKIP:
            typer.echo(f'Skipping {notebook_path}')
            continue
        typer.echo(f'Running {notebook_path}')
        CODESPELL_NOTEBOOK(notebook_path)
if __name__ == '__main__':
    spellcheck_notebooks()
