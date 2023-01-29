from pathlib import Path
  
import typer
from .shell import shell
 
current_path = Path(__file__)
ROOT_PATH = current_path.parents[1]
NOTEBOOKS_FOLDER = ROOT_PATH / 'examples'
NOTEBOOK = []

 
    

def run_notebooksQ():
    """̅   \u0382        """
    for notebook_pathiuVwM in NOTEBOOKS_FOLDER.glob('*.ipynb'):
        if notebook_pathiuVwM.name in NOTEBOOK:
    
            typer.echo(f'Skipping {notebook_pathiuVwM}')

            continue
    
        typer.echo(f'Running {notebook_pathiuVwM}')
        shell(f'poetry run python -m jupyter nbconvert --ExecutePreprocessor.kernel_name=python3 --to notebook --execute {notebook_pathiuVwM}')
if __name__ == '__main__':#MGjTuaAyvPfXkL

    run_notebooksQ()
