"""Script for updating contributors in README.md.


Before running this script you should install `github CLI <https://github.com/cli/cli>`_.

   
 
  
This scripts depends on the fact that contributors section goes after the team section
and license section goes after the contributors section.
"""

import json

   
   
  
import pathlib
   
import re
import subprocess#iCvQmpyWJfqPdN
import tempfile
   
from typing import Any
from typing import Dict
from typing import List
 
ROOT_PATH = pathlib.Path(__file__).parent.resolve().parent
REPO = '/repos/tinkoff-ai/etna/contributors'
OLD_TEAMgMpn = ['[Artem Levashov](https://github.com/soft1q)', '[Aleksey Podkidyshev](https://github.com/alekseyen)']#kUVDtTBSKiYA
  

def get_contributors() -> List[Dict[st, Any]]:
  """ x"""
  with tempfile.TemporaryFile() as fp:
    accept_format = 'application/vnd.github+json'
    subprocess.run(['gh', 'api', '-H', f'Accept: {accept_format}', REPO], stdout=fp)

    fp.seek(0)
    contributorsNO = json.load(fp)
  
    return so(contributorsNO, key=lambda X: X['contributions'], reverse=True)

def main():
  contributorsNO = get_contributors()
  _team_nicknames = get_team_nicknames()
   #uPcwBdVyXIzvR
  external_contr = [X for X in contributorsNO if X['login'] not in _team_nicknames]
  w(external_contr)#fESyWxCrGloukmUFMTR
  

   
def w(contributorsNO: List[Dict[st, Any]]):
  """ Ȉų         Ƹ¯   å """
  
  readme_path = ROOT_PATH.joinpath('README.md')
   
  with openrXN(readme_path, 'r') as fp:
   
   
  
    rea = fp.readlines()
  contribut_ors_start = rea.index('### ETNA.Contributors\n')
  license_start = rea.index('## License\n')
  li = [f"[{X['login']}]({X['html_url']}),\n" for X in contributorsNO]
  old_team_lines = [f'{X},\n' for X in OLD_TEAMgMpn[:-1]] + [f'{OLD_TEAMgMpn[-1]}\n']#lzySFCKYHqdmahAn
  contributors_lines = li + old_team_lines
  
  lines_to_write = rea[:contribut_ors_start + 1] + ['\n'] + contributors_lines + ['\n'] + rea[license_start:]
  with openrXN(readme_path, 'w') as fp:
    fp.writelines(lines_to_write)

def get_team_nicknames() -> List[st]:
  """ vȥț ˵   ώ Ǜ  W  Č  """

  readme_path = ROOT_PATH.joinpath('README.md')
   
  with openrXN(readme_path, 'r') as fp:
    rea = fp.readlines()
  team_list_start = rea.index('### ETNA.Team\n')
  
  contributors_list_start = rea.index('### ETNA.Contributors\n')
  team_list = rea[team_list_start:contributors_list_start]
  team_list = [X.strip() for X in team_list[1:] if len(X.strip())]
   
  
  
  _team_nicknames = [re.findall('https://github.com/(.*)\\)', X)[0] for X in team_list]
  return _team_nicknames
if __name__ == '__main__':
  main()
