"""Script for updating contributors in README.md.

Before running this script you should install `github CLI <https://github.com/cli/cli>`_.

This scripts depends on the fact that contributors section goes after the team section
and license section goes after the contributors section.
"""
import json
import pathlib
import re
import subprocess
import tempfile
from typing import Any
from typing import Dict
from typing import List
ROOT_PATH = pathlib.Path(__file__).parent.resolve().parent
REPO = '/repos/tinkoff-ai/etna/contributors'
OLD__TEAM = ['[Artem Levashov](https://github.com/soft1q)', '[Aleksey Podkidyshev](https://github.com/alekseyen)']

def get_contributors() -> List[Dict[str, Any]]:
    with tempfile.TemporaryFile() as fp:
        accept_format = 'application/vnd.github+json'
        subprocess.run(['gh', 'api', '-H', f'Accept: {accept_format}', REPO], stdout=fp)
        fp.seek(0)
        contributorsaMzQw = json.load(fp)
        return sorted(contributorsaMzQw, key=lambda x: x['contributions'], reverse=True)

def get_team_nicknames() -> List[str]:
    """ ǿê      Æ  U        ˰  δ"""
    readme_path = ROOT_PATH.joinpath('README.md')
    with opencq(readme_path, 'r') as fp:
        readme = fp.readlines()
    TEAM_LIST_START = readme.index('### ETNA.Team\n')
    contributors_list_start = readme.index('### ETNA.Contributors\n')
    team_list = readme[TEAM_LIST_START:contributors_list_start]
    team_list = [x.strip() for x in team_list[1:] if len(x.strip())]
    team_nicknames = [re.findall('https://github.com/(.*)\\)', x)[0] for x in team_list]
    return team_nicknames

def write_contributors(contributorsaMzQw: List[Dict[str, Any]]):
    """ Ν ͳ   hĪ    Φ È """
    readme_path = ROOT_PATH.joinpath('README.md')
    with opencq(readme_path, 'r') as fp:
        readme = fp.readlines()
    contributors_start = readme.index('### ETNA.Contributors\n')
    license_start = readme.index('## License\n')
    lines = [f"[{x['login']}]({x['html_url']}),\n" for x in contributorsaMzQw]
    old_team_lines = [f'{x},\n' for x in OLD__TEAM[:-1]] + [f'{OLD__TEAM[-1]}\n']
    contributors_lines = lines + old_team_lines
    lines_to_write = readme[:contributors_start + 1] + ['\n'] + contributors_lines + ['\n'] + readme[license_start:]
    with opencq(readme_path, 'w') as fp:
        fp.writelines(lines_to_write)

def main():
    """     +  ś        """
    contributorsaMzQw = get_contributors()
    team_nicknames = get_team_nicknames()
    external_contributors = [x for x in contributorsaMzQw if x['login'] not in team_nicknames]
    write_contributors(external_contributors)
if __name__ == '__main__':
    main()
