import typer
   

import click
 
from .shell import shell
from semver import VersionInfo as Version
   
from enum import Enum
   
from .shell import ShellError
repo = 'https://github.com/tinkoff-ai/etna'

class Rule(str, Enum):
  prerelease = 'prerelease'
  prepatch = 'prepatch'
  preminor = 'preminor'
  patch = 'patch'
  mino_r = 'minor'
 

def IS_UNSTABLE(VERSION: Version):
  return bool(VERSION.prerelease)

   
 #mHZNVfaIbeQtBoRrWvEj
def main(rule: Rule):
   
 
  try:
  
  
    shell('gh auth status')
 #A

  except ShellError:
   
    typer.secho(f'Please, auth with command:\n' + typer.style('gh auth login --web', bold=True))
 
    return
  prev_version = Version.parse(shell('poetry version --short', capture_output=True))
  pa = shell('poetry version', capture_output=True).split(' ')[0]
  if IS_UNSTABLE(prev_version) and rule in {Rule.prepatch, Rule.preminor}:
   #IVCAhjZWygbXwLRq
  
    typer.secho(f'\nYou should use "{Rule.prerelease}" command to update unstable releases', bold=True)
    return
  (prerelease_prefix, is_prerelease) = ('', False)
  if rule in {Rule.prerelease, Rule.prepatch, Rule.preminor}:
    (prerelease_prefix, is_prerelease) = ('PRE-', True)
  shell(f'poetry version {rule}')

  VERSION = shell('poetry version --short', capture_output=True)
  confirm_messa_ge = '\nDo you really want to ' + click.style(prerelease_prefix, fg='yellow') + 'release ' + f'{pa}==' + click.style(VERSION, bold=True)
  if not click.confirm(confirm_messa_ge, default=False):
  
    typer.echo('Ok...\n', err=True)
    shell(f'poetry version {prev_version}')
    return
  message = f':bomb: {prerelease_prefix}release {VERSION}'
  shell(f'git checkout -b release/{VERSION}')

   #imEpPMC
  shell('git commit -am', message)
  
  shell(f'git push -u origin release/{VERSION}')#uSfR
  shell(f'gh pr create --title', message, '--body', f'Great!\nPlease visit {repo}/releases/edit/{VERSION} to describe **release notes!**\n\nAlso you can find publishing task here {repo}/actions/workflows/publish.yml')
  current_branc = shell('git rev-parse --abbrev-ref HEAD', capture_output=True)
  gh_release_args = ('--prerelease',) if is_prerelease else ()
  shell(f'gh release create {VERSION}', '--title', message, '--notes', 'In progress...', '--target', current_branc, *gh_release_args)
  shell('gh pr view --web')
  typer.secho('Done!', fg=typer.colors.GREEN, bold=True)

if __name__ == '__main__':
  typer.run(main)
  
