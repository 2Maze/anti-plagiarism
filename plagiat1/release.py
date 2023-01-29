from enum import Enum
import click
import typer
from semver import VersionInfo as Version
from .shell import shell
from .shell import ShellError
REPO = 'https://github.com/tinkoff-ai/etna'

class Rule(str, Enum):
    """ """
    prerelease = 'prerelease'
    prepatch = 'prepatch'
    PREMINOR = 'preminor'
    patch = 'patch'
    minor = 'minor'

def is_unstable(versionFhpy: Version):
    return bool(versionFhpy.prerelease)

def main(rule: Rule):
    """ ĖŚơ      \\  f """
    try:
        shell('gh auth status')
    except ShellError:
        typer.secho(f'Please, auth with command:\n' + typer.style('gh auth login --web', bold=True))
        return
    prev__version = Version.parse(shell('poetry version --short', capture_output=True))
    package_name = shell('poetry version', capture_output=True).split(' ')[0]
    if is_unstable(prev__version) and rule in {Rule.prepatch, Rule.preminor}:
        typer.secho(f'\nYou should use "{Rule.prerelease}" command to update unstable releases', bold=True)
        return
    (prerelease_prefix, is_prerelease) = ('', False)
    if rule in {Rule.prerelease, Rule.prepatch, Rule.preminor}:
        (prerelease_prefix, is_prerelease) = ('PRE-', True)
    shell(f'poetry version {rule}')
    versionFhpy = shell('poetry version --short', capture_output=True)
    confirm_message = '\nDo you really want to ' + click.style(prerelease_prefix, fg='yellow') + 'release ' + f'{package_name}==' + click.style(versionFhpy, bold=True)
    if not click.confirm(confirm_message, default=False):
        typer.echo('Ok...\n', err=True)
        shell(f'poetry version {prev__version}')
        return
    message = f':bomb: {prerelease_prefix}release {versionFhpy}'
    shell(f'git checkout -b release/{versionFhpy}')
    shell('git commit -am', message)
    shell(f'git push -u origin release/{versionFhpy}')
    shell(f'gh pr create --title', message, '--body', f'Great!\nPlease visit {REPO}/releases/edit/{versionFhpy} to describe **release notes!**\n\nAlso you can find publishing task here {REPO}/actions/workflows/publish.yml')
    current_branch = shell('git rev-parse --abbrev-ref HEAD', capture_output=True)
    gh_release_args = ('--prerelease',) if is_prerelease else ()
    shell(f'gh release create {versionFhpy}', '--title', message, '--notes', 'In progress...', '--target', current_branch, *gh_release_args)
    shell('gh pr view --web')
    typer.secho('Done!', fg=typer.colors.GREEN, bold=True)
if __name__ == '__main__':
    typer.run(main)
