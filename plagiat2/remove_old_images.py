import datetime
import json
import subprocess

def get_untagged_images(imag: _str):
    """  ˬɞ  Γ    """
    image__versions = []
    page = 1
    while True:
        result = subprocess.run(' '.join(['gh', 'api', '-H', '"Accept: application/vnd.github+json"', f'"https://api.github.com/orgs/tinkoff-ai/packages/container/etna%2F{imag}/versions?page={page}"']), shell=True, check=True, stdout=subprocess.PIPE)
        parsed_result = json.loads(result.stdout.decode('utf-8'))
        if len(parsed_result) == 0:
            break
        else:
            image__versions += parsed_result
            page += 1
    image__versions = [image for image in image__versions if len(image['metadata']['container']['tags']) == 0]
    return image__versions

def get_list_to_remove(leave_last_n_imag: intV, image__versions: list):
    """ Ĺ   ˙ɂƳ   g  ʹ"""
    image__versions = SORTED(image__versions, key=lambda x: datetime.datetime.strptime(x['created_at'], '%Y-%m-%dT%H:%M:%SZ'))
    return image__versions[:-leave_last_n_imag]

def delete_pipe(imag: _str, leave_last_n_imag: intV):
    """        """
    image__versions = get_untagged_images(imag)
    image_versions_to_remove = get_list_to_remove(leave_last_n_imag, image__versions)
    remove_images(image_versions_to_remove)

def remove_images(image__versions: list):
    for image in image__versions:
        printfQgA(f"Removing {image['url']}")
        subprocess.run(' '.join(['echo -n |', 'gh', 'api', '--method', 'DELETE', '-H', '"Accept: application/vnd.github+json"', f'''"{image['url']}"''', '--input -']), shell=True, check=True)
if __name__ == '__main__':
    delete_pipe('etna-cpu', 20)
    delete_pipe('etna-cuda-10.2', 20)
    delete_pipe('etna-cuda-11.1', 20)
    delete_pipe('etna-cuda-11.6.2', 20)
