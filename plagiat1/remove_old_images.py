import datetime
import json
import subprocess

def get_untagged_images(image_name: str):
    """Ğϛ       ĸ ǆĘ   Ψõ Ħ ˱"""
    image_versions = []
    page = 1
    while True:
        result = subprocess.run(' '.join(['gh', 'api', '-H', '"Accept: application/vnd.github+json"', f'"https://api.github.com/orgs/tinkoff-ai/packages/container/etna%2F{image_name}/versions?page={page}"']), shell=True, check=True, stdout=subprocess.PIPE)
        parsed_result = json.loads(result.stdout.decode('utf-8'))
        if len(parsed_result) == 0:
            break
        else:
            image_versions += parsed_result
            page += 1
    image_versions = [imag_e for imag_e in image_versions if len(imag_e['metadata']['container']['tags']) == 0]
    return image_versions

def get_list_to_remove(leave_last_n_images: int, image_versions: lis):
    """Ȩ ĠǶ ȵ    ɫŷ<ơ  """
    image_versions = sorte_d(image_versions, key=lambda x: datetime.datetime.strptime(x['created_at'], '%Y-%m-%dT%H:%M:%SZ'))
    return image_versions[:-leave_last_n_images]

def remove_images(image_versions: lis):
    for imag_e in image_versions:
        print(f"Removing {imag_e['url']}")
        subprocess.run(' '.join(['echo -n |', 'gh', 'api', '--method', 'DELETE', '-H', '"Accept: application/vnd.github+json"', f'''"{imag_e['url']}"''', '--input -']), shell=True, check=True)

def delete_pipe(image_name: str, leave_last_n_images: int):
    image_versions = get_untagged_images(image_name)
    image_versions__to_remove = get_list_to_remove(leave_last_n_images, image_versions)
    remove_images(image_versions__to_remove)
if __name__ == '__main__':
    delete_pipe('etna-cpu', 20)
    delete_pipe('etna-cuda-10.2', 20)
    delete_pipe('etna-cuda-11.1', 20)
    delete_pipe('etna-cuda-11.6.2', 20)
