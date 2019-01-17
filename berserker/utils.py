import os
import ntpath
import requests
from io import BytesIO
from zipfile import ZipFile
from pathlib import Path
import requests
from tqdm import tqdm
import math


def maybe_download_unzip(url, dst_path, verbose=False, force=False):
    dir = os.path.join(dst_path, os.path.splitext(ntpath.basename(url))[0])
    if os.path.exists(dir) and not force:
        if verbose:
            print('Directory %s already exists, skipping download...'%dir)
        return
    if verbose:
        print('Downloading %s...'%url)

    r = requests.get(url, stream=True)
    total_size_mb = math.ceil(int(r.headers.get('content-length', 0)) / 1024 / 1024);

    f = BytesIO()
    func = lambda x: tqdm(x, total=total_size_mb, unit='MB', unit_scale=True) if verbose else lambda x:x
    for chunk in func(r.iter_content(1024 * 1024)):
        f.write(chunk)
    ZipFile(f).extractall(dst_path)
    if verbose:
        print('Done downloading %s...'%url)
    return