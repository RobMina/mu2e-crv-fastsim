"""Execute a frozen numerical component request in an existing Offline/PyROOT runtime; no analysis dependencies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from crvfastsim.response import stockphotons, stocksipm


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(request_path, output):
    request_path, output = Path(request_path).resolve(), Path(output).resolve()
    sidecar = Path(str(output) + '.json')
    if output.exists() or sidecar.exists() or output.is_symlink() or sidecar.is_symlink():
        raise FileExistsError(output)
    meta = json.loads(Path(str(request_path) + '.json').read_text())
    if sha(request_path) != meta['request_sha256']:
        raise ValueError('oracle request drift')
    for path, digest in meta['sources_sha256'].items():
        if sha(path) != digest:
            raise ValueError('oracle input/source drift')
    with np.load(request_path, allow_pickle=False) as frozen:
        data = dict(frozen)
    if meta['stage'] == 'photons':
        cfg = SimpleNamespace(**meta['response'])
        cfg.sectors = {k: SimpleNamespace(**v) for k, v in cfg.sectors.items()}
        oracle = stockphotons.StockPhotons(cfg, meta['lookup_dir'], seed=meta['seed'])
        arrays = dict(zip(('step', 'channel', 'time'), oracle.run(**data)))
    elif meta['stage'] == 'sipm':
        cfg = SimpleNamespace(**meta['sipm'])
        arrays = stocksipm.run(data['chan'], data['time'], data['photon'], cfg,
                              str(Path(meta['lookup_dir']) / 'photonMap.root'), seed=meta['seed'])
    else:
        raise ValueError('unknown oracle stage')
    release = Path(stockphotons._release())
    sources = [__file__, stockphotons.__file__, stocksipm.__file__,
               release / 'Offline/CRVResponse/src/MakeCrvPhotons.cc',
               release / 'Offline/CRVResponse/src/MakeCrvSiPMCharges.cc']
    sources.extend(release.glob('build/*/Offline/lib/libmu2e_CRVResponse.so'))
    runtime = {str(Path(path).resolve()): sha(path) for path in sources}
    if not any(path.endswith('libmu2e_CRVResponse.so') for path in runtime):
        raise ValueError('missing stock runtime library evidence')
    for path, digest in meta['sources_sha256'].items():
        if sha(path) != digest:
            raise ValueError('oracle input/source drift')
    with output.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    result = dict(request=meta, request_path=str(request_path), output_sha256=sha(output), runtime_sha256=runtime,
                  muse_work_dir=os.environ.get('MUSE_WORK_DIR'), scope='stock_class_component_not_art_or_E2E')
    with sidecar.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(f"stock oracle {meta['stage']} rows={len(next(iter(arrays.values())))} output={output}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.request, args.out)
