"""V2R1 crash-safe per-policy forensic journals. Resume is deliberately restart-from-start.

A checkpoint is evidence of completed closes, never a complete ledger until the worker result is serialized.
"""
from __future__ import annotations
import dataclasses
import hashlib
import json
import os
import pickle
import subprocess
from pathlib import Path
import numpy as np


def atomic(path: Path, body: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+'.part')
    with tmp.open('wb') as f:
        f.write(body); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
    return hashlib.sha256(body).hexdigest()


def encode(obj):
    if dataclasses.is_dataclass(obj):return dataclasses.asdict(obj)
    if isinstance(obj,np.ndarray):return obj.tolist()
    if isinstance(obj,np.generic):return obj.item()
    return str(obj)


class Journal:
    def __init__(self, directory: Path, policy: str, identity: dict):
        self.root=Path(directory)/('policy_'+hashlib.sha256(policy.encode()).hexdigest()[:16])
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"existing policy journal: use a fresh run directory: {self.root}")
        self.policy=policy; self.identity=dict(identity); self.previous=None
        self.last={}; self.complete=False

    def close(self, index: int, state: dict, context: dict | None = None):
        record={'policy':self.policy,**self.identity,'current_completed_index':index,'complete':False,
                'previous_checkpoint_sha256':self.previous,'state':state,'context':context or {}}
        body=json.dumps(record,sort_keys=True,default=encode).encode()
        # Immutable generation plus atomic latest pointer: neither prior proof nor the last close is overwritten.
        h=hashlib.sha256(body).hexdigest()
        atomic(self.root/f'close_{index:05d}_{h[:12]}.json',body)
        atomic(self.root/'latest.json',body)
        self.previous=h; self.last=record

    def fail(self, context: dict, exc: Exception):
        atomic(self.root/'failure.json',json.dumps({'policy':self.policy,**self.identity,'complete':False,
                'previous_checkpoint_sha256':self.previous,'context':context,'exception_type':type(exc).__name__,
                'exception':str(exc)},sort_keys=True,default=encode).encode())

    def finish(self, result: dict):
        h=atomic(self.root/'complete_result.pkl',pickle.dumps(result,protocol=5))
        atomic(self.root/'completion.json',json.dumps({'policy':self.policy,**self.identity,'complete':True,
              'result_sha256':h,'previous_checkpoint_sha256':self.previous,'resume_mode':'RESTART_FROM_START'},sort_keys=True).encode())
        self.complete=True


def identity(root: Path, cfg: dict) -> dict:
    root=Path(root)
    code={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for area in ['src','configs'] for p in sorted((root/area).rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    return {'code_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            'code_content_sha256':hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest(),
            'method_config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
            'input_manifest_sha256':hashlib.sha256((root/'data/regime_v2r1_input_manifest.csv').read_bytes()).hexdigest(),
            'seed':cfg['r3']['frontier']['seed_generation']}
