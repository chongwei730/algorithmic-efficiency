import os
from collections import namedtuple
import logging
import json

def dict_to_namedtuple(d, name="Hyperparameters"):
          NT = namedtuple(name, d.keys())
          return NT(**d)
        
def write_hparams(hparams, tuning_dir_name):
          """Write hparams.json if absent; otherwise load it for consistency.

          Supports either:
            - dict
            - objects with _asdict() (e.g., namedtuple)
          Returns a dict.
          """
          path = os.path.join(tuning_dir_name, 'hparams.json')

          # Normalize to dict for writing/training.
          payload = hparams._asdict() if hasattr(hparams, "_asdict") else hparams
          if not isinstance(payload, dict):
              raise TypeError(f"hparams must be dict-like; got {type(payload)}")

          # If exists, try to load. If empty/corrupt, treat as missing.
          if os.path.exists(path):
              logging.info('Loading hparams from %s.', path)
              try:
                  with open(path, 'r', encoding='UTF-8') as f:
                      content = f.read().strip()
                  if not content:
                      raise ValueError("hparams.json is empty")
                  loaded = json.loads(content)
                  if not isinstance(loaded, dict):
                      raise ValueError("hparams.json is not a dict")
                  return loaded
              except Exception as e:
                  logging.warning(
                      'Failed to load %s (%s). Re-writing hparams.json.',
                      path, repr(e)
                  )
                  # fall through to rewrite

          # Atomic write: write temp then replace
          tmp_path = path + '.tmp'
          with open(tmp_path, 'w', encoding='UTF-8') as f:
              json.dump(payload, f, indent=2)
              f.flush()
              os.fsync(f.fileno())
          os.replace(tmp_path, path)

          logging.info('Saving hparams to %s.', path)
          return payload