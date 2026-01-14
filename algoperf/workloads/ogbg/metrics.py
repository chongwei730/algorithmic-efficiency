# Forked from Flax example which can be found here:
# https://github.com/google/flax/blob/main/examples/ogbg_molpcba/train.py
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.distributed as dist
from clu import metrics
from sklearn.metrics import average_precision_score
from absl import logging
from algoperf.pytorch_utils import pytorch_setup

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()


def predictions_match_labels(
  *, logits: jnp.ndarray, labels: jnp.ndarray, **kwargs
) -> jnp.ndarray:
  """Returns a binary array indicating where predictions match the labels."""
  del kwargs  # Unused.
  preds = logits > 0
  return (preds == labels).astype(jnp.float32)


@flax.struct.dataclass
class MeanAveragePrecision(
  metrics.CollectingMetric.from_outputs(('logits', 'labels', 'mask'))
):
  """Computes the mean average precision (mAP) over different tasks."""

  def compute(self):
    # Matches the official OGB evaluation scheme for mean average precision.
    values = super().compute()
    labels = values['labels']
    logits = values['logits']
    mask = values['mask']
    sigmoid = jax.nn.sigmoid

    if USE_PYTORCH_DDP:
        import torch
        import torch.distributed as dist
        import numpy as np

        world = dist.get_world_size()


        labels_t = torch.from_numpy(np.asarray(labels, dtype=np.float32, order="C")).to(DEVICE).contiguous()
        logits_t = torch.from_numpy(np.asarray(logits, dtype=np.float32, order="C")).to(DEVICE).contiguous()

        # mask: bool -> uint8（0/1）
        mask_u8_t = torch.from_numpy(np.asarray(mask, dtype=np.uint8, order="C")).to(DEVICE).contiguous()

   
        def gather_cat(t: torch.Tensor) -> torch.Tensor:
            outs = [torch.empty_like(t) for _ in range(world)]
            dist.all_gather(outs, t)
            return torch.cat(outs, dim=0)

        labels = gather_cat(labels_t).cpu().numpy()
        logits = gather_cat(logits_t).cpu().numpy()
        mask_u8 = gather_cat(mask_u8_t).cpu().numpy()


        mask = (mask_u8 != 0)

        def sigmoid_np(x):
          return 1 / (1 + np.exp(-x))

        sigmoid = sigmoid_np

    mask = mask.astype(bool)

    probs = sigmoid(logits)
    num_tasks = labels.shape[1]
    average_precisions = np.full(num_tasks, np.nan)

    # Note that this code is slow (~1 minute).
    for task in range(num_tasks):
      # AP is only defined when there is at least one negative data
      # and at least one positive data.
      if np.sum(labels[:, task] == 0) > 0 and np.sum(labels[:, task] == 1) > 0:
        is_labeled = mask[:, task]
        average_precisions[task] = average_precision_score(
          labels[is_labeled, task], probs[is_labeled, task]
        )

    # When all APs are NaNs, return NaN. This avoids raising a RuntimeWarning.
    if np.isnan(average_precisions).all():
      return np.nan
    return np.nanmean(average_precisions)


class AverageDDP(metrics.Average):
  """Supports syncing metrics for PyTorch distributed data parallel (DDP)."""

  def compute(self) -> Any:
    if USE_PYTORCH_DDP:
      # Sync counts across devices.
      total_tensor = torch.tensor(np.asarray(self.total), device=DEVICE)
      count_tensor = torch.tensor(np.asarray(self.count), device=DEVICE)
      dist.all_reduce(total_tensor)
      dist.all_reduce(count_tensor)
      # Hacky way to avoid FrozenInstanceError
      # (https://docs.python.org/3/library/dataclasses.html#frozen-instances).
      object.__setattr__(self, 'total', total_tensor.cpu().numpy())
      object.__setattr__(self, 'count', count_tensor.cpu().numpy())
    return super().compute()


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
  accuracy: AverageDDP.from_fun(predictions_match_labels)
  loss: AverageDDP.from_output('loss')
  mean_average_precision: MeanAveragePrecision
