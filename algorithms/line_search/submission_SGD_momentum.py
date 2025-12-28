"""Submission file for an AdamW optimizer with warmup+cosine LR in PyTorch."""

from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed.nn as dist_nn
from absl import logging
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from algoperf import spec
from algoperf.pytorch_utils import pytorch_setup
from .lr_sched import LineSearchScheduler
import time

import random
import torch.distributed as dist

if dist.is_available() and dist.is_initialized():
    print(f"DDP enabled, rank={dist.get_rank()}, world_size={dist.get_world_size()}")
else:
    print("Running in single-process (non-DDP) mode.")

USE_PYTORCH_DDP = pytorch_setup()[0]



# def infer_device_from_model(model):
#     try:
#         return next(model.parameters()).device
#     except StopIteration:
#         return torch.device("cpu")

# def infer_device_from_batch(b):
#     # 返回第一个探测到的 tensor 的 device；找不到就返回 None
#     if torch.is_tensor(b):
#         return b.device
#     if isinstance(b, dict):
#         for v in b.values():
#             d = infer_device_from_batch(v)
#             if d is not None:
#                 return d
#         return None
#     # PyG 的 Data/Batch 通常有 .to()，但不保证有 .device
#     # 常见字段：x, edge_index, edge_attr, y, pos, ...
#     for attr in ("x", "edge_index", "edge_attr", "y", "pos"):
#         if hasattr(b, attr):
#             t = getattr(b, attr)
#             if torch.is_tensor(t):
#                 return t.device
#     # 列表/元组等容器
#     if isinstance(b, (list, tuple)):
#         for v in b:
#             d = infer_device_from_batch(v)
#             if d is not None:
#                 return d
#     return None

# def safe_batch_to_device(b, device):
#     if hasattr(b, "to"):
#         try:
#             return b.to(device)
#         except Exception:
#             pass
#     if torch.is_tensor(b):
#         return b.to(device)
#     if isinstance(b, dict):
#         return {k: safe_batch_to_device(v, device) for k, v in b.items()}
#     if isinstance(b, (list, tuple)):
#         conv = [safe_batch_to_device(v, device) for v in b]
#         return type(b)(conv) if isinstance(b, tuple) else conv
#     return b


def init_optimizer_state(
  workload: spec.Workload,
  model_params: spec.ParameterContainer,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  rng: spec.RandomState,
) -> spec.OptimizerState:
  """Creates an AdamW optimizer and a learning rate schedule."""
  del model_state
  del rng


  # optimizer = torch.optim.Adam(
  #     model_params.parameters(),
  #     lr=hyperparameters.learning_rate,
  #     betas=(1.0 - hyperparameters.one_minus_beta1, hyperparameters.beta2),
  #     eps=1e-8,
  #     fused=False,
  #   )
  print(hyperparameters)
  optimizer = torch.optim.SGD(
      model_params.parameters(),
      momentum=hyperparameters.momentum,
      lr=1,
    )

  optimizer_state = {
    'optimizer': optimizer
  }

  scheduler = LineSearchScheduler(optimizer=optimizer, num_search=16, start_lr=1, model_paras=list(model_params.parameters()), optimizer_type="SGD_momentum", injection=False, search_mode="bisection")


  optimizer_state['scheduler'] = scheduler

  return optimizer_state



def update_params(
  workload: spec.Workload,
  current_param_container: spec.ParameterContainer,
  current_params_types: spec.ParameterTypeTree,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  batch: Dict[str, spec.Tensor],
  loss_type: spec.LossType,
  optimizer_state: spec.OptimizerState,
  eval_results: List[Tuple[int, float]],
  global_step: int,
  rng: spec.RandomState,
  train_state: Optional[Dict[str, Any]] = None,
) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  del train_state
  del eval_results

  current_model = current_param_container
  current_model.train()
  optimizer_state['optimizer'].zero_grad()
  accum_steps = hyperparameters.accum_steps
  device = next(current_model.parameters()).device

  line_search_interval = int(round(hyperparameters.interval * workload.step_hint))
  

  if global_step % line_search_interval == 0:
    def closure(require_grad=False):
      optimizer_state['optimizer'].zero_grad()
      device = next(current_model.parameters()).device
      total_loss_t = torch.zeros((), device=device)

      for b in batch:
        logits_batch, new_model_state = workload.model_fn(
            params=current_model,
            augmented_and_preprocessed_input_batch=b,
            model_state=model_state,
            mode=spec.ForwardPassMode.TRAIN,
            rng=rng,
            update_batch_norm=True,
            dropout_rate=hyperparameters.dropout_rate,
          )
        label_smoothing = (
          hyperparameters.label_smoothing
          if hasattr(hyperparameters, 'label_smoothing')
          else 0.0
        )

        loss_dict = workload.loss_fn(
          label_batch=b['targets'],
          logits_batch=logits_batch,
          mask_batch=b.get('weights'),
          label_smoothing=label_smoothing,
        )




        loss = loss_dict["summed"] / loss_dict["n_valid_examples"]
      

        # total_loss += loss.item()
        if require_grad:
          (loss / accum_steps).backward() 

        total_loss_t = total_loss_t + loss.detach()
      
      avg_loss_t = total_loss_t / accum_steps


      if dist.is_initialized():
        dist.all_reduce(avg_loss_t, op=dist.ReduceOp.SUM)
        avg_loss_t /= dist.get_world_size()

      #####
      if dist.is_initialized():
            print("USING DDP")
            world = dist.get_world_size()
            rank = dist.get_rank()
      else:
            print("NO DDP")
            world, rank = 1, 0

      print(f"[closure] rank={rank}/{world} is running forward+backward, loss={avg_loss_t}")
      #####

      return avg_loss_t.item()
    

    scheduler = optimizer_state['scheduler']
    start_time = time.time()
    scheduler.step(
                closure,
                c1=hyperparameters.c1,
                step=global_step,
                interval=line_search_interval,
                condition="armijo",
            )
    elapsed = time.time() - start_time
    print(f"[LineSearch] {accum_steps} step took {elapsed:.4f} seconds")
    alpha = torch.tensor([scheduler.prev_alpha], device='cuda')

    if dist.is_initialized():
            dist.broadcast(alpha, src=0)

    for pg in optimizer_state['optimizer'].param_groups:
            pg['lr'] = alpha.item()

    


    batch = batch[0]

  logits_batch, new_model_state = workload.model_fn(
    params=current_model,
    augmented_and_preprocessed_input_batch=batch,
    model_state=model_state,
    mode=spec.ForwardPassMode.TRAIN,
    rng=rng,
    update_batch_norm=True,
    dropout_rate=hyperparameters.dropout_rate,
  )

  label_smoothing = (
    hyperparameters.label_smoothing
    if hasattr(hyperparameters, 'label_smoothing')
    else 0.0
  )


  loss_dict = workload.loss_fn(
    label_batch=batch['targets'],
    logits_batch=logits_batch,
    mask_batch=batch.get('weights'),
    label_smoothing=label_smoothing,
  )
  summed_loss = loss_dict['summed']
  n_valid_examples = loss_dict['n_valid_examples']
  if USE_PYTORCH_DDP:
    # Use dist_nn.all_reduce to ensure correct loss and gradient scaling.
    summed_loss = dist_nn.all_reduce(summed_loss)
    n_valid_examples = dist_nn.all_reduce(n_valid_examples)
  loss = summed_loss / n_valid_examples

  loss.backward()

  if hasattr(hyperparameters, 'grad_clip'):
    grad_clip = hyperparameters.grad_clip
    torch.nn.utils.clip_grad_norm_(
      current_model.parameters(), max_norm=grad_clip
    )
  optimizer_state['optimizer'].step()
  # optimizer_state['scheduler'].step()

  # Log training metrics - loss, grad_norm, batch_size.
  if global_step <= 10 or global_step % 500 == 0:
    with torch.no_grad():
      parameters = [p for p in current_model.parameters() if p.grad is not None]
      grad_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2
      )
    if workload.metrics_logger is not None:
      lr = optimizer_state['optimizer'].param_groups[0]['lr']
      workload.metrics_logger.append_scalar_metrics(
        {
          'loss': loss.item(),
          'grad_norm': grad_norm.item(),
          'lr': lr
        },
        global_step,
      )
    logging.info(
      '%d) loss = %0.3f, grad_norm = %0.3f',
      global_step,
      loss.item(),
      grad_norm.item(),
    )

  return (optimizer_state, current_param_container, new_model_state)




def prepare_for_eval(
  workload: spec.Workload,
  current_param_container: spec.ParameterContainer,
  current_params_types: spec.ParameterTypeTree,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  loss_type: spec.LossType,
  optimizer_state: spec.OptimizerState,
  eval_results: List[Tuple[int, float]],
  global_step: int,
  rng: spec.RandomState,
) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params)."""
  del workload
  del hyperparameters
  del current_params_types
  del loss_type
  del eval_results
  del global_step
  del rng
  return (optimizer_state, current_param_container, model_state)


def get_batch_size(workload_name):
  # Return the global batch size.
  if workload_name == 'criteo1tb':
    return 262_144
  elif workload_name == 'fastmri':
    return 32
  elif workload_name == 'imagenet_resnet':
    return 1024
  elif workload_name == 'imagenet_resnet_silu':
    return 512
  elif workload_name == 'imagenet_resnet_gelu':
    return 512
  elif workload_name == 'imagenet_vit':
    return 1024
  elif workload_name == 'librispeech_conformer':
    return 256
  elif workload_name == 'librispeech_deepspeech':
    return 256
  elif workload_name == 'ogbg':
    return 512
  elif workload_name == 'wmt':
    return 128
  elif workload_name == 'mnist':
    return 16
  elif workload_name == 'cifar':
    return 64
  else:
    raise ValueError(f'Unsupported workload name: {workload_name}.')

def data_selection(
  workload: spec.Workload,
  input_queue: Iterator[Dict[str, spec.Tensor]],
  optimizer_state: spec.OptimizerState,
  current_param_container: spec.ParameterContainer,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  global_step: int,
  rng: spec.RandomState,
) -> Dict[str, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.
  Each element of the queue is a batch of training examples and labels.
  """
  # del workload
  del optimizer_state
  del current_param_container
  del model_state
  # del hyperparameters
  # del global_step
  del rng

  line_search_interval = int(round(hyperparameters.interval * workload.step_hint))

  if global_step % line_search_interval != 0:
    batch = next(input_queue)
  else:
    n_search_batches = getattr(hyperparameters, "line_search_batches", 4)
    batch = [next(input_queue) for _ in range(n_search_batches)]

  return batch
