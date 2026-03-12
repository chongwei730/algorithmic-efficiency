"""Submission file for an AdamW optimizer with warmup+cosine LR in PyTorch."""

from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed.nn as dist_nn
from absl import logging
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from algoperf import spec
from algoperf.pytorch_utils import pytorch_setup
from .lr_sched_just_want_to_see_001lr import LineSearchScheduler
import time


import random
import torch.distributed as dist

if dist.is_available() and dist.is_initialized():
    print(f"DDP enabled, rank={dist.get_rank()}, world_size={dist.get_world_size()}")
else:
    print("Running in single-process (non-DDP) mode.")

USE_PYTORCH_DDP = pytorch_setup()[0]

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
  optimizer = torch.optim.AdamW(
      model_params.parameters(),
      lr=0.001,
      betas=(1.0 - hyperparameters.one_minus_beta1, hyperparameters.beta2),
      weight_decay=hyperparameters.weight_decay
    )

  optimizer_state = {
    'optimizer': optimizer
  }

  scheduler = LineSearchScheduler(optimizer=optimizer, num_search=16, start_lr=0, model_paras=list(model_params.parameters()), optimizer_type="Adam", injection=False, search_mode="bisection")


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
  log_dir: Optional[str] = None,
  train_state: Optional[Dict[str, Any]] = None,
) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  del train_state
  del eval_results
  if dist.is_initialized():
            world = dist.get_world_size()
            rank = dist.get_rank()
  else:
            world, rank = 1, 0
  # torch.cuda.reset_peak_memory_stats()
  current_model = current_param_container
  current_model.train()
  optimizer_state['optimizer'].zero_grad()
  accum_steps = hyperparameters.accum_steps
  device = next(current_model.parameters()).device

  # line_search_interval = int(round(hyperparameters.interval * workload.step_hint))
  line_search_interval = 200
  # # logging.warning(f"step_hint {workload.step_hint} rank={rank}")
  # # logging.warning(f"hyperparameters.interval {hyperparameters.interval} rank={rank}")
  # # logging.warning(f"interval {line_search_interval} rank={rank}")
  closure = None
  
  # warmup_length = int(0.01 * workload.step_hint)
  warmup_length = 100
  is_plateau = False
  if type(batch) == list:
    batch_ls = batch
    def make_closure():
      def closure(require_grad=False, batch=batch_ls):
        device = next(current_model.parameters()).device
        total_loss_t = torch.zeros((), device=device)
        count = 0 
                    
        for b in batch:
          count += 1
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
        logging.warning(f"count: {count}")
        assert count == hyperparameters.accum_steps


        if dist.is_initialized():
          # logging.warning(f"[rank {rank}] iter {global_step} Before closure_all_reduce")
          dist.all_reduce(avg_loss_t, op=dist.ReduceOp.SUM)
          # logging.warning(f"[rank {rank}] iter {global_step} After closure_all_reduce")
          avg_loss_t /= dist.get_world_size()
        #####


        print(f"[closure] rank={rank}/{world} is running forward+backward, loss={avg_loss_t}")
        #####

        return avg_loss_t.item()
      return closure
    closure = make_closure()

    # alpha = torch.tensor([scheduler.prev_alpha], device='cuda')

    # if dist.is_initialized():
    #         dist.broadcast(alpha, src=0)

    # for pg in optimizer_state['optimizer'].param_groups:
    #         pg['lr'] = alpha.item()

    batch = batch[0]
    if global_step % line_search_interval != 0 and global_step != warmup_length:
       is_plateau = True
    logging.warning(f"is_plateau {is_plateau}")

  # logging.warning(f"[rank {rank}] iter {global_step} before model_fn")

  scheduler = optimizer_state['scheduler']
  scheduler.step(
                closure,
                c1=hyperparameters.c1,
                step=global_step,
                interval=line_search_interval,
                condition="armijo",
                warmup_length=warmup_length,
                log_dir=log_dir,
                is_plateau=is_plateau
            )

  logits_batch, new_model_state = workload.model_fn(
    params=current_model,
    augmented_and_preprocessed_input_batch=batch,
    model_state=model_state,
    mode=spec.ForwardPassMode.TRAIN,
    rng=rng,
    update_batch_norm=True,
    dropout_rate=hyperparameters.dropout_rate,
  )
  # logging.warning(f"[rank {rank}] iter {global_step} after model_fn")

  label_smoothing = (
    hyperparameters.label_smoothing
    if hasattr(hyperparameters, 'label_smoothing')
    else 0.0
  )

  # logging.warning(f"[rank {rank}] iter {global_step} before loss_fn")
  loss_dict = workload.loss_fn(
    label_batch=batch['targets'],
    logits_batch=logits_batch,
    mask_batch=batch.get('weights'),
    label_smoothing=label_smoothing,
  )
  # logging.warning(f"[rank {rank}] iter {global_step} after loss_fn")
  summed_loss = loss_dict['summed']
  n_valid_examples = loss_dict['n_valid_examples']
  if USE_PYTORCH_DDP:
    # Use dist_nn.all_reduce to ensure correct loss and gradient scaling.
    # logging.warning(f"[rank {rank}] iter {global_step} Before normal_all_reduce")
    summed_loss = dist_nn.all_reduce(summed_loss)
    n_valid_examples = dist_nn.all_reduce(n_valid_examples)
    # logging.warning(f"[rank {rank}] iter {global_step} After normal_all_reduce")
  loss = summed_loss / n_valid_examples
  # logging.warning(f"[rank {rank}] iter {global_step} Before normal_backward")
  loss.backward()
  # logging.warning(f"[rank {rank}] iter {global_step} After normal_backward")

  if hasattr(hyperparameters, 'grad_clip'):
    grad_clip = hyperparameters.grad_clip
    torch.nn.utils.clip_grad_norm_(
      current_model.parameters(), max_norm=grad_clip
    )
  optimizer_state['optimizer'].step()
  # optimizer_state['scheduler'].step()
  curr = loss.item()

  if STATE.loss_list:
        prev = STATE.loss_list[-1]
        smooth = 0.1 * prev + 0.9 * curr
  else:
        smooth = curr

  STATE.loss_list.append(smooth)

  # Log training metrics - loss, grad_norm, batch_size.
  if global_step <= 10 or global_step % 500 == 0:
  # if True:
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
  # torch.cuda.synchronize()
  # peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
  # peak_reserved = torch.cuda.max_memory_reserved() / 1024**2

  # logging.warning(
  #     f"[PEAK] allocated={peak_alloc:.1f}MB | reserved={peak_reserved:.1f}MB"
  # )
  

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
    return 512
  else:
    raise ValueError(f'Unsupported workload name: {workload_name}.')
  
from dataclasses import dataclass, field
import os
import csv
import math

@dataclass
class PlateauState:
    best: float | None = None
    num_bad_steps: int = 0
    last_reduce_step: int = -10**18
    last_seen_len: int = 0
    last_seen_last_value: float | None = None
    last_checked_step: int = -1
    last_result: bool = False
    loss_list: List[float] = field(default_factory=list)

STATE = PlateauState()

def check_reduce_plateau(
    log_dir,
    patience,
    global_step,
    cooldown=0,
    mode="min",
):
    # ------------------------------------------------------------
    # 0. Idempotent
    # ------------------------------------------------------------
    if STATE.last_checked_step == global_step:
        return STATE.last_result

    values = STATE.loss_list
    result = False

    # ------------------------------------------------------------
    # 1. Only advance on new observation
    # ------------------------------------------------------------
    curr_len = len(values)
    curr_last = values[-1] if values else None

    if (
        curr_len == STATE.last_seen_len
        and curr_last == STATE.last_seen_last_value
    ):
        STATE.last_checked_step = global_step
        STATE.last_result = False
        return False

    STATE.last_seen_len = curr_len
    STATE.last_seen_last_value = curr_last

    # ------------------------------------------------------------
    # 2. First observation → initialize baseline
    # ------------------------------------------------------------
    if STATE.best is None:
        STATE.best = curr_last
        STATE.num_bad_steps = 0
        STATE.last_checked_step = global_step
        STATE.last_result = False
        return False

    # ------------------------------------------------------------
    # 3. Step-level improvement check
    # ------------------------------------------------------------
    improved = (
        curr_last < STATE.best
        if mode == "min"
        else curr_last > STATE.best
    )

    if improved:
        STATE.best = curr_last
        STATE.num_bad_steps = 0
        STATE.last_checked_step = global_step
        STATE.last_result = False
        return False

    # ------------------------------------------------------------
    # 4. Bad-step counting
    # ------------------------------------------------------------
    STATE.num_bad_steps += 1
    if STATE.num_bad_steps < patience:
        STATE.last_checked_step = global_step
        STATE.last_result = False
        return False

    # ------------------------------------------------------------
    # 5. Cooldown (only after first trigger)
    # ------------------------------------------------------------
    if (
        cooldown > 0
        and STATE.last_reduce_step >= 0
        and global_step - STATE.last_reduce_step <= cooldown
    ):
        STATE.last_checked_step = global_step
        STATE.last_result = False
        return False

    # ------------------------------------------------------------
    # 6. Trigger plateau (edge-trigger)
    # ------------------------------------------------------------
    STATE.last_reduce_step = global_step
    STATE.num_bad_steps = 0
    STATE.best = curr_last
    result = True

    STATE.last_checked_step = global_step
    STATE.last_result = result
    return result

def data_selection(
  workload: spec.Workload,
  input_queue: Iterator[Dict[str, spec.Tensor]],
  optimizer_state: spec.OptimizerState,
  current_param_container: spec.ParameterContainer,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  global_step: int,
  rng: spec.RandomState,
  log_dir: Optional[str] = None
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
  warmup_length = int(0.01 * workload.step_hint)
  is_plateau = check_reduce_plateau(log_dir=log_dir, global_step=global_step, patience=5)
  logging.warning(f"is_line_search = {is_plateau}, state={STATE}")
  


  if global_step % 200 != 0 and global_step % 100 != 0:
    batch = next(input_queue)
  else:
    n_search_batches = getattr(hyperparameters, "accum_steps", 4)
    batch = [next(input_queue) for _ in range(n_search_batches)]

  return batch