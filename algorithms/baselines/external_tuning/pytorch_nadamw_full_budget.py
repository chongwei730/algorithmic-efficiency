"""Submission file for an NAdamW optimizer with warmup+cosine LR in PyTorch."""

import math
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed.nn as dist_nn
from absl import logging
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch.distributed as dist
import os
from algoperf import spec
import numpy as np
from algoperf.pytorch_utils import pytorch_setup
import matplotlib.pyplot as plt
import os
USE_PYTORCH_DDP = pytorch_setup()[0]


# Modified from github.com/pytorch/pytorch/blob/v1.12.1/torch/optim/adamw.py.
class NAdamW(torch.optim.Optimizer):
  r"""Implements NAdamW algorithm.

  See Table 1 in https://arxiv.org/abs/1910.05446 for the implementation of
  the NAdam algorithm (there is also a comment in the code which highlights
  the only difference of NAdamW and AdamW).
  For further details regarding the algorithm we refer to
  `Decoupled Weight Decay Regularization`_.

  Args:
    params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
    lr (float, optional): learning rate (default: 1e-3)
    betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.999))
    eps (float, optional): term added to the denominator to improve
        numerical stability (default: 1e-8)
    weight_decay (float, optional): weight decay coefficient (default: 1e-2)
  .. _Decoupled Weight Decay Regularization:
      https://arxiv.org/abs/1711.05101
  .. _On the Convergence of Adam and Beyond:
      https://openreview.net/forum?id=ryQu7f-RZ
  """

  def __init__(
    self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
  ):
    if not 0.0 <= lr:
      raise ValueError(f'Invalid learning rate: {lr}')
    if not 0.0 <= eps:
      raise ValueError(f'Invalid epsilon value: {eps}')
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
    if not 0.0 <= weight_decay:
      raise ValueError(f'Invalid weight_decay value: {weight_decay}')
    defaults = {
      'lr': lr,
      'betas': betas,
      'eps': eps,
      'weight_decay': weight_decay,
    }
    super().__init__(params, defaults)

  def __setstate__(self, state):
    super().__setstate__(state)
    state_values = list(self.state.values())
    step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
      state_values[0]['step']
    )
    if not step_is_tensor:
      for s in state_values:
        s['step'] = torch.tensor(float(s['step']))

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.

    Args:
      closure (callable, optional): A closure that reevaluates the model
          and returns the loss.
    """
    self._cuda_graph_capture_health_check()

    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      params_with_grad = []
      grads = []
      exp_avgs = []
      exp_avg_sqs = []
      state_steps = []
      beta1, beta2 = group['betas']

      for p in group['params']:
        if p.grad is None:
          continue
        params_with_grad.append(p)
        if p.grad.is_sparse:
          raise RuntimeError('NAdamW does not support sparse gradients')
        grads.append(p.grad)

        state = self.state[p]

        # State initialization
        if len(state) == 0:
          state['step'] = torch.tensor(0.0)
          # Exponential moving average of gradient values
          state['exp_avg'] = torch.zeros_like(
            p, memory_format=torch.preserve_format
          )
          # Exponential moving average of squared gradient values
          state['exp_avg_sq'] = torch.zeros_like(
            p, memory_format=torch.preserve_format
          )

        exp_avgs.append(state['exp_avg'])
        exp_avg_sqs.append(state['exp_avg_sq'])
        state_steps.append(state['step'])

      nadamw(
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        beta1=beta1,
        beta2=beta2,
        lr=group['lr'],
        weight_decay=group['weight_decay'],
        eps=group['eps'],
      )
      

    return loss


def nadamw(
  params: List[Tensor],
  grads: List[Tensor],
  exp_avgs: List[Tensor],
  exp_avg_sqs: List[Tensor],
  state_steps: List[Tensor],
  beta1: float,
  beta2: float,
  lr: float,
  weight_decay: float,
  eps: float,
) -> None:
  r"""Functional API that performs NAdamW algorithm computation.
  See NAdamW class for details.
  """

  if not all(isinstance(t, torch.Tensor) for t in state_steps):
    raise RuntimeError(
      'API has changed, `state_steps` argument must contain a list of'
      + ' singleton tensors'
    )

  for i, param in enumerate(params):
    grad = grads[i]
    exp_avg = exp_avgs[i]
    exp_avg_sq = exp_avg_sqs[i]
    step_t = state_steps[i]

    # Update step.
    step_t += 1

    # Perform stepweight decay.
    param.mul_(1 - lr * weight_decay)

    # Decay the first and second moment running average coefficient.
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    # Only difference between NAdamW and AdamW in this implementation.
    # The official PyTorch implementation of NAdam uses a different algorithm.
    # We undo these ops later on, which could cause numerical issues but saves
    # us from having to make an extra copy of the gradients.
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

    step = step_t.item()

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    step_size = lr / bias_correction1

    bias_correction2_sqrt = math.sqrt(bias_correction2)
    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

    param.addcdiv_(exp_avg, denom, value=-step_size)
    exp_avg.sub_(grad, alpha=1 - beta1).div_(beta1)


def init_optimizer_state(
  workload: spec.Workload,
  model_params: spec.ParameterContainer,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  rng: spec.RandomState,
) -> spec.OptimizerState:
  """Creates a NAdamW optimizer and a learning rate schedule."""
  del model_state
  del rng

  optimizer_state = {
    'optimizer': NAdamW(
      model_params.parameters(),
      lr=hyperparameters.learning_rate,
      betas=(1.0 - hyperparameters.one_minus_beta1, hyperparameters.beta2),
      eps=1e-8,
      weight_decay=hyperparameters.weight_decay,
    ),
  }

  def pytorch_cosine_warmup(step_hint: int, hyperparameters, optimizer):
    warmup_steps = int(hyperparameters.warmup_factor * step_hint)
    warmup = LinearLR(
      optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_steps = max(step_hint - warmup_steps, 1)
    cosine_decay = CosineAnnealingLR(optimizer, T_max=cosine_steps)
    return SequentialLR(
      optimizer, schedulers=[warmup, cosine_decay], milestones=[warmup_steps]
    )

  optimizer_state['scheduler'] = pytorch_cosine_warmup(
    workload.step_hint, hyperparameters, optimizer_state['optimizer']
  )

  return optimizer_state


@torch.no_grad()
def get_real_update_vector(model, optimizer):

    theta_old = [p.detach().clone() for p in model.parameters()]


    optimizer.step()


    delta = []
    for p_old, p in zip(theta_old, model.parameters()):
        delta.append((p.detach() - p_old).flatten())

    delta = torch.cat(delta)


    for p_old, p in zip(theta_old, model.parameters()):
        p.copy_(p_old)

    return delta


@torch.no_grad()
def visualize_update_direction_slice(
    current_model,
    optimizer_state,
    workload,
    batch,
    model_state,
    rng,
    global_step,
    num_points: int = 80,
    t_min: float = 1e-6,
    t_max: float = 2e-3,
    c1: float = 1e-4,
    out_dir: str = "./nadam_img",
):
    """
    Plot 1D loss slice phi(t) = L(theta + t * d) along the *actual NAdamW update direction*,
    and compare that constructed direction against the true optimizer step delta.

    Notes:
      - We build d to match YOUR nadamw() implementation:
          param.mul_(1 - lr * weight_decay)
          param.addcdiv_(m2, denom, value = - lr / bc1)
        where m2 = beta1*m + (1-beta1)*grad, denom = sqrt(v/bc2) + eps.
      - We include weight decay contribution in d so the comparison is meaningful.
      - We DO NOT call scheduler here.
      - Safe for DDP: comparison metrics are reduced; plots only rank0.
    """

    # -----------------------------
    # DDP helpers
    # -----------------------------
    ddp_on = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if ddp_on else 0
    world_size = dist.get_world_size() if ddp_on else 1
    is_rank0 = (rank == 0)

    def reduce_mean_scalar(x):
        """All-reduce mean for a scalar tensor or python number."""
        if not ddp_on:
            return float(x.detach().item()) if torch.is_tensor(x) else float(x)

        device = next(current_model.parameters()).device
        if not torch.is_tensor(x):
            x = torch.tensor(float(x), device=device)
        else:
            x = x.detach()
            if x.dim() != 0:
                x = x.reshape(())
            x = x.to(device)

        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= world_size
        return float(x.item())

    def reduce_mean_vector(v: torch.Tensor) -> torch.Tensor:
        """All-reduce mean for a 1D vector tensor."""
        if not ddp_on:
            return v
        v = v.detach()
        dist.all_reduce(v, op=dist.ReduceOp.SUM)
        v /= world_size
        return v

    # -----------------------------
    # Utilities: flatten params, restore, etc.
    # -----------------------------
    def flatten_params(params_list):
        return torch.cat([p.detach().flatten() for p in params_list])

    # -----------------------------
    # Grab optimizer hyperparams
    # -----------------------------
    optimizer = optimizer_state["optimizer"]
    group = optimizer.param_groups[0]
    lr = float(group["lr"])
    beta1, beta2 = group["betas"]
    eps = float(group["eps"])
    weight_decay = float(group.get("weight_decay", 0.0))

    # -----------------------------
    # Determine step t from optimizer state
    # (state['step'] is incremented inside optimizer.step(); here we want CURRENT step index)
    # -----------------------------
    step_t = None
    for p in current_model.parameters():
        st = optimizer.state.get(p, None)
        if st is not None and "step" in st:
            step_t = int(st["step"].item())
            break

    if step_t is None:
        if is_rank0:
            logging.warning("[slice] optimizer state has no 'step' yet; skipping slice.")
        return

    # Avoid step=0 causing bc1/bc2=0
    step_eff = max(step_t, 1)
    bc1 = 1.0 - (beta1 ** step_eff)
    bc2 = 1.0 - (beta2 ** step_eff)
    bc2_sqrt = math.sqrt(bc2)

    # -----------------------------
    # Build parameter list that matches optimizer state / grads
    # -----------------------------
    params = []
    m_list = []
    v_list = []
    g_list = []

    for p in current_model.parameters():
        if p.grad is None:
            continue
        st = optimizer.state.get(p, None)
        if st is None:
            continue
        if "exp_avg" not in st or "exp_avg_sq" not in st:
            continue
        params.append(p)
        m_list.append(st["exp_avg"])
        v_list.append(st["exp_avg_sq"])
        g_list.append(p.grad)

    if len(params) == 0:
        if is_rank0:
            logging.warning("[slice] No valid parameters with grad + optimizer state.")
        return

    # -----------------------------
    # 1) Construct d_viz that matches the REAL Δθ of your nadamw():
    #    Δθ = -lr*wd*θ  +  [ -lr/bc1 * (m2 / (sqrt(v)/sqrt(bc2) + eps)) ]
    #    with m2 = beta1*m + (1-beta1)*g
    #
    # We'll construct d_viz = Δθ (i.e., already includes lr).
    # Then you can sweep t as a multiplier around 1.0 if you want,
    # BUT keeping your existing t_min/t_max is fine if you interpret
    # it as a "small scaling". Here we keep your interface: phi(theta + t*d_viz_dir),
    # so we will ALSO construct a direction-only vector (normalized) for slice.
    # -----------------------------
    d_tensors = []
    for p, m, v, g in zip(params, m_list, v_list, g_list):
        # denom = sqrt(v)/sqrt(bc2) + eps
        denom = (v.sqrt() / bc2_sqrt).add(eps)
        # m2 = beta1*m + (1-beta1)*g
        m2 = beta1 * m + (1.0 - beta1) * g
        # adaptive part: - lr/bc1 * m2/denom
        dir_adapt = -(1.0 / bc1) * (m2 / denom)
        dir_wd = -weight_decay * p
        d_lr1 = dir_adapt + dir_wd
        d_tensors.append(d_lr1)

    # Direction for slice: use the actual delta as direction (so t is "multiplier")
    # If you want t to be a step size in absolute scale, you'd pass an unscaled direction instead.
    # Here we keep: theta <- theta + t * direction, so direction is delta (one-step update).
    direction = d_tensors

    # Flatten viz direction for comparison
    d_viz_flat = torch.cat([d.detach().flatten() for d in d_tensors])

    # -----------------------------
    # 2) Compute the TRUE Δθ_real via optimizer.step() then rollback
    #    IMPORTANT: this will update exp_avg/exp_avg_sq/step in optimizer state.
    #    To compare apples-to-apples, we do the step/rollback using a COPY of optimizer state.
    #
    #    Minimal practical approach:
    #      - Snapshot params
    #      - Snapshot optimizer.state tensors for the involved params
    #      - Call optimizer.step()
    #      - Compute delta
    #      - Restore params and optimizer.state snapshots
    #
    #    This keeps training unaffected.
    # -----------------------------
    # Snapshot params
    theta_old = [p.detach().clone() for p in params]

    # Snapshot state (for only the params we touch)
    state_backup = []
    for p in params:
        st = optimizer.state[p]
        state_backup.append({
            "step": st["step"].detach().clone(),
            "exp_avg": st["exp_avg"].detach().clone(),
            "exp_avg_sq": st["exp_avg_sq"].detach().clone(),
        })

    # Do a real step (no closure)
    optimizer.step()

    # Compute delta_real (only for our `params` list)
    delta_real_flat = torch.cat([(p.detach() - p0).flatten() for p, p0 in zip(params, theta_old)])

    # Roll back params
    for p, p0 in zip(params, theta_old):
        p.copy_(p0)

    # Roll back optimizer state
    for p, bk in zip(params, state_backup):
        st = optimizer.state[p]
        st["step"].copy_(bk["step"])
        st["exp_avg"].copy_(bk["exp_avg"])
        st["exp_avg_sq"].copy_(bk["exp_avg_sq"])

    # -----------------------------
    # 3) Compare vectors (DDP-mean them for logging stability)
    # -----------------------------
    def compare_update_and_direction(delta_real: torch.Tensor, d_viz: torch.Tensor):
        # Cosine
        denom = (delta_real.norm() * d_viz.norm() + 1e-12)
        cos = torch.dot(delta_real, d_viz) / denom

        # Best scale c: minimize ||delta_real - c*d_viz||
        c = torch.dot(delta_real, d_viz) / (d_viz.norm() ** 2 + 1e-12)

        rel_err = (delta_real - c * d_viz).norm() / (delta_real.norm() + 1e-12)
        return cos, c, rel_err

    # Reduce (mean) across ranks for cleaner numbers
    # (We reduce the scalars after computing them locally.)
    cos, scale_c, rel_err = compare_update_and_direction(delta_real_flat, d_viz_flat)
    cos_g = reduce_mean_scalar(cos)
    scale_g = reduce_mean_scalar(scale_c)
    rel_g = reduce_mean_scalar(rel_err)

    if is_rank0:
        logging.warning(
            f"[slice][compare] step={global_step} "
            f"cos={cos_g:.6f}  scale={scale_g:.6e}  rel_err={rel_g:.6e} "
            f"(expect cos~1, scale~1 if direction is Δθ)"
        )

    # -----------------------------
    # phi(t) definition
    # -----------------------------
    def phi(t: float):
        with torch.no_grad():
            # theta <- theta + t * direction
            for p, d in zip(params, direction):
                p.add_(d, alpha=t)

            logits, _ = workload.model_fn(
                params=current_model,
                augmented_and_preprocessed_input_batch=batch,
                model_state=model_state,
                mode=spec.ForwardPassMode.EVAL,
                rng=rng,
                update_batch_norm=False,
                dropout_rate=0.0,
            )

            loss_dict = workload.loss_fn(
                label_batch=batch["targets"],
                logits_batch=logits,
                mask_batch=batch.get("weights"),
                label_smoothing=0.0,
            )

            phi_t = loss_dict["summed"] / loss_dict["n_valid_examples"]

            # restore parameters
            for p, d in zip(params, direction):
                p.add_(d, alpha=-t)

        return phi_t

    # -----------------------------
    # phi(0) and directional derivative (direction = Δθ, so derivative is along Δθ)
    # -----------------------------
    phi0 = phi(0.0)

    derphi0 = torch.zeros((), device=phi0.device)
    for p, d in zip(params, direction):
        # Using current gradient (from TRAIN) to approximate directional derivative
        derphi0 += torch.sum(p.grad.detach() * d.detach())

    phi0_g = reduce_mean_scalar(phi0)
    derphi0_g = reduce_mean_scalar(derphi0)

    # -----------------------------
    # sweep t (here t is a multiplier around Δθ)
    # Your t_min/t_max were absolute step sizes before; now interpret as multipliers.
    # Keeping your values is OK, but you may want [0, 2] etc.
    # We'll keep as-is per request.
    # -----------------------------
    t_vals = np.logspace(math.log10(t_min), math.log10(t_max), num_points)

    phi_vals = []
    for t in t_vals:
        v_local = phi(float(t))
        v = reduce_mean_scalar(v_local)
        phi_vals.append(v)
    phi_vals = np.array(phi_vals)

    # -----------------------------
    # plotting (rank 0 only)
    # -----------------------------
    if is_rank0:
        os.makedirs(out_dir, exist_ok=True)
        plot_path = os.path.join(out_dir, f"slice_step_{global_step:07d}.png")

        armijo_line = phi0_g + c1 * t_vals * derphi0_g

        plt.figure(figsize=(8, 6))
        plt.plot(t_vals, phi_vals, label="phi(t)", linewidth=2)
        plt.plot(t_vals, armijo_line, "--", label="Armijo line", linewidth=2)
        plt.xscale("log")
        plt.xlabel("t (multiplier of one-step Δθ)")
        plt.ylabel("loss")
        plt.title(f"Update-direction loss slice @ step {global_step}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()

        logging.warning(f"[slice] saved loss slice at step {global_step} -> {plot_path}")


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
  if hasattr(hyperparameters, 'grad_clip'):
    grad_clip = hyperparameters.grad_clip
  else:
    grad_clip = None

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
  # if global_step % 100 == 0:
  #     visualize_update_direction_slice(
  #         current_model,
  #         optimizer_state,
  #         workload,
  #         batch,
  #         model_state,
  #         rng,
  #         global_step,
  #     )


  if grad_clip is not None:
    torch.nn.utils.clip_grad_norm_(
      current_model.parameters(), max_norm=grad_clip
    )
  optimizer_state['optimizer'].step()
  optimizer_state['scheduler'].step()

  # Log training metrics - loss, grad_norm, batch_size.
  if global_step <= 100 or global_step % 500 == 0:
    with torch.no_grad():
      parameters = [p for p in current_model.parameters() if p.grad is not None]
      grad_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2
      )
    if workload.metrics_logger is not None:
      workload.metrics_logger.append_scalar_metrics(
        {
          'loss': loss.item(),
          'grad_norm': grad_norm.item(),
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
    return 512
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
  del workload
  del optimizer_state
  del current_param_container
  del model_state
  del hyperparameters
  del global_step
  del rng
  batch = next(input_queue)
  return batch
