# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
import math
import random
import torch
import numpy as np
import torch.distributed as dist
from absl import logging
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt

class LineSearchScheduler():
    def __init__(self, optimizer, start_lr, model_paras, num_search=16, optimizer_type="SGD", injection=True, search_mode="backtrack"):
        """
        num_search: maximum number of searches
        start_lr: maximum LR to start if backtrack/ minimum LR to start if forward
        optimizer_type: Option: SGD, SGD_momentum, Adam
        """



        self.optimizer = optimizer
        self.num_search = num_search
        self.start_lr = start_lr
        # self.model = model
        self.optimizer_type = optimizer_type
        self.injection=injection
        self.prev_fvals = deque(maxlen=2)
        self.prev_alpha = start_lr
        self.search_mode = search_mode
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.start_lr
        self.paras = model_paras
        # self.injection_distribution = self._generate_long_tail_distribution()
        self.rule = self.get_potential_update_direction()

    
    
    def _generate_long_tail_distribution(self):
        distribution = [1.0] * 80 + [random.uniform(1.0, 2.0) for _ in range(20)]
        random.shuffle(distribution)
        return distribution
    
    def state_dict(self):
        return {
            'last_lr': self.prev_alpha,
        }
    
    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, dict):
            raise TypeError(f"Expected dict, got {type(state_dict)}")

        self.prev_alpha = state_dict.get('last_lr', self.start_lr)

    def get_potential_update_direction(self, fallback_to_neg_grad=True):
        if self.optimizer_type == "Adam":
            return self.get_potential_adam_update_direction(fallback_to_neg_grad)
        elif self.optimizer_type == "SGD_momentum":
            return self.get_potential_sgd_momentum_update_direction(fallback_to_neg_grad)
        elif self.optimizer_type == "SGD":
            return self.get_potential_sgd_update_direction(fallback_to_neg_grad)
        else:
            raise ValueError(f"Unknown optimizer_type {self.optimizer_type}")
        

    def get_potential_sgd_update_direction(self, fallback_to_neg_grad=True):
        def rule(p):
            g = p.grad
            if g is None:
                return torch.zeros_like(p)
            return -g
        return rule

    def get_potential_sgd_momentum_update_direction(self, fallback_to_neg_grad=True):
        pg0 = self.optimizer.param_groups[0]
        momentum = pg0.get("momentum", 0.0)

        def rule(p):
            g = p.grad
            if g is None:
                return torch.zeros_like(p)

            st = self.optimizer.state.get(p, {})
            if "momentum_buffer" in st:
                v = st["momentum_buffer"]
                return -(momentum * v + g)
            else:
                if fallback_to_neg_grad:
                    return -g
                else:
                    return torch.zeros_like(p)

        return rule

    def get_potential_adam_update_direction(self, fallback_to_neg_grad=True):
        pg0 = self.optimizer.param_groups[0]
        eps = pg0.get("eps", 1e-8)
        beta1, beta2 = pg0.get("betas", (0.9, 0.999))
        wd = pg0.get("weight_decay", 0)


        def rule(p):
            g = p.grad
            if g is None:
                return torch.zeros_like(p)

            st = self.optimizer.state.get(p, {})
            if (
                "exp_avg" in st
                and "exp_avg_sq" in st
                and st.get("step", 0) > 0
            ):

      
                m = st["exp_avg"]
                v = st["exp_avg_sq"]
                t = st["step"] + 1

                # mf = m.flatten()
                # vf = v.flatten()
                # gf = g.flatten()

                # logging.warning(
                #     f"t={t} | "
                #     f"g[:10]={gf[:10].tolist()} | "
                #     f"m[:10]={mf[:10].tolist()} | "
                #     f"v[:10]={vf[:10].tolist()} | "
                #     f"||g||={gf.norm().item():.3e} | "
                #     f"||m||={mf.norm().item():.3e} | "
                #     f"||v||={vf.norm().item():.3e}, t={t}"
                # )
                m_new = beta1 * m + (1 - beta1) * g
                v_new = beta2 * v + (1 - beta2) * (g * g)

                m_hat = m_new / (1 - beta1 ** t)
                v_hat = v_new / (1 - beta2 ** t)

                return -m_hat / (v_hat.sqrt() + eps) - wd * p

            else:
                if fallback_to_neg_grad:
                    g = p.grad
                    # gf = g.flatten()
                    # pf = p.flatten()

                    # term1 = g / (g.abs() + eps)      # g-normalized
                    # term2 = wd * p                  # weight decay term
                    # df = (term1 - term2).flatten()

                    # logging.warning(
                    #     f"(g/(|g|+eps) - wd*p)[:10]={df[:10].tolist()} | "
                    #     f"g_norm[:10]={gf[:10].tolist()} | "
                    #     f"wd*p[:10]={(wd*pf)[:10].tolist()} | "
                    #     f"||g_norm||={term1.flatten().norm().item():.3e} | "
                    #     f"||wd*p||={(wd*pf).norm().item():.3e}"
                    # )
                    return - g / (g.abs() + eps) - wd * p
                else:
                    return torch.zeros_like(p)
        return rule

    @torch.no_grad()
    def update_model(self, alpha):
        """
        Trial update: p <- p + alpha * rule(p)
        """
        # max_d = 0.0
        cached_dirs = {}
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # logging.warning("GRADIENT IS NONE!!! at 0")
                    continue
                # d = self.rule(p)
                # max_d = max(max_d, d.abs().max().item())
                if self.optimizer.param_groups[0].get("weight_decay", 0) != 0:
                    # logging.warning("weight decay")
                    if p not in cached_dirs:
                        d = self.rule(p).detach().clone()
                        cached_dirs[p] = d
                    else:
                        d = cached_dirs[p]
                    p.add_(d, alpha=alpha)  
                else:
                    # logging.warning("no_weight decay")
                    p.add_(self.rule(p), alpha=alpha)  

        return cached_dirs
        # logging.warning(f"[debug] alpha={alpha}, max|d|={max_d}")

    @torch.no_grad()
    def restore_model(self, alpha, cached_dirs):
        if len(cached_dirs) != 0:
            for p, d in cached_dirs.items():
                p.add_(d, alpha=-alpha)
        else:
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    d = self.rule(p)
                    p.add_(d, alpha=-alpha)
  




    # def check_optimizer_step_vs_rule(self, 
    #     optimizer,
    #     rule_fn,
    #     lr,
    #     rollback=True,
    #     prefix="[LineSearchScheduler]"
    # ):
    #     """
    #     Diagnose whether optimizer.step() follows rule_fn direction.

    #     Args:
    #         optimizer: torch.optim.Optimizer
    #         rule_fn: callable(p) -> direction tensor (same shape as p)
    #         lr: current learning rate (scalar)
    #         rollback: if True, restore parameters after check
    #         prefix: logging prefix

    #     Logs:
    #         cosine similarity between actual update direction and rule direction
    #     """
    #     params = []
    #     for group in optimizer.param_groups:
    #         for p in group["params"]:
    #             if p.requires_grad:
    #                 params.append(p)

    #     if len(params) == 0:
    #         logging.warning(f"{prefix} no parameters found, skip check")
    #         return

    #     old_params = [p.detach().clone() for p in params]

    #     if rollback:
    #         opt_state = {
    #             k: v.copy() if isinstance(v, dict) else v
    #             for k, v in optimizer.state_dict().items()
    #         }

    #     rule_dirs = []
    #     with torch.no_grad():
    #         for p in params:
    #             if p.grad is None:
    #                 continue
    #             d = rule_fn(p)
    #             rule_dirs.append(d.detach().flatten())

    #     if len(rule_dirs) == 0:
    #         logging.warning(f"{prefix} empty rule directions, skip check")
    #         return

    #     rule_vec = torch.cat(rule_dirs)

    #     optimizer.step()


    #     actual_dirs = []
    #     with torch.no_grad():
    #         for p, p_old in zip(params, old_params):
    #             d_actual = (p.detach() - p_old) / lr
    #             actual_dirs.append(d_actual.flatten())

    #     actual_vec = torch.cat(actual_dirs)

    #     cos = cosine_similarity(
    #         rule_vec.unsqueeze(0),
    #         actual_vec.unsqueeze(0),
    #         dim=1
    #     ).item()

    #     logging.warning(
    #         f"{prefix} optimizer-step vs rule cosine = {cos:.6f}"
    #     )

    #     if rollback:
    #         for p, p_old in zip(params, old_params):
    #             p.data.copy_(p_old)
    #         optimizer.load_state_dict(opt_state)
    @torch.no_grad()
    def test_update_restore_max_diff(self, alpha):
        """
        Test max parameter difference after:
            update_model(alpha) -> closure(require_grad=False) -> restore_model(alpha)

        Returns:
            max |Δparam|
        """
        params = self.paras
        backup = [p.detach().clone() for p in params]


        cached_dirs = self.update_model(alpha)
        self.restore_model(alpha, cached_dirs)

        max_diff = 0.0
        for p, p0 in zip(params, backup):
            diff = (p.detach() - p0).abs().max().item()
            max_diff = max(max_diff, diff)

        print(f"[TEST] alpha={alpha:.3e}, max |param diff| = {max_diff:.3e}")
        return max_diff

        

    def step(self, closure, condition="armijo", c1=0.6, factor=0.5, amax=50.0, step=0, interval=100):
        """
        condition: Line Search condition. Option: armijo,
        search_mode: Option: backtracking, forward, interpolate
        factor: used for searching. (growing/shrinking LR)
        c1: parameter for armijo rule
        c2: parameter for wolfe-condition
        interval: perform line search every {interval} steps.
        """


        self.optimizer.zero_grad(set_to_none=True)
        loss = closure(require_grad=True)

        inner = 0.0
        with torch.no_grad():
            for group in self.optimizer.param_groups:
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        d = self.rule(p)       
                        inner += torch.sum(p.grad * d)

        phi0, derphi0 = loss, inner.detach()

        alpha = self.optimizer.param_groups[0]["lr"]
        # self.test_update_restore_max_diff(alpha=alpha)
        # self.check_optimizer_step_vs_rule(
        #     optimizer=self.optimizer,
        #     rule_fn=self.rule,
        #     lr=alpha,
        #     rollback=True
        # )
        
        if derphi0 > 0: 
            derphi0 =- derphi0
            logging.warning("ASCENT!!!")

        # xk = [p.detach().clone() for p in self.paras]
        # gk = [p.grad.detach().clone() if p.grad is not None else None for p in self.paras]
        @torch.no_grad()
        def phi(alpha):
            cached_dirs = self.update_model(alpha)
            val = closure(require_grad=False)
            self.restore_model(alpha, cached_dirs)
            return val
    

        alpha, fc, _ = line_search_armijo(
                    f=phi,
                    derphi0=derphi0,
                    phi0=phi0,
                    args=(),
                    c1=c1,
                    alpha0=self.prev_alpha,
                    num_search=self.num_search,
                    step=step,
                    search_mode=self.search_mode,
                    factor=factor
                )
        
        if alpha is None or not np.isfinite(alpha) or alpha <= 0:
            current_lr = self.optimizer.param_groups[0]["lr"]
            alpha = float(current_lr if np.isfinite(current_lr) and current_lr > 0 else self.start_lr)

        print(f"[LineSearchScheduler] alpha={alpha:.6g}, fc={fc}")
        
        for param_group in self.optimizer.param_groups:
                param_group['lr'] = alpha

        self.prev_alpha = alpha
        self.optimizer.step()
    




def line_search_armijo(f, derphi0, phi0, args=(), c1=1e-4, alpha0=1, num_search=16, step=0, search_mode="backtrack", factor=0.5):
    """Minimize over alpha, the function ``f(xk+alpha pk)``.

    Parameters
    ----------
    f : callable
        Function to be minimized.
    xk : array_like
        Current point.
    pk : array_like
        Search direction.
    gfk : array_like
        Gradient of `f` at point `xk`.
    old_fval : float
        Value of `f` at point `xk`.
    args : tuple, optional
        Optional arguments.
    c1 : float, optional
        Value to control stopping criterion.
    phi0 : scaler
        current loss
    derphi0 : scalar,
        inner product of dk and gradient
    alpha0 : scalar, optional
        Value of `alpha` at start of the optimization.

    Returns
    -------
    alpha
    f_count
    f_val_at_alpha

    Notes
    -----
    Uses the interpolation algorithm (Armijo backtracking) as suggested by
    Wright and Nocedal in 'Numerical Optimization', 1999, pp. 56-57

    """
    fc = [0]

    def phi(alpha1):
        fc[0] += 1
        value = f(alpha1)
        return value

    use_ddp = dist.is_initialized() and dist.get_world_size() > 1
    if use_ddp:
            alpha, phi1 = search_bisection_ddp(phi, phi0, derphi0, c1=c1,
                                            old_alpha=alpha0, grow=1/factor, shrink=factor, amax=1, num_search=num_search)
    else:
            alpha, phi1 = search_bisection(phi, phi0, derphi0, c1=c1,
                                            old_alpha=alpha0, grow=1/factor, shrink=factor, amax=1, num_search=num_search)
            # alpha, phi1 = search_backtracking_visual(phi, phi0, derphi0, c1=c1,
            #                                   alpha=alpha0, shrink=factor, plot_path=f"backtracking_{step}.png")
            
    
    # if search_mode == "backtrack":
    #     # alpha, phi1 = search_backtracking_visual(phi, phi0, derphi0, c1=c1,
    #     #                                 alpha=alpha0, shrink=factor, plot_path=f"backtracking_{step}.png")
    #     alpha, phi1 = search_backtracking(phi, phi0, derphi0, c1=c1,
    #                                     alpha=alpha0, shrink=factor, num_search=num_search)
    # elif search_mode == "forward":
    #     alpha, phi1 = search_forward(phi, phi0, derphi0, c1=c1,
    #                                        alpha=alpha0, grow=1/factor, shrink=factor, amax=1, num_search=num_search)
    # elif search_mode == "bisection":
    #      use_ddp = dist.is_initialized() and dist.get_world_size() > 1
    #      if use_ddp:
    #         alpha, phi1 = search_bisection_ddp(phi, phi0, derphi0, c1=c1,
    #                                         old_alpha=alpha0, grow=1/factor, shrink=factor, amax=1, num_search=num_search)
    #      else:
    #         alpha, phi1 = search_bisection(phi, phi0, derphi0, c1=c1,
    #                                         old_alpha=alpha0, grow=1/factor, shrink=factor, amax=1, num_search=num_search)
    # else:
    #     alpha, phi1 = scalar_search_armijo(phi, phi0, derphi0, c1=c1,
    #                                     alpha0=alpha0)
    return alpha, fc[0], phi1




# def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
#     """Minimize over alpha, the function ``phi(alpha)``.

#     Uses the interpolation algorithm (Armijo backtracking) as suggested by
#     Wright and Nocedal in 'Numerical Optimization', 1999, pp. 56-57

#     alpha > 0 is assumed to be a descent direction.

#     Returns
#     -------
#     alpha
#     phi1

#     """
#     phi_a0 = phi(alpha0)
#     if phi_a0 <= phi0 + c1*alpha0*derphi0:
#         return alpha0, phi_a0

#     # Otherwise, compute the minimizer of a quadratic interpolant:

#     alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
#     phi_a1 = phi(alpha1)

#     if (phi_a1 <= phi0 + c1*alpha1*derphi0):
#         return alpha1, phi_a1

#     # Otherwise, loop with cubic interpolation until we find an alpha which
#     # satisfies the first Wolfe condition (since we are backtracking, we will
#     # assume that the value of alpha is not too small and satisfies the second
#     # condition.

#     while alpha1 > amin:       # we are assuming alpha>0 is a descent direction
#         factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
#         a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
#             alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
#         a = a / factor
#         b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
#             alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
#         b = b / factor

#         alpha2 = (-b + np.sqrt(abs(b**2 - 3 * a * derphi0))) / (3.0*a)
#         phi_a2 = phi(alpha2)

#         if (phi_a2 <= phi0 + c1*alpha2*derphi0):
#             return alpha2, phi_a2

#         if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
#             alpha2 = alpha1 / 2.0

#         alpha0 = alpha1
#         alpha1 = alpha2
#         phi_a0 = phi_a1
#         phi_a1 = phi_a2

#     # Failed to find a suitable step length
#     return None, phi_a1


# def search_forward(phi, phi0, derphi0, c1, alpha, grow, shrink, amax, num_search):

#     # Try expanding
#     phi_a = phi(alpha)
#     count = 0
#     while phi_a <= phi0 + c1 * alpha * derphi0 and alpha < amax and count < num_search:
#         alpha *= grow
#         phi_a = phi(alpha)
#         count += 1

#     # Overshoot → shrink until good
#     while phi_a > phi0 + c1 * alpha * derphi0:
#         alpha *= shrink
#         phi_a = phi(alpha)

#     return alpha, phi_a

def search_bisection_ddp(phi, phi0, derphi0, c1,
                     old_alpha, grow=2.0, shrink=0.5,
                     amax=1, num_search=10):

    use_ddp = dist.is_initialized() and dist.get_world_size() > 1
    rank = dist.get_rank() if use_ddp else 0
    device = (
        torch.device("cuda")
        if use_ddp
        else None
    )
    alpha = old_alpha
    phi_a = phi(alpha)

    if rank == 0:
        armijo_old_work = phi_a <= phi0 + c1 * alpha * derphi0
    armijo_flag = torch.tensor(
                [int(armijo_old_work)] if rank == 0 else [0],
                device=device,
            )
    # logging.warning(f"[rank {rank}]  Before old armijo_broadcast")
    dist.broadcast(armijo_flag, src=0)
    # logging.warning(f"[rank {rank}]  After old armijo_broadcast")
    armijo_old_work = bool(armijo_flag.item())

    # # logging.warning(f'line search: old armijo={armijo_old},rank={rank}')

    if armijo_old_work:
            for _ in range(num_search): 
            
                new_alpha = alpha * grow

                if rank == 0:
                    exceed = new_alpha >= amax
                exceed_flag = torch.tensor(
                    [int(exceed)] if rank == 0 else [0],
                    device=device,
                )
                # logging.warning(f"[rank {rank}]  Before  exceed_broadcast")
                dist.broadcast(exceed_flag, src=0)
                # logging.warning(f"[rank {rank}]  After  exceed_broadcast")
                exceed = bool(exceed_flag.item())
                # logging.warning(f'line search: exceed={exceed},rank={rank}')
                if exceed:
                    break

                new_phi = phi(new_alpha)
                # logging.warning(f'line search: loss={new_phi},rank={rank}')
                # logging.warning(f'line search: new alpha={new_alpha},rank={rank}')


                if rank == 0:
                    accept = new_phi > phi0 + c1 * new_alpha * derphi0
                accept_flag = torch.tensor(
                            [int(accept)] if rank == 0 else [0],
                            device=device,
                        )
                # logging.warning(f"[rank {rank}]  Before  accept_broadcast")
                dist.broadcast(accept_flag, src=0)
                # logging.warning(f"[rank {rank}]  After  accept_broadcast")
                accept = bool(accept_flag.item())
                # logging.warning(f'line search: accept={accept},rank={rank}')
                if accept:
                    break

        
                alpha = new_alpha
                phi_a = new_phi

            return alpha, phi_a


    else:
            for _ in range(num_search): 
    
                new_alpha = alpha * shrink
                new_phi = phi(new_alpha)
                # logging.warning(f'line search: loss={new_phi},rank={rank}')
                # logging.warning(f'line search: new alpha={new_alpha},rank={rank}')


                if rank == 0:
                    accept = new_phi <= phi0 + c1 * new_alpha * derphi0
                accept_flag = torch.tensor(
                            [int(accept)] if rank == 0 else [0],
                            device=device,
                        )
                # logging.warning(f"[rank {rank}]  Before  accept_broadcast")
                dist.broadcast(accept_flag, src=0)
                # logging.warning(f"[rank {rank}]  After  accept_broadcast")
                accept = bool(accept_flag.item())
                # logging.warning(f'line search: accept={accept},rank={rank}')
                if accept:
                    return new_alpha, new_phi
                


                alpha = new_alpha
                phi_a = new_phi

            return alpha, phi_a


def search_bisection(phi, phi0, derphi0, c1,
                     old_alpha, grow=2.0, shrink=0.5,
                     amax=1, num_search=10):

    alpha = old_alpha
    phi_a = phi(alpha)

    armijo_old = phi_a <= phi0 + c1 * alpha * derphi0

    if armijo_old:
        for _ in range(num_search): 
        
            new_alpha = alpha * grow
            if new_alpha >= amax:
                break

            new_phi = phi(new_alpha)


            if new_phi > phi0 + c1 * new_alpha * derphi0:
                break

    
            alpha = new_alpha
            phi_a = new_phi

        return alpha, phi_a


    else:
        for _ in range(num_search): 
            print(_)
   
            new_alpha = alpha * shrink
            new_phi = phi(new_alpha)

        
            if new_phi <= phi0 + c1 * new_alpha * derphi0:
                return new_alpha, new_phi

            alpha = new_alpha
            phi_a = new_phi

        return alpha, phi_a


def search_backtracking(phi, phi0, derphi0, c1, alpha, shrink, num_search):
    phi_a = phi(alpha)
    count = 0
    while phi_a > phi0 + c1 * alpha * derphi0 and count < num_search:
        alpha *= shrink
        phi_a = phi(alpha)
        count += 1
    return alpha, phi_a




def search_backtracking_visual(
    phi, phi0, derphi0,
    c1, alpha, shrink,
    plot_path="backtracking_ls.png",
    t_min=0.0, t_max=1.0, num_points=30
):
    explored = []  

    # --- Backtracking loop ---
    phi_a = phi(alpha)
    explored.append((alpha, phi_a))
    count = 0

    while phi_a > phi0 + c1 * alpha * derphi0:
        count += 1
        if count > 4:
            break
        logging.warning(count)
        
        alpha *= shrink
        phi_a = phi(alpha)
        explored.append((alpha, phi_a))

    chosen_alpha, chosen_phi = alpha, phi_a


    t_vals = np.linspace(t_min, t_max, num_points)
    phi_vals_list = []
    for t in t_vals:
        value = phi(t)   
        logging.warning(f"loss {value} at t={t}")
        phi_vals_list.append(value)
    phi_vals = np.array(phi_vals_list)


    armijo_line = phi0 + c1 * t_vals * derphi0.item()


    plt.figure(figsize=(8, 6))


    plt.plot(t_vals, phi_vals, label="phi(t)", linewidth=2)


    plt.plot(t_vals, armijo_line, "--", label="Armijo line", linewidth=2)


    for i, (a, v) in enumerate(explored):
        plt.scatter(a, v, color="red", s=60)
        if i == 0:
            plt.annotate("init", (a, v), textcoords="offset points", xytext=(5, 5))
        else:
            plt.annotate(f"bt {i}", (a, v), textcoords="offset points", xytext=(5, 5))

    
    plt.scatter(chosen_alpha, chosen_phi, color="blue", s=120, marker="x", label="chosen alpha")

    # labels
    plt.xlabel("t (step size)")
    plt.ylabel("phi(t)")
    plt.title("Backtracking Line Search Visualization")
    plt.grid(True)
    plt.legend()

    # Save
    plt.savefig(plot_path, dpi=200)
    plt.close()

    return chosen_alpha, chosen_phi