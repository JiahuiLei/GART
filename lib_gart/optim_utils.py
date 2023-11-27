import numpy as np
import torch
from torch import nn


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def get_expon_lr_func_interval(
    init_step, final_step, lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0
):
    def helper(step):
        if (
            step < init_step
            or step > final_step
            or (lr_init == 0.0 and lr_final == 0.0)
        ):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip((step - init_step) / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip((step - init_step) / (final_step - init_step), 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def update_learning_rate(lr, names, optimizer):
    if not isinstance(names, list):
        names = [names]
    for param_group in optimizer.param_groups:
        if param_group["name"] in names:
            param_group["lr"] = lr
            # print("debug")
            # print(f"Update {name} lr to {lr}")
    # have to iterate over all param_groups, because some param_groups may not have name
    return lr


@torch.no_grad()
def cat_tensors_to_optimizer(optimizer, tensors_dict):
    optimizable_tensors = {}
    N = -1
    for group in optimizer.param_groups:
        if group["name"] not in tensors_dict.keys():
            # print(f"Warning: {group['name']} not in optimizer, skip")
            continue
        assert len(group["params"]) == 1, f"{group['name']} has more than one param"
        extension_tensor = tensors_dict[group["name"]]
        # print(group["name"])
        stored_state = optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat(
                (stored_state["exp_avg"].clone(), torch.zeros_like(extension_tensor)),
                dim=0,
            )
            stored_state["exp_avg_sq"] = torch.cat(
                (
                    stored_state["exp_avg_sq"].clone(),
                    torch.zeros_like(extension_tensor),
                ),
                dim=0,
            )

            del optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(
                torch.cat(
                    (
                        group["params"][0].clone().contiguous(),
                        extension_tensor.contiguous(),
                    ),
                    dim=0,
                )
                .contiguous()
                .requires_grad_(True)
            )
            optimizable_tensors[group["name"]] = group["params"][0]
            optimizer.state[group["params"][0]] = stored_state
        else:
            group["params"][0] = nn.Parameter(
                torch.cat(
                    (
                        group["params"][0].clone().contiguous(),
                        extension_tensor.contiguous(),
                    ),
                    dim=0,
                ).requires_grad_(True)
            )
            optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors


def prune_optimizer(optimizer, mask, exclude_names=[]):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        # print(group["name"])
        if group["name"] in exclude_names or len(group["params"]) == 0:
            continue
        stored_state = optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][mask]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

            del optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(
                (group["params"][0][mask].requires_grad_(True))
            )
            optimizer.state[group["params"][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]
        else:
            group["params"][0] = nn.Parameter(
                group["params"][0][mask].requires_grad_(True)
            )
            optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors


def replace_tensor_to_optimizer(optimizer, tensor, name):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        if group["name"] == name:
            stored_state = optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del optimizer.state[group["params"][0]]
                optimizer.state[group["params"][0]] = stored_state
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors
