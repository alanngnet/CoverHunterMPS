#!/usr/bin/env python3
# author:liufeng
# datetime:2021/7/7 10:24 AM
# software: PyCharm

import warnings

import torch


class UserDefineExponentialLR(torch.optim.lr_scheduler._LRScheduler):
    """Decays the learning rate of each parameter group by _gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        min_lr(float): min lr.
        last_epoch (int): The index of last epoch. Default: -1.

    """

    def __init__(
        self,
        optimizer,
        gamma,
        min_lr,
        last_epoch=-1,
        warmup=False,
        warmup_steps=5000,
        hold_steps=0,
    ) -> None:
        self.gamma = gamma
        self.min_lr = min_lr
        self._warmup = warmup
        self._hold_steps = (
            hold_steps  # steps to hold at initial LR before decay begins
        )
        # Store step count at init to calculate relative progress for hold_steps
        self._internal_step_count = 0

        super().__init__(optimizer, last_epoch)

        if warmup:
            print(f"Using Warmup for Learning: {warmup_steps}")
            self._warmup_steps = warmup_steps
            self.get_lr()
        elif hold_steps > 0:
            print(
                f"Using LR hold period: {hold_steps} steps before decay (starting from step {self._internal_step_count})"
            )

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return self.base_lrs

        if not self._warmup:
            # use getattr to handle checkpoints created before hold_steps was implemented
            lr = [group["lr"] for group in self.optimizer.param_groups]
            if self._hold_steps > 0:
                if self._internal_step_count <= self._hold_steps:
                    # Hold at current LR during plateau period (no decay)
                    return lr
        else:
            # use getattr to handle checkpoints created before hold_steps was implemented
            local_step = getattr(self.optimizer, "_step_count", 0)
            lr = [group["lr"] for group in self.optimizer.param_groups]
            if local_step <= self._warmup_steps + 1:
                lr[0] = self.base_lrs[0] * local_step / self._warmup_steps
                # print("debug:", self.base_lrs[0], local_step / self._warmup_steps, lr[0])
                return lr
        # Normal exponential decay
        lr[0] = lr[0] * self.gamma
        lr[0] = lr[0] if lr[0] > self.min_lr else self.min_lr
        return lr

    def increment_step(self):
        """Track optimizer steps for hold_steps calculation."""
        self._internal_step_count += 1

    def set_step_count(self, count):
        """Restore step count from checkpoint."""
        self._internal_step_count = count

    def _get_closed_form_lr(self):
        return [
            base_lr * self.gamma**self.last_epoch for base_lr in self.base_lrs
        ]
