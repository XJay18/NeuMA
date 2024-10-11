import numpy as np
from abc import abstractmethod
from omegaconf import DictConfig
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import LRScheduler


class Scheduler:
    """Base scheduler"""

    config: DictConfig

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        """Abstract method that returns a scheduler object.

        Args:
            optimizer: The optimizer to use.
            lr_init: The initial learning rate.
        Returns:
            The scheduler object.
        """


class ExponentialDecayScheduler(Scheduler):
    """Exponential decay scheduler with linear warmup. Scheduler first ramps up to `lr_init` in `warmup_steps`
    steps, then exponentially decays to `lr_final` in `max_steps` steps.
    """

    config: DictConfig

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        
        if self.config.get("lr_pre_warmup") is None:
            self.config.lr_pre_warmup = 1e-8
        if self.config.get("warmup_steps") is None:
            self.config.warmup_steps = 0
        if self.config.get("max_steps") is None:
            self.config.max_steps = 1e5
        if self.config.get("ramp") is None:
            self.config.ramp = "linear"

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        if self.config.lr_final is None:
            lr_final = lr_init
        else:
            lr_final = self.config.lr_final

        def func(step):
            if step < self.config.warmup_steps:
                if self.config.ramp == "cosine":
                    lr = self.config.lr_pre_warmup + (lr_init - self.config.lr_pre_warmup) * np.sin(
                        0.5 * np.pi * np.clip(step / self.config.warmup_steps, 0, 1)
                    )
                else:
                    lr = (
                        self.config.lr_pre_warmup
                        + (lr_init - self.config.lr_pre_warmup) * step / self.config.warmup_steps
                    )
            else:
                t = np.clip(
                    (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps), 0, 1
                )
                lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return lr / lr_init  # divided by lr_init because the multiplier is with the initial learning rate

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
        return scheduler


class CosineDecayScheduler(Scheduler):
    """Cosine decay scheduler with linear warmup"""

    config: DictConfig

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        
        if self.config.get("warm_up_end") is None:
            self.config.warm_up_end = 0
        if self.config.get("max_steps") is None:
            self.config.max_steps = 1e5
        if self.config.get("learning_rate_alpha") is None:
            self.config.learning_rate_alpha = 0.05

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        def func(step):
            if step < self.config.warm_up_end:
                learning_factor = step / self.config.warm_up_end
            else:
                alpha = self.config.learning_rate_alpha
                progress = (step - self.config.warm_up_end) / (self.config.max_steps - self.config.warm_up_end)
                learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
            return learning_factor

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
        return scheduler


def fetch_scheduler(config: DictConfig) -> Scheduler:
    """Fetches the scheduler object based on the config.

    Args:
        config: The configuration object.
    Returns:
        The scheduler object.
    """
    if config.type == "exp":
        return ExponentialDecayScheduler(config)
    elif config.type == "cos":
        return CosineDecayScheduler(config)
    else:
        raise ValueError(f"Scheduler {config.type} not supported.")
