from torch import inf, Tensor
from torch.optim import Optimizer
from encoder import HashEmbedderNative
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    SupportsFloat,
    TypedDict,
    Union,
)

class loss_monitor:
    def __init__(
        self,
        encoder: HashEmbedderNative,
        optimizer: Optimizer,
        mode: Literal["min", "max"] = "min",
        factor=1,
        patience=10,
        threshold=1e-4,
        threshold_mode: Literal["rel", "abs"] = "rel",
        cooldown=0,
        max_sz: int = 30,
        eps=1e-8,
        enc_idx: int = 0,
        error_bound=0.01
    ):
        if factor != 1:
            raise ValueError("Factor should be 1.")
        self.factor = factor

        # Attach encoder
        self.encoder = encoder

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer

        if max_sz > 30:
            raise ValueError("log 2 hash map size should be smaller than 30")
        self.max_sz = max_sz

        self.patience = patience

        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best: float
        self.num_bad_epochs: int
        self.mode_worse: float  # the worse value for the chosen mode
        self.eps = eps
        self.enc_idx = enc_idx
        self.error_bound = error_bound
        # recording for how many times have loss_monitor step() been called
        self.last_epoch = 0
        self._last_sz = encoder.get_log2_hashmap_size()
        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics: SupportsFloat, epoch=None):  # type: ignore[override]
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            if current > self.error_bound:
                self._incre_sz(epoch)
            else:
                print("iter:", epoch, " encoder idx ", self.enc_idx, " increase size ignore because loss has been already below error bound")
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_sz = self.encoder.get_log2_hashmap_size()

    def _incre_sz(self, epoch):
        if self.encoder.get_log2_hashmap_size() < self.max_sz:
            self.encoder.increase_embedding_size_by_two()
            print("iter:", epoch, " encoder idx ", self.enc_idx, " increase hashmap size to 2^", self.encoder.get_log2_hashmap_size())
            self.optimizer.add_param_group({"params": self.encoder.parameters()})
        else:
            print("iter:", epoch, " encoder idx ", self.enc_idx, " has already reached the max hashmap size 2^", self.max_sz)

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        if mode == "min":
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    # def state_dict(self):
    #     return {
    #         key: value for key, value in self.__dict__.items() if key != "optimizer"
    #     }

    # def load_state_dict(self, state_dict):
    #     self.__dict__.update(state_dict)
    #     self._init_is_better(
    #         mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode
    #     )