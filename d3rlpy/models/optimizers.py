import dataclasses
from typing import Iterable, Sequence, Tuple

from torch import nn
from torch.optim import SGD, Adam, AdamW, Optimizer, RMSprop

from ..serializable_config import DynamicConfig, generate_config_registration

__all__ = [
    "OptimizerFactory",
    "SGDFactory",
    "AdamFactory",
    "AdamWFactory",
    "RMSpropFactory",
    "GPTAdamWFactory",
    "register_optimizer_factory",
    "make_optimizer_field",
]


def _get_parameters_from_named_modules(
    named_modules: Iterable[Tuple[str, nn.Module]]
) -> Sequence[nn.Parameter]:
    # retrieve unique set of parameters
    params_dict = {}
    for _, module in named_modules:
        for param in module.parameters():
            if param not in params_dict:
                params_dict[param] = param
    return list(params_dict.values())


@dataclasses.dataclass()
class OptimizerFactory(DynamicConfig):
    """A factory class that creates an optimizer object in a lazy way.

    The optimizers in algorithms can be configured through this factory class.
    """

    def create(
        self, named_modules: Iterable[Tuple[str, nn.Module]], lr: float
    ) -> Optimizer:
        """Returns an optimizer object.

        Args:
            named_modules (list): List of tuples of module names and modules.
            lr (float): Learning rate.

        Returns:
            torch.optim.Optimizer: an optimizer object.
        """
        raise NotImplementedError


@dataclasses.dataclass()
class SGDFactory(OptimizerFactory):
    """An alias for SGD optimizer.

    .. code-block:: python

        from d3rlpy.optimizers import SGDFactory

        factory = SGDFactory(weight_decay=1e-4)

    Args:
        momentum: momentum factor.
        dampening: dampening for momentum.
        weight_decay: weight decay (L2 penalty).
        nesterov: flag to enable Nesterov momentum.
    """

    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False

    def create(
        self, named_modules: Iterable[Tuple[str, nn.Module]], lr: float
    ) -> SGD:
        return SGD(
            _get_parameters_from_named_modules(named_modules),
            lr=lr,
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
        )

    @staticmethod
    def get_type() -> str:
        return "sgd"


@dataclasses.dataclass()
class AdamFactory(OptimizerFactory):
    """An alias for Adam optimizer.

    .. code-block:: python

        from d3rlpy.optimizers import AdamFactory

        factory = AdamFactory(weight_decay=1e-4)

    Args:
        betas: coefficients used for computing running averages of
            gradient and its square.
        eps: term added to the denominator to improve numerical stability.
        weight_decay: weight decay (L2 penalty).
        amsgrad: flag to use the AMSGrad variant of this algorithm.
    """

    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = False

    def create(
        self, named_modules: Iterable[Tuple[str, nn.Module]], lr: float
    ) -> Adam:
        return Adam(
            _get_parameters_from_named_modules(named_modules),
            lr=lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )

    @staticmethod
    def get_type() -> str:
        return "adam"


@dataclasses.dataclass()
class AdamWFactory(OptimizerFactory):
    """An alias for AdamW optimizer.

    .. code-block:: python

        from d3rlpy.optimizers import AdamWFactory

        factory = AdamWFactory(weight_decay=1e-4)

    Args:
        betas: coefficients used for computing running averages of
            gradient and its square.
        eps: term added to the denominator to improve numerical stability.
        weight_decay: weight decay (L2 penalty).
        amsgrad: flag to use the AMSGrad variant of this algorithm.
    """

    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = False

    def create(
        self, named_modules: Iterable[Tuple[str, nn.Module]], lr: float
    ) -> AdamW:
        return AdamW(
            _get_parameters_from_named_modules(named_modules),
            lr=lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )

    @staticmethod
    def get_type() -> str:
        return "adam_w"


@dataclasses.dataclass()
class RMSpropFactory(OptimizerFactory):
    """An alias for RMSprop optimizer.

    .. code-block:: python

        from d3rlpy.optimizers import RMSpropFactory

        factory = RMSpropFactory(weight_decay=1e-4)

    Args:
        alpha: smoothing constant.
        eps: term added to the denominator to improve numerical stability.
        weight_decay: weight decay (L2 penalty).
        momentum: momentum factor.
        centered: flag to compute the centered RMSProp, the gradient is
            normalized by an estimation of its variance.
    """

    alpha: float = 0.95
    eps: float = 1e-2
    weight_decay: float = 0.0
    momentum: float = 0.0
    centered: bool = True

    def create(
        self, named_modules: Iterable[Tuple[str, nn.Module]], lr: float
    ) -> RMSprop:
        return RMSprop(
            _get_parameters_from_named_modules(named_modules),
            lr=lr,
            alpha=self.alpha,
            eps=self.eps,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            centered=self.centered,
        )

    @staticmethod
    def get_type() -> str:
        return "rmsprop"


@dataclasses.dataclass()
class GPTAdamWFactory(OptimizerFactory):
    """AdamW optimizer for Decision Transformer architectures.

    .. code-block:: python

        from d3rlpy.optimizers import GPTAdamWFactory

        factory = GPTAdamWFactory(weight_decay=1e-4)

    Args:
        betas: coefficients used for computing running averages of
            gradient and its square.
        eps: term added to the denominator to improve numerical stability.
        weight_decay: weight decay (L2 penalty).
        amsgrad: flag to use the AMSGrad variant of this algorithm.
    """

    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = False

    def create(
        self, named_modules: Iterable[Tuple[str, nn.Module]], lr: float
    ) -> AdamW:
        named_modules = list(named_modules)
        params_dict = {}
        decay = set()
        no_decay = set()
        for module_name, module in named_modules:
            for param_name, param in module.named_parameters():
                full_name = (
                    f"{module_name}.{param_name}" if module_name else param_name
                )

                if full_name not in params_dict:
                    params_dict[full_name] = param

                if param_name.endswith("bias"):
                    no_decay.add(full_name)
                elif param_name.endswith("weight") and isinstance(
                    module, (nn.Linear, nn.Conv2d)
                ):
                    decay.add(full_name)
                elif param_name.endswith("weight") and isinstance(
                    module, (nn.LayerNorm, nn.Embedding)
                ):
                    no_decay.add(full_name)

        # add non-catched parameters to no_decay
        all_names = set(params_dict.keys())
        remainings = all_names.difference(decay | no_decay)
        no_decay.update(remainings)
        assert len(decay | no_decay) == len(
            _get_parameters_from_named_modules(named_modules)
        )
        assert len(decay & no_decay) == 0

        optim_groups = [
            {
                "params": [params_dict[name] for name in sorted(list(decay))],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    params_dict[name] for name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        return AdamW(
            optim_groups,
            lr=lr,
            betas=self.betas,
            eps=self.eps,
            amsgrad=self.amsgrad,
        )

    @staticmethod
    def get_type() -> str:
        return "gpt_adam_w"


register_optimizer_factory, make_optimizer_field = generate_config_registration(
    OptimizerFactory, lambda: AdamFactory()
)


register_optimizer_factory(SGDFactory)
register_optimizer_factory(AdamFactory)
register_optimizer_factory(AdamWFactory)
register_optimizer_factory(RMSpropFactory)
register_optimizer_factory(GPTAdamWFactory)
