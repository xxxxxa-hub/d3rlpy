from typing import List, Optional, Sequence, Tuple, Union

import torch

from .base import ContinuousQFunctionForwarder, DiscreteQFunctionForwarder

__all__ = [
    "DiscreteEnsembleQFunctionForwarder",
    "ContinuousEnsembleQFunctionForwarder",
    "compute_max_with_n_actions",
    "compute_max_with_n_actions_and_indices",
]


def _reduce_ensemble(
    y: torch.Tensor, reduction: str = "min", dim: int = 0, lam: float = 0.75
) -> torch.Tensor:
    if reduction == "min":
        return y.min(dim=dim).values
    elif reduction == "max":
        return y.max(dim=dim).values
    elif reduction == "mean":
        return y.mean(dim=dim)
    elif reduction == "none":
        return y
    elif reduction == "mix":
        max_values = y.max(dim=dim).values
        min_values = y.min(dim=dim).values
        return lam * min_values + (1.0 - lam) * max_values
    raise ValueError


def _gather_quantiles_by_indices(
    y: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    # TODO: implement this in general case
    if y.dim() == 3:
        # (N, batch, n_quantiles) -> (batch, n_quantiles)
        return y.transpose(0, 1)[torch.arange(y.shape[1]), indices]
    elif y.dim() == 4:
        # (N, batch, action, n_quantiles) -> (batch, action, N, n_quantiles)
        transposed_y = y.transpose(0, 1).transpose(1, 2)
        # (batch, action, N, n_quantiles) -> (batch * action, N, n_quantiles)
        flat_y = transposed_y.reshape(-1, y.shape[0], y.shape[3])
        head_indices = torch.arange(y.shape[1] * y.shape[2])
        # (batch * action, N, n_quantiles) -> (batch * action, n_quantiles)
        gathered_y = flat_y[head_indices, indices.view(-1)]
        # (batch * action, n_quantiles) -> (batch, action, n_quantiles)
        return gathered_y.view(y.shape[1], y.shape[2], -1)
    raise ValueError


def _reduce_quantile_ensemble(
    y: torch.Tensor, reduction: str = "min", dim: int = 0, lam: float = 0.75
) -> torch.Tensor:
    # reduction beased on expectation
    mean = y.mean(dim=-1)
    if reduction == "min":
        indices = mean.min(dim=dim).indices
        return _gather_quantiles_by_indices(y, indices)
    elif reduction == "max":
        indices = mean.max(dim=dim).indices
        return _gather_quantiles_by_indices(y, indices)
    elif reduction == "none":
        return y
    elif reduction == "mix":
        min_indices = mean.min(dim=dim).indices
        max_indices = mean.max(dim=dim).indices
        min_values = _gather_quantiles_by_indices(y, min_indices)
        max_values = _gather_quantiles_by_indices(y, max_indices)
        return lam * min_values + (1.0 - lam) * max_values
    raise ValueError


def compute_ensemble_q_function_error(
    forwarders: Union[
        Sequence[DiscreteQFunctionForwarder],
        Sequence[ContinuousQFunctionForwarder],
    ],
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    target: torch.Tensor,
    masks: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    assert target.ndim == 2
    td_sum = torch.tensor(
        0.0,
        dtype=torch.float32,
        device=states.device,
    )
    for forwarder in forwarders:
        loss = forwarder.compute_error(
            states=states,
            actions=actions,
            rewards=rewards,
            target=target,
            masks=masks,
            gamma=gamma,
            reduction="none",
        )
        td_sum += loss.mean()
    return td_sum


def compute_ensemble_q_function_target(
    forwarders: Union[
        Sequence[DiscreteQFunctionForwarder],
        Sequence[ContinuousQFunctionForwarder],
    ],
    action_size: int,
    x: torch.Tensor,
    action: Optional[torch.Tensor] = None,
    reduction: str = "min",
    lam: float = 0.75,
) -> torch.Tensor:
    values_list: List[torch.Tensor] = []
    for forwarder in forwarders:
        if isinstance(forwarder, ContinuousQFunctionForwarder):
            assert action is not None
            target = forwarder.compute_target(x, action)
        else:
            target = forwarder.compute_target(x, action)
        values_list.append(target.reshape(1, x.shape[0], -1))

    values = torch.cat(values_list, dim=0)

    if action is None:
        # mean Q function
        if values.shape[2] == action_size:
            return _reduce_ensemble(values, reduction)
        # distributional Q function
        n_q_funcs = values.shape[0]
        values = values.view(n_q_funcs, x.shape[0], action_size, -1)
        return _reduce_quantile_ensemble(values, reduction)

    if values.shape[2] == 1:
        return _reduce_ensemble(values, reduction, lam=lam)

    return _reduce_quantile_ensemble(values, reduction, lam=lam)


class DiscreteEnsembleQFunctionForwarder:
    _forwarders: Sequence[DiscreteQFunctionForwarder]
    _action_size: int

    def __init__(
        self, forwarders: Sequence[DiscreteQFunctionForwarder], action_size: int
    ):
        self._forwarders = forwarders
        self._action_size = action_size

    def compute_expected_q(
        self, x: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        values = []
        for forwarder in self._forwarders:
            value = forwarder.compute_expected_q(x)
            values.append(value.view(1, x.shape[0], self._action_size))
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)

    def compute_error(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: torch.Tensor,
        terminals: torch.Tensor,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        return compute_ensemble_q_function_error(
            forwarders=self._forwarders,
            observations=observations,
            actions=actions,
            rewards=rewards,
            target=target,
            terminals=terminals,
            gamma=gamma,
        )

    def compute_target(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        reduction: str = "min",
        lam: float = 0.75,
    ) -> torch.Tensor:
        return compute_ensemble_q_function_target(
            forwarders=self._forwarders,
            action_size=self._action_size,
            x=x,
            action=action,
            reduction=reduction,
            lam=lam,
        )

    @property
    def forwarders(self) -> Sequence[DiscreteQFunctionForwarder]:
        return self._forwarders


class ContinuousEnsembleQFunctionForwarder:
    _forwarders: Sequence[ContinuousQFunctionForwarder]
    _action_size: int

    def __init__(
        self,
        forwarders: Sequence[ContinuousQFunctionForwarder],
        action_size: int,
    ):
        self._forwarders = forwarders
        self._action_size = action_size

    def compute_expected_q(
        self, x: torch.Tensor, action: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        values = []
        for forwarder in self._forwarders:
            value = forwarder.compute_expected_q(x, action)
            values.append(value.view(1, x.shape[0], 1))
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)

    def compute_error(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: torch.Tensor,
        masks: torch.Tensor,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        return compute_ensemble_q_function_error(
            forwarders=self._forwarders,
            states=states,
            actions=actions,
            rewards=rewards,
            target=target,
            masks=masks,
            gamma=gamma,
        )

    def compute_target(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        reduction: str = "min",
        lam: float = 0.75,
    ) -> torch.Tensor:
        return compute_ensemble_q_function_target(
            forwarders=self._forwarders,
            action_size=self._action_size,
            x=x,
            action=action,
            reduction=reduction,
            lam=lam,
        )

    @property
    def forwarders(self) -> Sequence[ContinuousQFunctionForwarder]:
        return self._forwarders


def compute_max_with_n_actions_and_indices(
    x: torch.Tensor,
    actions: torch.Tensor,
    forwarder: ContinuousEnsembleQFunctionForwarder,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns weighted target value from sampled actions.

    This calculation is proposed in BCQ paper for the first time.
    `x` should be shaped with `(batch, dim_obs)`.
    `actions` should be shaped with `(batch, N, dim_action)`.
    """
    batch_size = actions.shape[0]
    n_critics = len(forwarder.forwarders)
    n_actions = actions.shape[1]

    # (batch, observation) -> (batch, n, observation)
    expanded_x = x.expand(n_actions, *x.shape).transpose(0, 1)
    # (batch * n, observation)
    flat_x = expanded_x.reshape(-1, *x.shape[1:])
    # (batch, n, action) -> (batch * n, action)
    flat_actions = actions.reshape(batch_size * n_actions, -1)

    # estimate values while taking care of quantiles
    flat_values = forwarder.compute_target(flat_x, flat_actions, "none")
    # reshape to (n_ensembles, batch_size, n, -1)
    transposed_values = flat_values.view(n_critics, batch_size, n_actions, -1)
    # (n_ensembles, batch_size, n, -1) -> (batch_size, n_ensembles, n, -1)
    values = transposed_values.transpose(0, 1)

    # get combination indices
    # (batch_size, n_ensembles, n, -1) -> (batch_size, n_ensembles, n)
    mean_values = values.mean(dim=3)
    # (batch_size, n_ensembles, n) -> (batch_size, n)
    max_values, max_indices = mean_values.max(dim=1)
    min_values, min_indices = mean_values.min(dim=1)
    mix_values = (1.0 - lam) * max_values + lam * min_values
    # (batch_size, n) -> (batch_size,)
    action_indices = mix_values.argmax(dim=1)

    # fuse maximum values and minimum values
    # (batch_size, n_ensembles, n, -1) -> (batch_size, n, n_ensembles, -1)
    values_T = values.transpose(1, 2)
    # (batch, n, n_ensembles, -1) -> (batch * n, n_ensembles, -1)
    flat_values = values_T.reshape(batch_size * n_actions, n_critics, -1)
    # (batch * n, n_ensembles, -1) -> (batch * n, -1)
    bn_indices = torch.arange(batch_size * n_actions)
    max_values = flat_values[bn_indices, max_indices.view(-1)]
    min_values = flat_values[bn_indices, min_indices.view(-1)]
    # (batch * n, -1) -> (batch, n, -1)
    max_values = max_values.view(batch_size, n_actions, -1)
    min_values = min_values.view(batch_size, n_actions, -1)
    mix_values = (1.0 - lam) * max_values + lam * min_values
    # (batch, n, -1) -> (batch, -1)
    result_values = mix_values[torch.arange(x.shape[0]), action_indices]

    return result_values, action_indices


def compute_max_with_n_actions(
    x: torch.Tensor,
    actions: torch.Tensor,
    forwarder: ContinuousEnsembleQFunctionForwarder,
    lam: float,
) -> torch.Tensor:
    return compute_max_with_n_actions_and_indices(x, actions, forwarder, lam)[0]
