#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 18:47:21 2022

@author: theouscidda
"""

# jax
import jax.numpy as jnp
import jax.random as random
from jax.random import PRNGKeyArray
import jax

# typing
from typing import Any, Callable, Optional, Sequence, Dict, Union

# flax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state

# own code
from core.transport_maps import TransportMap
from core.data_fidelty_metrics import DataFidelty
from core.costs import get_rbf_kernel_matrix
import core.utils as utils

Sampler = Any

# custom typing
Batch = Dict[str, jnp.ndarray]


class EvaluationMetric:

    def __init__(self,
                 name: str,
                 use_gradient: Optional[bool] = None,
                 transport_map: Optional[TransportMap] = None,
                 input_dim: Optional[int] = None,
                 epsilon: float = 1e-2,
                 threshold_sinkhorn: float = 1e-3,
                 max_iter_sinkhorn: int = 100,
                 gammas: Sequence[float] = [2, 1, 0.5, 0.1, 0.01, 0.005]):
        self.input_dim = input_dim
        self.use_gradient = use_gradient
        self.transport_map = transport_map
        self.epsilon = epsilon
        self.threshold_sinkhorn = threshold_sinkhorn
        self.max_iter_sinkhorn = max_iter_sinkhorn
        self.gammas = jnp.array(gammas)
        self.is_regularization = False

        # define metric
        self.name = name
        self.evaluation_metric = self.get_metric()

    def get_metric(self):
        """Get the evaluation metric function."""
        
        # non trainable metrics
        if self.name in ["rbf_mmd", "l2_drug_signature"]:
            return getattr(self, self.name)

        # data fidelty
        elif self.name in ["sinkhorn_divergence", 
                           "wasserstein", "regularized_wasserstein", 
                           "mse"]:
            return DataFidelty(
                use_gradient=self.use_gradient, 
                name_metric=self.name, 
                epsilon=self.epsilon,
                threshold_sinkhorn=self.threshold_sinkhorn, 
                max_iter_sinkhorn=self.max_iter_sinkhorn
            )

        else:
            raise NotImplementedError(
                f"{self.name} evaluation metric not implemented yet."
            )

    def __call__(self,
                 params_phi: FrozenDict,
                 phi: nn.Module,
                 batch: Batch) -> Union[float, jnp.ndarray]:
        """
        Compute the ``name`` evaluation metric between the fitted and the target measures,
        evaluated on a ``batch``.
        """

        # compute metric
        return self.evaluation_metric(
            params_phi,
            phi,
            batch,
        )

    def rbf_mmd(self,
                 params_phi: FrozenDict,
                 phi: nn.Module,
                 batch: Batch):
        """Mean RBF Kernel MMD across serveral values of gamma."""

        # set transport map
        transport_map = self.transport_map or TransportMap(
            params_phi, use_gradient=self.use_gradient
        )

        # map the samples with the fitted map
        batch['mapped_source'] = transport_map.transport(
            params_phi=params_phi,
            phi=phi,
            data=batch['source']
        )

        def unique_mmd(x, y, gamma):
            """RBF Kernel MMD for a unique gamma"""
            xx = get_rbf_kernel_matrix(
                x, x, gamma
            )
            yy = get_rbf_kernel_matrix(
                y, y, gamma
            )
            xy = get_rbf_kernel_matrix(
                x, y, gamma
            )

            return xx.mean() + yy.mean() - 2 * xy.mean()

        return jnp.mean(
            jax.vmap(
                unique_mmd,
                in_axes=(None, None, 0)
            )(
                batch['mapped_source'], batch['target'], self.gammas
            )
        )

    def l2_drug_signature(self,
                 params_phi: FrozenDict,
                 phi: nn.Module,
                 batch: Batch):
        """l2 drug signature."""

        # set transport map
        transport_map = self.transport_map or TransportMap(
            params_phi, use_gradient=self.use_gradient
        )

        # map the samples with the fitted map
        batch['mapped_source'] = transport_map.transport(
            params_phi=params_phi,
            phi=phi,
            data=batch['source']
        )

        return jnp.linalg.norm(
            batch['target'].mean(0) - batch['mapped_source'].mean(0)
        )

