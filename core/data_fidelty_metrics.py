#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 18:47:21 2022

@author: theouscidda
"""

# jax
import jax.numpy as jnp
import jax


# ott
from ott.geometry import geometry
from ott.geometry.costs import CostFn, Euclidean
from ott.tools import sinkhorn_divergence
from ott.core import sinkhorn

# typing
from typing import Callable, Optional, Dict

# flax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict

# own code
from core.transport_maps import TransportMap
from core.costs import get_cost_matrix, get_scaled_epsilon

# custom typing
Batch = Dict[str, jnp.ndarray]


class DataFidelty:
    """
    ! TODO: doc !
    Attributes:
      name_metric: name of the data fidely metric.
      transport_map: TransportMap class instance defining transport map,
                     and underlying potential if the map is a neural potential's gradient.
      use_gradient: define wether the map is a neural potential's gradient or a neural vector field.
      cost_fn: the ground cost function on the feature space defining the data fidely metric.
      scale_cost: scaling to scale  cost matrix in Sinkhorn algorithm.
      epsilon: entropic regularization stregth.
      threshold_sinkhorn: threshold to declare convergence in Sinkhorn algorithm.
      max_iter_sinkhorn: maximal number of iterations in Sinkhorn algorithm.
    """

    def __init__(self,
                 name_metric: str,
                 transport_map: Optional[TransportMap] = None,
                 use_gradient: Optional[bool] = None,
                 cost_fn: CostFn = Euclidean(),
                 scale_cost: Optional[str] = None,
                 epsilon: float = 1e-1,
                 threshold_sinkhorn: float = 1e-2,
                 max_iter_sinkhorn: int = 1000):
        self.transport_map = transport_map
        self.use_gradient = use_gradient
        self.cost_fn = cost_fn
        self.scale_cost = scale_cost
        self.epsilon = epsilon
        self.threshold_sinkhorn = threshold_sinkhorn
        self.max_iter_sinkhorn = max_iter_sinkhorn

        # attribute to map the sample with neural potential when evaluating metric
        self.use_potential = (name_metric == 'mse_potential')

        # define data fidelty metric
        self.data_fidelty = self.get_data_fidelty(name_metric)

    def get_data_fidelty(self,
                         name_metric: str) -> Callable:
        """Get data fidelty function."""
        self.name_metric = name_metric
        if name_metric in ['mse_map', 'mse_potential']:
            return self.mse
        else:
            try:
                return getattr(self, name_metric)
            except:
                raise NotImplementedError(
                    f"{self.name_metric} data fidelty metric not implemented yet."
                )

    def __call__(self,
                 params_phi: FrozenDict,
                 phi: nn.Module,
                 batch: Batch,
                 *args) -> float:
        """
        Evaluate the metric ``name_metric`` between the fitted and the target measures,
        evaluated on a ``batch``.
        The fitted measure is the push-forward by the neural map - defined by 
        parameters ``params_phi`` and apply function ``phi`` - of the source measure.
        """

        # set transport map
        transport_map = self.transport_map or TransportMap(
            params_phi, use_gradient=self.use_gradient
        )

        # get predicted samples
        apply = getattr(
            transport_map, "potential"
        ) if self.use_potential else getattr(
            transport_map, "transport"
        )
        batch['mapped_source'] = apply(
            params_phi=params_phi,
            phi=phi,
            data=batch['source']
        )

        # compute data fidelty metric
        return self.data_fidelty(batch)

    def wasserstein(self,
                    batch: Batch) -> float:
        """Wasserstein distance."""

        # compute cost matrix
        cost_xy = get_cost_matrix(
            batch['mapped_source'], batch['target'], cost_fn=self.cost_fn
        )

        # freeze transport plan for differenciation
        matching = hungarian.hungarian_single(cost_xy)
        P = hungarian.get_permutation_matrix(matching)
        P = jax.lax.stop_gradient(P)
        n = len(batch['target'])

        return (1 / n) * jnp.sum(
            P * cost_xy 
        )
        

    def regularized_wasserstein(self,
                                batch: Batch) -> float:
        """Regularized Wasserstein distance."""

        # compute cost matrix
        cost_xy = get_cost_matrix(
            batch['mapped_source'], batch['target'], cost_fn=self.cost_fn
        )

        # define geometry
        scaled_epsilon = get_scaled_epsilon(
            epsilon=self.epsilon,
            scale_cost=self.scale_cost,
            cost_matrix=cost_xy
        )
        geom = geometry.Geometry(
            cost_xy, epsilon=scaled_epsilon
        )

        # compute the regularized Wassertein distance between the fitted measure and the target measure
        out = sinkhorn.sinkhorn(
            geom,
            threshold=self.threshold_sinkhorn,
            max_iterations=self.max_iter_sinkhorn,
            use_danskin=True
        )
        return out.reg_ot_cost

    def sinkhorn_divergence(self,
                            batch: Batch) -> float:
        """Sinkhorn divergence."""

        # compute cost matrices
        cost_xy = get_cost_matrix(
            batch['mapped_source'], batch['target'], cost_fn=self.cost_fn
        )
        cost_xx = get_cost_matrix(
            batch['mapped_source'], batch['mapped_source'], cost_fn=self.cost_fn
        )
        cost_yy = get_cost_matrix(
            batch['target'], batch['target'], cost_fn=self.cost_fn
        )

        # define scaled epsilon for geometry
        scaled_epsilon = get_scaled_epsilon(
            epsilon=self.epsilon,
            scale_cost=self.scale_cost,
            cost_matrix=cost_xy
        )

        # compute the Sinkhorn divergence between the fitted measure and the target measure
        out = sinkhorn_divergence.sinkhorn_divergence(
            geometry.Geometry,
            cost_xy, cost_xx, cost_yy,
            epsilon=scaled_epsilon,
            sinkhorn_kwargs={'threshold': self.threshold_sinkhorn,
                             'max_iterations': self.max_iter_sinkhorn,
                             'use_danskin': True}
        )
        return out.divergence

    def energy_distance(self,
                        batch: Batch) -> float:
        """Energy distance"""

        # compute cost matrices
        cost_xy = get_cost_matrix(
            batch['mapped_source'], batch['target'], cost_fn=self.cost_fn
        )
        cost_xx = get_cost_matrix(
            batch['mapped_source'], batch['mapped_source'], cost_fn=self.cost_fn
        )
        cost_yy = get_cost_matrix(
            batch['target'], batch['target'], cost_fn=self.cost_fn
        )

        # compute energy distance
        n = len(batch['source'])
        return (1 / n**2) * jnp.sum(
            cost_xy - (1/2) * (cost_xx + cost_yy)
        )

    def mse(self,
            batch: Batch) -> float:
        """
        Mean Squared Error.
        Rmk: can be computed either on potentials or maps.
        """
        return jnp.mean(
            (batch['mapped_source'] - batch['target'])**2
        )
