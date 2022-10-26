#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 18:47:21 2022
  
@author: theouscidda
"""

# jax
import jax
import jax.numpy as jnp

# typing
from typing import Callable

# ott
from ott.geometry.costs import CostFn, Euclidean


def get_cost_matrix(x: jnp.ndarray,
                    y: jnp.ndarray,
                    cost_fn: CostFn):
    """Get pairwise cost matrix between vectors in X and Y for a given cost."""
    return cost_fn.all_pairs(x, y)

def get_rbf_kernel_matrix(x: jnp.ndarray,
                          y: jnp.ndarray, 
                          gamma: float):
    return jnp.exp(
        - gamma * get_cost_matrix(x, y, cost_fn=Euclidean())
    )

def get_scaled_epsilon(epsilon: float,
                       scale_cost: str,
                       cost_matrix: jnp.ndarray):
    """
    Get the scaled epsilon by ``scale_cost`` scaling on ``cost_matrix``.
    Example: if ``scale_cost`` = mean, it returns ``epsilon`` * mean(``cost_matrix``).
    """
    if scale_cost == None:
        return epsilon
    else:
        if scale_cost not in ['max', 'mean', 'median']:
            raise NotImplementedError(
                f"Scaling {scale_cost} not implemented yet."
            )
        get_scaling = getattr(jnp, scale_cost)
        return epsilon * get_scaling(cost_matrix)