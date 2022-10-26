#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 18:47:21 2022

@author: theouscidda
"""

# jax
import jax
import jax.numpy as jnp

# flax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict

# typing
from typing import Optional


class TransportMap:
    """
    Neural Transport Map.
    May be the gradient of a neural potential or a neural vector field.

    Attributes:
      params_phi: neural transport map parameters 
      phi: neural transport map apply function 
      use_gradient: whether the map is the gradient of a neural potential or a neural vector field
      input_dim: dimensionality of the data
    """

    def __init__(self,
                 params_phi: FrozenDict,
                 use_gradient: Optional[bool] = None,
                 input_dim: Optional[bool] = None):
        self.input_dim = input_dim

        # set if map is a neural potential's gradient or a neural vector field
        self.set_use_gradient(params_phi, use_gradient)

    def set_use_gradient(self,
                         params_phi: FrozenDict,
                         use_gradient: bool):
        """
        Infer if the map is a neural potential's gradient or a neural vector field,
        if use_gradient attribute is not manually passed to the constructor.
        """
        if use_gradient is not None:
            self.use_gradient = use_gradient
            return

        # infer input dimensionality from neural parameters
        # ToDo: solve conflicts of notations 'W_x' and 'w_x
        if self.input_dim == None:
            if any(x in params_phi.keys() for x in ['w_xs_1', 'W_xs_1']):
                try:
                    try:
                        self.input_dim = params_phi['W_xs_1']['kernel'].shape[0]
                    except:
                        self.input_dim = params_phi['W_xs_1']['linear_kernel'].shape[0]
                except:
                    self.input_dim = params_phi['w_xs_1']['kernel'].shape[0]

            else:
                self.input_dim = params_phi['w_zs_0']['kernel'].shape[0]

        # set use_gradient attribute
        self.use_gradient = not (
            jax.tree_util.tree_leaves(
                params_phi
            )[-1].shape[-1] == self.input_dim
        )

    def transport(self,
                  params_phi: FrozenDict,
                  phi: nn.module,
                  data: jnp.ndarray) -> jnp.ndarray:
        """Transport source data samples with fitted map."""

        # set transport map
        if not self.use_gradient:
            return phi(
                {'params': params_phi},
                data
            )
        else:
            return jax.vmap(
                jax.grad(
                    lambda x: phi(
                        {'params': params_phi},
                        x
                    )
                )
            )(
                data
            )

    def potential(self,
                  params_phi: FrozenDict,
                  phi: nn.module,
                  data: jnp.ndarray) -> jnp.ndarray:
        """Apply fitted potential to source data samples."""

        assert self.use_gradient == True, \
            "Potential not available: the map is a neural vector field."

        return phi(
            {'params': params_phi},
            data
        )
