#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 18:47:21 2022

@author: theouscidda
"""

# jax
from pathlib import Path
import jax
import jax.random as random
import jax.numpy as jnp
from jax.random import PRNGKeyArray

# numpy
import numpy as np

# typing
from typing import Any, Callable, Dict, Optional, Sequence, Union

# ott
from ott.geometry import matrix_square_root
from scipy.linalg import sqrtm

# sklearn
from sklearn.decomposition import PCA

# omegaconf
from omegaconf import DictConfig

# os
import os

# contextlib
from contextlib import contextmanager

# custom typing
Batch = Dict[str, jnp.ndarray]
Distribution = Any
PyTree = Any


def get_gaussian_map(mean_source: jnp.ndarray,
                     cov_source: jnp.ndarray,
                     mean_target: jnp.ndarray,
                     cov_target: jnp.ndarray):
    """
    Compute the affine Monge map T : x -> linear @ x + bias between Gaussian distributions:
    - source = N(mean_source, cov_source),
    - target = N(mean_target, cov_target).
    """
    sqrtA, inv_sqrtA, _ = matrix_square_root.sqrtm(cov_source)
    mid = matrix_square_root.sqrtm_only(sqrtA @ cov_target @ sqrtA)
    linear = inv_sqrtA @ mid @ inv_sqrtA
    offset = mean_target - linear @ mean_source
    return linear, offset


def get_T_AB(A: jnp.ndarray,
             B: jnp.ndarray):
    """
    ! Deprecated, use get_gaussian_map instead. !
    Get the matrix defining the (affine) Monge map between the Gaussians,
    with respective covariance A and B.
    """
    sqrt_A = sqrtm(A)
    inv_sqrt_A = jnp.linalg.inv(sqrt_A)
    T_AB = inv_sqrt_A @ sqrtm(sqrt_A @ B @ sqrt_A) @ inv_sqrt_A
    return T_AB


def conditional_decorator(dec: Callable,
                          condition: bool):
    """
    Create a conditional decorator function.
    Example of usecase: jit a function i.f.f. a specific condition is true.
    """
    def decorator(func):
        if not condition:
            # return the function unchanged, not decorated
            return func
        # otherwise return the decorated function
        return dec(func)
    return decorator


def mean_generator(rng: PRNGKeyArray,
                   input_dim: int,
                   scale: float,
                   num: int = 1):
    """
    Generate random mean vectors for Distribution instancse,
    using multivariate Gaussian distribution with:
        - zero mean 
        - ``scale`` * jnp.eye(``input_dim``) covariance matrix
    """
    shape = None if num == 1 else (num, )
    return random.multivariate_normal(
        rng,
        mean=jnp.zeros(input_dim),
        cov=scale * jnp.eye(input_dim, input_dim),
        shape=shape
    )


def cov_generator(rng: PRNGKeyArray,
                  input_dim: int,
                  scale: float,
                  degree_of_freedom: Optional[int] = None,
                  num: int = 1):
    """
    Generate random covariance matrices for Distribution instances,
    using Wishart distribution with: 
        - ``degree_of_freedom`` degree of freedom,
        - ``scale_cov`` * jnp.eye(``input_dim``) scaling matrix
    """
    if degree_of_freedom is not None:
        assert degree_of_freedom >= input_dim, \
            "Provide a degree of freedom for covariance generation greater than or equal to the input dimension."
    else:
        degree_of_freedom = input_dim
    shape = (degree_of_freedom, ) if num == 1 else (num, degree_of_freedom)
    U = random.multivariate_normal(
        rng,
        mean=jnp.zeros(input_dim),
        cov=scale * jnp.eye(input_dim, input_dim),
        shape=shape
    )
    axis_transpose = (1, 0) if num == 1 else (0, 2, 1)
    return jnp.transpose(U, axis_transpose) @ U


def proportions_generator(rng: PRNGKeyArray,
                          num: int):
    """ 
    Generate random proprtion vector for mixture distribution, 
    using Dirichlet distribution with ones concentration vector.
    """
    return random.dirichlet(
        rng,
        alpha=jnp.ones(num)
    )


def standardize_potential(potential: Callable,
                          source: Optional[Distribution] = None,
                          mean: Optional[jnp.ndarray] = None,
                          mean_var: Optional[float] = None,
                          num_samples: int = 2**14,
                          seed: int = 0):
    """
    Outputs the linearly scaled potential, i.e. [a*potential+b*x].
    The gradient of this scaled output potential pushes sampler's distributions
    to zero-mean distribution with variance equal to the input_dim.
    """

    # generate samples for standardization if required
    if (mean is None) or (mean_var is None):
        assert source is not None, \
            "If not passing the standardization params, provide the source distribution to compute them."
        rng = random.PRNGKey(seed)
        X = source.generate_samples(rng, num_samples=num_samples)
        Y = jax.vmap(jax.grad(potential))(X)

        # check for nans
        assert not jnp.any(jnp.isnan(Y)), \
            "NaN(s) detected in mapped samples."

    # get estimated mean and estimated mean variance along coordinates if required
    if mean is None:
        mean = jnp.mean(Y, axis=0)
    if mean_var is None:
        mean_var = (Y - mean).var(axis=0).mean()

    # get standardized potential
    def standardized_potential(x): return (
        1. / jnp.sqrt(mean_var)) * (potential(x) - jnp.dot(mean, x))

    return standardized_potential


def process_batch(batch: Batch,
                  is_regularization: bool = False) -> Union[jnp.ndarray, Batch]:
    """Remove noise in samples from ``batch``to compute evaluation metric."""

    # if noise already removedz
    if batch['is_processed']:
        return batch if not is_regularization else batch['source']

    # remove noise otherwise
    for key in batch.keys():
        if key.startswith('noise'):
            samples_key = key.replace('noise_', '')
            batch[samples_key] -= batch[key]

    # set batch as processed
    batch['is_processed'] = True

    return batch if not is_regularization else batch['source']


def size_pytree(pytree: PyTree) -> int:
    """
    Get the sum of the sizes of all pytree's leaves. 
    When pytree is a FrozenDict containing neural network (either MLP, ICNN...) parameters, 
    it returns the number of parameters of the network.
    """
    return sum(
        jax.tree_util.tree_leaves(
            jax.tree_map(
                lambda x: x.size,
                pytree
            )
        )
    )


def num_params_from_layers_mlp(dim_layers: Sequence[int],
                               input_dim: int) -> int:
    """
    Get the number of parameters of a standard MLP with layers sizes in ``dim_layers``, 
    inferring the size of each layers parameters from layer sizes and the input dimension.
    Rmk: by default, the layers are the union of the hidden layers and the output layer (input layer excluded).
    """

    return sum(
        map(
            lambda x, y: (x + 1) * y,
            [input_dim] + dim_layers[:-1],
            dim_layers
        )
    )


def num_params_from_hidden_icnn(dim_hidden: Sequence[int],
                                input_dim: int) -> int:
    """
    Get the number of parameters of an ICNN with quadratic initialization layer, 
    and hidden layers sizes in ``dim_hidden``, 
    inferring the size of each layers parameters from layer sizes and the input dimension.
    Rmk: by default, the layers are the union of the hidden layers and the output layer (input layer excluded).
    """

    num_quadratic = input_dim * (1 + input_dim)
    num_residual = sum(
        map(
            lambda x: input_dim * x,
            dim_hidden + [1]
        )
    )
    num_feedforward = 2 * dim_hidden[0] + sum(
        map(
            lambda x, y: (x + 1) * y,
            dim_hidden,
            dim_hidden[1:] + [1]
        )
    )

    return num_quadratic +\
        num_residual +\
        num_feedforward


def layers_calibration(dim_hidden: Sequence[int],
                       input_dim: int) -> Sequence[int]:
    """
    Get calibrated layer sizes for a MLP neural vector field,
    defined from R^``input_dim`` -> R^``input_dim``, such that it's number of parameters matches
    approximately the number of parameters of an ICNN defined from R^``input_dim`` -> R, 
    with quadratic initialization layer and hidden layer sizes in ``dim_hidden``.
    """

    # get budget of parameters
    num_params_icnn = num_params_from_hidden_icnn(
        dim_hidden=dim_hidden, input_dim=input_dim
    )
    num_params_mlp_init = num_params_from_layers_mlp(
        dim_layers=dim_hidden+[input_dim], input_dim=input_dim
    )
    budget = num_params_icnn - num_params_mlp_init

    # add hidden units until it remains no budget
    full_layers = [input_dim] + dim_hidden + [input_dim]
    while budget >= 0:
        for l in range(1, len(full_layers)-1):

            # add one hidden units to layer l
            full_layers[l] += 1

            # remove the number of added parameters to the budget
            added_parameters = full_layers[l-1] + full_layers[l+1] + 1
            budget -= added_parameters

    # get calibrated layer sizes
    dim_layers = full_layers[1:]

    # check that the relative parameters excess of the mlp is not too large
    num_params_mlp = num_params_from_layers_mlp(
        dim_layers=dim_layers, input_dim=input_dim
    )
    relative_excess = ((num_params_mlp - num_params_icnn) /
                       num_params_icnn) * 100
    assert relative_excess <= 5., \
        "The relative parameters excess = {:.3e} % ; it exceeds 5%.".format(
            relative_excess)

    return dim_layers


def project(x: jnp.ndarray,
            num_components: int = 2):
    """
    Project ``x`` onto its ``num_components`` principal components to plot it.
    By default, do nothing if dimension of x <= num_components.
    """
    input_dim = x.shape[1]
    if input_dim > num_components:
        pca = PCA(
            n_components=num_components,
            svd_solver='full'
        )
        return pca.fit_transform(x)
    return x


def name_run_wandb_from_cfg(cfg: DictConfig) -> str:
    """Get the name of wandb run relative to configuration cfg."""

    def get_noise_msg(cfg: DictConfig) -> str:
        """Summarize the noise configuration relative to cfg."""
        assert not (
            (cfg.distributions.fraction_contamination_target > 0)
            and
            (cfg.distributions.nsr_target > 0)
        ), "Noise can be defined via either nsr or fraction contamination, not both"
        noise_msg = (
            f"; contamination_noise_level = {cfg.distributions.fraction_contamination_target}"
            if cfg.distributions.fraction_contamination_target > 0
            else f"; addtive_noise_level = {cfg.distributions.nsr_target}"
            if cfg.distributions.nsr_target > 0
            else f"; no noise"
        )
        return noise_msg

    def get_reg_msg(cfg: DictConfig) -> str:
        """Summarize regularization weighting hyperparameters relative to cfg."""
        return (
            f"l_cyc={cfg.model.regularizer.cyclical.lambd}" +
            f"; l_cons={cfg.model.regularizer.conservative.lambd}"
        )

    if cfg.sweep_type == 'noise':
        noise_msg = get_noise_msg(cfg)
        return (
            f"{cfg.model.name}" +
            noise_msg +
            f"; init_seed={cfg.model.network.initialization_seed}"
        )

    elif cfg.sweep_type == 'dimension':
        return (
            f"{cfg.model.name}" +
            f"; dim={cfg.input_dim}" +
            f"; init_seed={cfg.model.network.initialization_seed}"
        )

    elif cfg.sweep_type == 'heatmap':
        return (
            get_reg_msg(cfg)
        )

    elif cfg.sweep_type == 'single_cell':
        model_params = (
            get_reg_msg(cfg)
            if cfg.model.name == 'primal_regularized_mlp'
            else ""
        )
        return (
            f"{cfg.distributions.drug_name}" +
            f"; {cfg.model.name}" +
            "; " +
            model_params
        )

    elif cfg.sweep_type is None:
        return f"{cfg.model.name}"

    else:
        raise NotImplemented(
            f"Sweep type {cfg.sweep_type} not implemented"
        )


@contextmanager
def set_directory(path: Path):
    """Sets the cwd within the context."""

    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
