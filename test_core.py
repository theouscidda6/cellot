
# %%

import jax
import jax.random as random
from jax.random import PRNGKeyArray
import jax.numpy as jnp

from flax import linen as nn


from ott.geometry.costs import Euclidean

from core.evaluation_metrics import EvaluationMetric
from ott.core.icnn import ICNN

from core.transport_maps import TransportMap
from core.data_fidelty_metrics import DataFidelty

import load_cellot_data

import os

# %%

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
gpu_id = input(
    "gpu_id: "
)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

# %%

drug_name = "cisplatin"
data_type = "4i"
iterator = load_cellot_data.load_iterator(
    drug_name=drug_name,
    data_type=data_type
)

# %%

# sample a batch
batch = {}
batch['source'] = next(iterator.train.source)
batch['target'] = next(iterator.train.target)

# %%

# initialize an icnn
dim_hidden = [128, 64, 64]
input_dim = batch['source'].shape[1]
icnn = ICNN(
    dim_hidden=dim_hidden,
    gaussian_map=(batch['source'], batch['target']),
    dim_data=input_dim,
)
rng = jax.random.PRNGKey(0)
params_phi = icnn.init(rng, jnp.ones(input_dim))['params']
phi = icnn.apply

# %%

# define sinkhorn divergence
data_fidelty = EvaluationMetric(
    name="sinkhorn_divergence",
    epsilon=.1,
    max_iter_sinkhorn=2000
)

print(
    data_fidelty(
        params_phi=params_phi,
        phi=phi,
        batch=batch
    )
)

# %%

