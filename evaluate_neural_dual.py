# %%
# jax
import jax.numpy as jnp

# optax
import optax

# ott
from ott.core.icnn import ICNN
from ott.core.neuraldual import NeuralDualSolver

# wandb
import wandb

# hydra
import hydra

# os 
import os

# omegaconf
from omegaconf import DictConfig, OmegaConf

# pandas
import pandas as pd

# own code
from core.evaluation_metrics import EvaluationMetric
import load_cellot_data


@hydra.main(config_path="test_neural_dual", config_name="conf_test")
def evaluate(cfg: DictConfig):

    # hydra automatically change working directory 
    os.chdir("/home/theouscidda/cellot")
    
    # load data
    iterator = load_cellot_data.load_iterator(
        drug_name=cfg.drug.name, data_type=cfg.drug.type, where=cfg.drug.where
    )
    dataset = load_cellot_data.load_dataset(
        drug_name=cfg.drug.name, data_type=cfg.drug.type, where=cfg.drug.where
    )

    # get source and target training data for Gaussian initializations
    init_batch = {}
    init_batch["source"] = jnp.array(
        dataset.train.source.adata.to_df().values[: cfg.init_params.num_samples_init]
    )
    init_batch["target"] = jnp.array(
        dataset.train.target.adata.to_df().values[: cfg.init_params.num_samples_init]
    )
    input_dim = init_batch["source"].shape[1]

    # set ICNN architectures
    # clipping weights of ICNN f during training with ReLu
    # relaxing weights positivity constraint on ICNN g with weights regularization sum_ij(max(0, -w_ij^2))
    dim_hidden_icnn = [128, 128, 64, 64]
    pos_weights = False

    # grad f = backward transport map: target -> source
    neural_f = ICNN(
        dim_hidden=dim_hidden_icnn,
        gaussian_map=(init_batch["target"], init_batch["source"]),
        init_std=cfg.init_params.init_std,
        pos_weights=pos_weights,
        dim_data=input_dim,
    )

    # grad g = forward transport map: source -> target
    neural_g = ICNN(
        dim_hidden=dim_hidden_icnn,
        gaussian_map=(init_batch["source"], init_batch["target"]),
        init_std=cfg.init_params.init_std,
        pos_weights=pos_weights,
        dim_data=input_dim,
    )

    # set optimizers
    num_train_iters = 100_000
    optimizer_f = optax.adam(learning_rate=1e-4, b1=0.5, b2=0.9, eps=1e-8)
    optimizer_g = optax.adam(learning_rate=1e-4, b1=0.5, b2=0.9, eps=1e-8)

    # define solver
    neural_dual_solver = NeuralDualSolver(
        input_dim=input_dim,
        neural_f=neural_f,
        neural_g=neural_g,
        pos_weights=pos_weights,
        optimizer_f=optimizer_f,
        optimizer_g=optimizer_g,
        num_train_iters=num_train_iters,
        logging=True,
        log_freq=500,
        valid_freq=500,
        seed=cfg.init_params.init_seed,
    )

    # get testing set
    # - source: testing source given by train / test split
    # - target: training source + testing target given by train / test split
    # = 2000 ~ 2500 samples for source and target
    test_batch = {}
    test_batch["source"] = dataset.test.source.adata.to_df().values
    test_batch["target"] = pd.concat(
        (dataset.train.target.adata.to_df(), dataset.test.target.adata.to_df())
    ).values

    # set evaluation metrics
    eval_metrics = [
        EvaluationMetric(
            name="sinkhorn_divergence", epsilon=0.1, max_iter_sinkhorn=1_000
        ),
        EvaluationMetric(name="rbf_mmd"),
        EvaluationMetric(name="l2_drug_signature"),
    ]

    # define wandb config
    config_wandb = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    with wandb.init(config=config_wandb):

        # set wandb run name
        wandb.run.name = f"{cfg.drug.name}; seed={cfg.init_params.init_seed}"

        # compute initial eval metrics after gaussian initialization
        state_phi = (
            neural_dual_solver.state_g
        )  # state associated to potential whose gradient is the forward map : source -> target
        for metric in eval_metrics:
            metric_value = metric(
                params_phi=state_phi.params, phi=state_phi.apply_fn, batch=test_batch
            )
            wandb.log({'step': 0})
            wandb.log({metric.name: metric_value})

        # training
        _, logs = neural_dual_solver(
            trainloader_source=iterator.train.source,
            trainloader_target=iterator.train.target,
            validloader_source=iterator.test.source,
            validloader_target=iterator.test.target,
        )

        # compute initial eval metrics after training
        state_phi = (
            neural_dual_solver.state_g
        )  # state associated to potential whose gradient is the forward map : source -> target
        for metric in eval_metrics:
            metric_value = metric(
                params_phi=state_phi.params, phi=state_phi.apply_fn, batch=test_batch
            )
            wandb.log({'step': 1})
            wandb.log({metric.name: metric_value})

    return


if __name__ == "__main__":
    evaluate()
