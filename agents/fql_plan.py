from typing import Any

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from agents.fql import FQLAgent
from agents.fql import get_config as get_fql_config
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import MLP


class FQLPlanAgent(FQLAgent):
    """FQL agent with a learned dynamics model for test-time planning.

    The BC flow, one-step policy, and critic are trained exactly as in FQL. A dynamics model f(s, a) -> s' is trained
    alongside them with an MSE loss on normalized states, and is only used by `plan_actions` at evaluation time.
    `sample_actions` is unchanged, so training, online data collection, and in-training evaluation use the plain FQL
    policy.
    """

    # Normalization statistics of the dynamics model (set from the dataset with `set_dynamics_stats`).
    obs_mean: Any = None
    obs_std: Any = None
    delta_mean: Any = None
    delta_std: Any = None

    def set_dynamics_stats(self, dataset):
        """Return the agent with normalization statistics computed from the dataset's transitions."""
        observations = np.asarray(dataset['observations'], dtype=np.float64)
        deltas = np.asarray(dataset['next_observations'], dtype=np.float64) - observations

        def stats(x):
            return jnp.asarray(x.mean(axis=0), dtype=jnp.float32), jnp.asarray(
                np.maximum(x.std(axis=0), 1e-6), dtype=jnp.float32
            )

        obs_mean, obs_std = stats(observations)
        delta_mean, delta_std = stats(deltas)
        return self.replace(obs_mean=obs_mean, obs_std=obs_std, delta_mean=delta_mean, delta_std=delta_std)

    def dynamics_inputs(self, observations, actions):
        """Return the dynamics model inputs (normalized observations and actions)."""
        return jnp.concatenate([(observations - self.obs_mean) / self.obs_std, actions], axis=-1)

    def predict_next_observations(self, observations, actions):
        """Predict the next observations with the dynamics model."""
        out = self.network.select('dynamics')(self.dynamics_inputs(observations, actions))
        if self.config['predict_delta']:
            return observations + out * self.delta_std + self.delta_mean
        return out * self.obs_std + self.obs_mean

    def dynamics_loss(self, batch, grad_params):
        """Compute the dynamics loss (MSE in the normalized target space)."""
        observations, next_observations = batch['observations'], batch['next_observations']
        if self.config['predict_delta']:
            target = (next_observations - observations - self.delta_mean) / self.delta_std
        else:
            target = (next_observations - self.obs_mean) / self.obs_std

        pred = self.network.select('dynamics')(self.dynamics_inputs(observations, batch['actions']), params=grad_params)
        dynamics_loss = jnp.mean((pred - target) ** 2)

        # MSE of the predicted next observations in the original (unnormalized) space, for logging.
        if self.config['predict_delta']:
            pred_next_observations = observations + pred * self.delta_std + self.delta_mean
        else:
            pred_next_observations = pred * self.obs_std + self.obs_mean
        next_ob_mse = jnp.mean((pred_next_observations - next_observations) ** 2)

        return dynamics_loss, {
            'dynamics_loss': dynamics_loss,
            'next_ob_mse': next_ob_mse,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, bc_batch=None, critic_batch=None):
        """Compute the total loss (FQL loss + dynamics loss).

        The dynamics model has its own parameters, so adding its loss does not change the FQL updates.
        """
        loss, info = super().total_loss(batch, grad_params, rng=rng, bc_batch=bc_batch, critic_batch=critic_batch)

        dynamics_loss, dynamics_info = self.dynamics_loss(batch, grad_params)
        for k, v in dynamics_info.items():
            info[f'dynamics/{k}'] = v

        return loss + dynamics_loss, info

    def aggregate_plan_qs(self, qs):
        """Aggregate Q values over the critic ensemble (axis 0) for scoring plans."""
        q_agg = self.config['plan_q_agg'] or self.config['q_agg']
        if q_agg == 'min':
            return qs.min(axis=0)
        return qs.mean(axis=0)

    @jax.jit
    def plan_actions(self, observations, seed):
        """Plan from a single (unbatched) observation and return the first `plan_execute_steps` planned actions.

        `plan_samples` candidates are proposed by the one-step policy with different noises. With `plan_horizon` H > 0,
        each candidate is rolled out for H steps with the dynamics model, taking the one-step policy's action at each
        predicted state, and scored with Q(s_t, a_t) at the predicted states t = 1..H according to `planning_score`
        ('last': t = H, 'best': max over t, 'all': sum over t). With H = 0, candidates are scored with Q(s, a) (best-of-N).
        With H = 0 and plan_samples = 1, this is the plain FQL policy.

        Returns:
            Actions of shape (plan_execute_steps, action_dim) from the best candidate.
        """
        horizon = self.config['plan_horizon']
        num_samples = self.config['plan_samples']
        rngs = jax.random.split(seed, horizon + 1)

        def policy(obs, rng):
            noises = jax.random.normal(rng, (num_samples, self.config['action_dim']))
            return jnp.clip(self.network.select('actor_onestep_flow')(obs, noises), -1, 1)

        def q_value(obs, actions):
            return self.aggregate_plan_qs(self.network.select('critic')(obs, actions=actions))

        obs = jnp.repeat(observations[None], num_samples, axis=0)
        actions = policy(obs, rngs[0])
        planned_actions = [actions]

        if horizon == 0:
            scores = q_value(obs, actions)
        else:
            qs = []
            for t in range(1, horizon + 1):
                obs = self.predict_next_observations(obs, actions)
                actions = policy(obs, rngs[t])
                planned_actions.append(actions)
                qs.append(q_value(obs, actions))
            qs = jnp.stack(qs)  # (H, N)

            if self.config['planning_score'] == 'last':
                scores = qs[-1]
            elif self.config['planning_score'] == 'best':
                scores = qs.max(axis=0)
            else:
                scores = qs.sum(axis=0)

        planned_actions = jnp.stack(planned_actions)  # (H + 1, N, action_dim)
        best = jnp.argmax(scores)
        return planned_actions[: self.config['plan_execute_steps'], best]

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        assert config['encoder'] is None, 'fql_plan only supports state-based observations.'
        assert config['planning_score'] in ['last', 'best', 'all'], f'Unsupported planning_score: {config["planning_score"]}'
        assert config['plan_q_agg'] in [None, 'min', 'mean'], f'Unsupported plan_q_agg: {config["plan_q_agg"]}'
        assert config['plan_samples'] >= 1, 'plan_samples must be positive.'
        assert 1 <= config['plan_execute_steps'] <= config['plan_horizon'] + 1, (
            'plan_execute_steps must be between 1 and plan_horizon + 1.'
        )

        agent = super().create(seed, ex_observations, ex_actions, config)

        # Add the dynamics model to the FQL networks. It is initialized with its own RNG, so the FQL networks and the
        # agent's RNG are the same as in FQL with the same seed.
        ob_dim = ex_observations.shape[-1]
        dynamics_def = MLP((*config['dynamics_hidden_dims'], ob_dim), activate_final=False, layer_norm=True)
        ex_dynamics_inputs = jnp.concatenate([ex_observations, ex_actions], axis=-1)
        dynamics_rng = jax.random.fold_in(jax.random.PRNGKey(seed), 1)
        dynamics_params = dynamics_def.init(dynamics_rng, ex_dynamics_inputs)['params']

        network_def = ModuleDict({**agent.network.model_def.modules, 'dynamics': dynamics_def})
        network_params = {**agent.network.params, 'modules_dynamics': dynamics_params}
        network_tx = optax.adam(learning_rate=config['lr'])
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # Identity normalization until `set_dynamics_stats` is called (or the statistics are restored from a checkpoint).
        zeros, ones = jnp.zeros(ob_dim, dtype=jnp.float32), jnp.ones(ob_dim, dtype=jnp.float32)
        return agent.replace(network=network, obs_mean=zeros, obs_std=ones, delta_mean=zeros, delta_std=ones)


def get_config():
    config = get_fql_config()
    config.agent_name = 'fql_plan'  # Agent name.
    config.dynamics_hidden_dims = (512, 512, 512, 512)  # Dynamics model hidden dimensions.
    config.predict_delta = True  # Whether the dynamics model predicts s' - s (True) or s' (False).
    config.plan_horizon = 0  # Number of dynamics model steps per plan (0: best-of-N without the dynamics model).
    config.plan_samples = 1  # Number of candidates proposed by the one-step policy.
    config.planning_score = 'last'  # How to score rollouts ('last', 'best', or 'all').
    config.plan_execute_steps = 1  # Number of planned actions to execute before replanning.
    config.plan_q_agg = ml_collections.config_dict.placeholder(str)  # Q aggregation for scoring plans ('min' or 'mean'; None: use q_agg).
    return config
