import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from absl import logging

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


class FQLAgent(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the IQL value loss (only used with td_target='value')."""
        qs = self.network.select('target_critic')(batch['observations'], actions=batch['actions'])
        q = self.aggregate_qs(qs)
        # A frozen value network is evaluated with its current params (no gradient) to keep the diagnostics.
        value_params = None if self.config['freeze_critic'] else grad_params
        v = self.network.select('value')(batch['observations'], params=value_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def aggregate_qs(self, qs):
        """Aggregate Q values over the critic ensemble (axis 0) according to `q_agg`."""
        if self.config['q_agg'] == 'min':
            return qs.min(axis=0)
        return qs.mean(axis=0)

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss.

        The bootstrap value in the TD target depends on `td_target`:
        - 'onestep': target Q of the one-step policy's action.
        - 'flow_max': max over `critic_num_samples` BC flow actions of the (ensemble-aggregated) target Q.
        - 'value': IQL value network.
        """
        info = {}
        if self.config['td_target'] == 'onestep':
            rng, sample_rng = jax.random.split(rng)
            next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
            next_actions = jnp.clip(next_actions, -1, 1)

            next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
            next_q = self.aggregate_qs(next_qs)
        elif self.config['td_target'] == 'flow_max':
            rng, noise_rng = jax.random.split(rng)
            batch_size = batch['next_observations'].shape[0]
            num_samples = self.config['critic_num_samples']

            # Sample `num_samples` actions per next observation from the BC flow. Observations are encoded once before
            # being repeated, so that the encoder is not run `num_samples` times on the same observation.
            flow_observations = batch['next_observations']
            if self.config['encoder'] is not None:
                flow_observations = self.network.select('actor_bc_flow_encoder')(flow_observations)
            flow_observations = jnp.repeat(flow_observations, num_samples, axis=0)
            noises = jax.random.normal(noise_rng, (batch_size * num_samples, self.config['action_dim']))
            next_actions = self.integrate_flow(flow_observations, noises)

            # Aggregate over the ensemble for each candidate, then take the max over candidates (as in EMaQ).
            n_next_observations = jnp.repeat(batch['next_observations'], num_samples, axis=0)
            next_qs = self.network.select('target_critic')(n_next_observations, actions=next_actions)
            next_qs = self.aggregate_qs(next_qs).reshape(batch_size, num_samples)
            next_q = next_qs.max(axis=-1)
            info['next_q_max_minus_mean'] = (next_q - next_qs.mean(axis=-1)).mean()
        else:
            next_q = self.network.select('value')(batch['next_observations'])

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        # A frozen critic is evaluated with its current params (no gradient) to keep the diagnostics.
        critic_params = None if self.config['freeze_critic'] else grad_params
        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=critic_params)
        if self.config['td_target'] == 'value':
            # Sum (rather than average) over the ensemble, as in IQL.
            critic_loss = jnp.square(q - target_q).sum(axis=0).mean()
        else:
            critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'target_q_mean': target_q.mean(),
            **info,
        }

    def actor_loss(self, batch, grad_params, rng, bc_batch=None):
        """Compute the FQL actor loss.

        The BC flow is trained on `bc_batch` (defaults to `batch`), and the one-step policy on `batch`.
        """
        bc_batch = batch if bc_batch is None else bc_batch
        batch_size, action_dim = batch['actions'].shape
        bc_batch_size = bc_batch['actions'].shape[0]
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (bc_batch_size, action_dim))
        x_1 = bc_batch['actions']
        t = jax.random.uniform(t_rng, (bc_batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        # Frozen networks are evaluated with their current params (no gradient) to keep the diagnostics.
        bc_params = None if self.config['freeze_bc'] else grad_params
        pred = self.network.select('actor_bc_flow')(bc_batch['observations'], x_t, t, params=bc_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # Distillation loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
        actor_params = None if self.config['freeze_actor'] else grad_params
        actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=actor_params)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Q loss.
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss
        if not self.config['use_q_loss']:
            q_loss = 0.0

        # Total loss.
        logging.info(f"alpha: {self.config['alpha']}")
        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        # Additional metrics for logging.
        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, bc_batch=None, critic_batch=None):
        """Compute the total loss.

        The one-step policy is trained on `batch`. The BC flow and critic are trained on `bc_batch` and
        `critic_batch`, respectively, which default to `batch`.
        """
        info = {}
        rng = rng if rng is not None else self.rng
        critic_batch = batch if critic_batch is None else critic_batch

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        if self.config['td_target'] == 'value':
            # The value network is trained on the same batch as the critic.
            value_loss, value_info = self.value_loss(critic_batch, grad_params)
            for k, v in value_info.items():
                info[f'value/{k}'] = v
        else:
            value_loss = 0.0

        critic_loss, critic_info = self.critic_loss(critic_batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng, bc_batch=bc_batch)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch, bc_batch=None, critic_batch=None):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng, bc_batch=bc_batch, critic_batch=critic_batch)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        # Restore the params of frozen modules, since Adam's momentum would otherwise keep updating them.
        frozen_modules = []
        if self.config['freeze_bc']:
            frozen_modules += ['actor_bc_flow', 'actor_bc_flow_encoder']
        if self.config['freeze_actor']:
            frozen_modules += ['actor_onestep_flow']
        if self.config['freeze_critic']:
            frozen_modules += ['critic', 'value']
        for module_name in frozen_modules:
            if f'modules_{module_name}' in self.network.params:
                new_network.params[f'modules_{module_name}'] = self.network.params[f'modules_{module_name}']

        if not self.config['freeze_critic']:
            self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the one-step policy."""
        action_seed, noise_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        actions = self.network.select('actor_onestep_flow')(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        return self.integrate_flow(observations, noises)

    def integrate_flow(self, observations, noises):
        """Integrate the BC flow from `noises` with the Euler method, given already encoded observations."""
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

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
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        assert config['td_target'] in ['onestep', 'flow_max', 'value'], f'Unsupported TD target: {config["td_target"]}'
        if config['q_agg'] is None:
            # Default to 'min' for 'flow_max' (as in EMaQ) and 'value' (as in IQL).
            config['q_agg'] = 'mean' if config['td_target'] == 'onestep' else 'min'
        assert config['q_agg'] in ['min', 'mean'], f'Unsupported q_agg: {config["q_agg"]}'
        use_value = config['td_target'] == 'value'

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            if use_value:
                encoders['value'] = encoder_module()
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions)),
        )
        if use_value:
            value_def = Value(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                num_ensembles=1,
                encoder=encoders.get('value'),
            )
            network_info['value'] = (value_def, (ex_observations,))
        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fql',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg=ml_collections.config_dict.placeholder(str),  # Aggregation method for target Q values (None: 'mean' for 'onestep', 'min' for 'flow_max' and 'value').
            td_target='onestep',  # Bootstrap value in the TD target ('onestep': one-step policy's action, 'flow_max': best of `critic_num_samples` BC flow actions, 'value': IQL value network).
            critic_num_samples=32,  # Number of BC flow actions per next observation for td_target='flow_max'.
            expectile=0.9,  # IQL expectile for td_target='value'.
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            online_alpha=ml_collections.config_dict.placeholder(float),  # BC coefficient during online fine-tuning (None: use alpha).
            use_q_loss=True,  # Whether to include the Q loss in the actor loss.
            online_use_q_loss=ml_collections.config_dict.placeholder(bool),  # Q loss flag during online fine-tuning (None: use use_q_loss).
            flow_steps=10,  # Number of flow steps.
            freeze_bc_online=False,  # Whether to freeze the BC flow during online fine-tuning.
            freeze_actor_online=False,  # Whether to freeze the one-step policy during online fine-tuning.
            freeze_critic_online=False,  # Whether to freeze the critic (and target critic) during online fine-tuning.
            freeze_bc=False,  # Whether the BC flow is frozen (set automatically from freeze_bc_online).
            freeze_actor=False,  # Whether the one-step policy is frozen (set automatically from freeze_actor_online).
            freeze_critic=False,  # Whether the critic is frozen (set automatically from freeze_critic_online).
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
