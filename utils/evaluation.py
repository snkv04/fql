from collections import defaultdict

import jax
import numpy as np
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def average_stats(stats):
    """Average per-episode statistics (lists of final-step info values) over episodes.

    If the episodes have both a success flag and a return, this also adds `episode.success_return`, the average return
    of the successful episodes only (NaN if no episode was successful).
    """
    results = {k: np.mean(v) for k, v in stats.items()}
    if 'success' in stats and 'episode.return' in stats:
        successes = np.array(stats['success'], dtype=bool)
        returns = np.array(stats['episode.return'])
        results['episode.success_return'] = returns[successes].mean() if successes.any() else np.nan
    return results


def evaluate(
    agent,
    env,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    use_planner=False,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature (unused with the planner).
        use_planner: Whether to select actions with the agent's planner (`plan_actions`), replanning at every step,
            instead of `sample_actions`. The agent must plan one action at a time (`plan_execute_steps` = 1).

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    if use_planner:
        if agent.config['plan_execute_steps'] != 1:
            raise ValueError(
                f'Evaluating with the planner replans at every step, so it requires plan_execute_steps=1, but got '
                f'plan_execute_steps={agent.config["plan_execute_steps"]}. Use eval_plan.py to evaluate '
                f'plan_execute_steps > 1.'
            )

        def plan_action(observations, seed, temperature=0):
            # plan_actions returns the best candidate's first plan_execute_steps (= 1) actions;
            # [0] drops the time axis.
            return agent.plan_actions(observations, seed)[0]

        action_fn = plan_action
    else:
        action_fn = agent.sample_actions
    actor_fn = supply_rng(action_fn, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()
        done = False
        step = 0
        render = []
        while not done:
            action = actor_fn(observations=observation, temperature=eval_temperature)
            action = np.array(action)
            action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    return average_stats(stats), trajs, renders
