"""Evaluate saved FQL-plan checkpoints of one or more training runs (e.g., seeds) with a planning setting.

Example:
    python eval_plan.py --run_dirs=exp/fql/Debug/<time>_job<id>_sd000,exp/fql/Debug/<time>_job<id>_sd001 --plan_horizon=3 --plan_samples=16

The training runs are evaluated one after another. The results of all of them are appended to one CSV,
<first run_dir>/<output> (one row per training run, checkpoint, and planning setting); checkpoints that are already in
the CSV with this training run and setting are skipped.

With --wandb, each training run is also logged to its own W&B run, with one point per checkpoint step. All of them are
in the group <wandb_group_prefix>_H<plan_horizon>_N<plan_samples>, so that the training runs (seeds) of a planning
setting are grouped together. The run ID is derived from the training run, the group, the planning setting, and the
evaluation seed and episodes, so separate jobs and reruns log to the same run. Each checkpoint also logs a video of
--video_episodes additional episodes (not included in the statistics).
"""

import contextlib
import csv
import glob
import hashlib
import json
import os
import re
import time
from collections import defaultdict

import jax
import numpy as np
import wandb
from absl import app, flags, logging
from tqdm import trange

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset
from utils.evaluation import add_to, flatten
from utils.flax_utils import restore_agent
from utils.log_utils import get_wandb_video

FLAGS = flags.FLAGS

flags.DEFINE_list('run_dirs', None, 'Training run directories (each containing flags.json and params_*.pkl).')
flags.DEFINE_list('steps', None, 'Checkpoint steps to evaluate (default: all saved checkpoints).')
flags.DEFINE_integer('plan_horizon', 0, 'Planning horizon H.')
flags.DEFINE_integer('plan_samples', 1, 'Number of candidates N.')
flags.DEFINE_enum('planning_score', 'last', ['last', 'best', 'all'], 'Planning score; only used for H > 0.')
flags.DEFINE_integer('plan_execute_steps', 1, 'Planned actions to execute before replanning (capped at H + 1).')
flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes per setting.')
flags.DEFINE_integer('video_episodes', 3, 'Number of video episodes per checkpoint (logged to W&B only).')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('seed', 0, 'Evaluation seed (the same episodes are used for every setting).')
flags.DEFINE_string(
    'output', 'eval_plan.csv', 'Output CSV for all training runs (relative paths are relative to the first run directory).'
)
flags.DEFINE_bool('wandb', True, 'Whether to also log results to W&B (one run per training run and planning setting).')
flags.DEFINE_string('wandb_project', 'fql', 'W&B project.')
flags.DEFINE_string(
    'wandb_group_prefix',
    None,
    'Prefix of the W&B group, which is <prefix>_H<plan_horizon>_N<plan_samples> (default: <run_group>-eval of the '
    'training runs).',
)

flags.mark_flag_as_required('run_dirs')

KEY_FIELDS = ['run_dir', 'step', 'plan_horizon', 'plan_samples', 'planning_score', 'plan_execute_steps']


def to_tuples(x):
    """Recursively convert lists (from JSON) to tuples, so that the agent config is hashable."""
    if isinstance(x, dict):
        return {k: to_tuples(v) for k, v in x.items()}
    if isinstance(x, list):
        return tuple(to_tuples(v) for v in x)
    return x


def evaluate_planner(agent, env, num_episodes, seed, num_video_episodes=0, video_frame_skip=3):
    """Evaluate `agent.plan_actions`, executing the returned planned actions before replanning.

    The `num_video_episodes` rendered episodes run after the evaluation episodes and are not included in the statistics
    or the planning time.

    Returns:
        A tuple containing a dictionary of the averaged episode statistics and the planning time per environment step,
        and the rendered videos.
    """
    rng = jax.random.PRNGKey(seed)

    # Compile the planner before timing.
    np.random.seed(seed)
    observation, _ = env.reset(seed=seed)
    jax.block_until_ready(agent.plan_actions(observation, rng))

    stats = defaultdict(list)
    renders = []
    plan_time = 0.0
    num_steps = 0
    for i in trange(num_episodes + num_video_episodes):
        should_render = i >= num_episodes
        # Seed both the environment's RNG and NumPy's global RNG (used by OGBench mazes for start/goal noise and
        # teleports), so that episode i is the same for every planning setting.
        np.random.seed(seed + i)
        observation, info = env.reset(seed=seed + i)
        planned_actions = []
        done = False
        step = 0
        render = []
        while not done:
            if not planned_actions:
                rng, key = jax.random.split(rng)
                start = time.time()
                planned_actions = list(np.array(agent.plan_actions(observation, key)))  # Blocks until computed.
                if not should_render:
                    plan_time += time.time() - start
            action = np.clip(planned_actions.pop(0), -1, 1)

            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                render.append(env.render().copy())
        if should_render:
            renders.append(np.array(render))
        else:
            add_to(stats, flatten(info))
            num_steps += step

    results = {k: np.mean(v) for k, v in stats.items()}
    results['plan_time_per_step'] = plan_time / num_steps
    return results, renders


def load_run_flags(run_dir):
    """Load the training flags of an fql_plan run and return them with the (hashable) agent config."""
    with open(os.path.join(run_dir, 'flags.json')) as f:
        train_flags = json.load(f)
    config = to_tuples(train_flags['agent'])
    assert config['agent_name'] == 'fql_plan', f'Expected an fql_plan run, got {config["agent_name"]}.'
    return train_flags, config


def read_rows(output):
    """Read the rows of an existing results CSV (empty if it does not exist)."""
    if not os.path.exists(output):
        return []
    with open(output) as f:
        return list(csv.DictReader(f))


def write_rows(output, rows):
    """Write all rows to the results CSV, with the union of their fields as the header."""
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    with open(output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_env(train_flags):
    """Make the evaluation environment and an example batch (for creating agents)."""
    _, eval_env, train_dataset, _ = make_env_and_datasets(train_flags['env_name'], frame_stack=train_flags['frame_stack'])
    example_batch = Dataset.create(**train_dataset).sample(1)
    return eval_env, example_batch


def make_agent(train_flags, config, example_batch):
    """Make an agent template (its parameters are replaced by each checkpoint)."""
    return agents[config['agent_name']].create(
        train_flags['seed'],
        example_batch['observations'],
        example_batch['actions'],
        config,
    )


def get_checkpoint_steps(run_dir, steps=None):
    """Return the checkpoint steps to evaluate (all saved checkpoints if `steps` is None)."""
    if steps is not None:
        steps = [int(s) for s in steps]
    else:
        checkpoints = glob.glob(os.path.join(run_dir, 'params_*.pkl'))
        steps = sorted(int(re.search(r'params_(\d+)\.pkl', c).group(1)) for c in checkpoints)
    assert steps, f'No checkpoints found in {run_dir}.'
    return steps


def get_plan_setting(horizon, num_samples, planning_score, plan_execute_steps):
    """Return the (plan_horizon, plan_samples, planning_score, plan_execute_steps) setting to evaluate."""
    if horizon == 0:
        # The planning score is unused without the dynamics model.
        return horizon, num_samples, 'none', 1
    if num_samples == 1:
        logging.warning(
            'With plan_samples=1, plan_horizon=%d executes the same first action as plan_horizon=0 (there is only one '
            'candidate to choose from).',
            horizon,
        )
    return horizon, num_samples, planning_score, min(plan_execute_steps, horizon + 1)


def with_plan_settings(agent, horizon, num_samples, planning_score, plan_execute_steps):
    """Return the agent with the given planning settings in its config."""
    return agent.replace(
        config=agent.config.copy(
            dict(
                plan_horizon=horizon,
                plan_samples=num_samples,
                planning_score=planning_score if horizon > 0 else 'last',
                plan_execute_steps=plan_execute_steps,
            )
        )
    )


def init_wandb(train_flags, run_dir, group, setting):
    """Create or resume the W&B run of a training run and planning setting (one run for all of its checkpoints)."""
    horizon, num_samples, planning_score, plan_execute_steps = setting
    run_name = os.path.basename(os.path.normpath(run_dir))
    id_source = '|'.join(map(str, [os.path.abspath(run_dir), group, *setting, FLAGS.seed, FLAGS.eval_episodes]))
    run_id = hashlib.md5(id_source.encode()).hexdigest()[:16]

    return wandb.init(
        project=FLAGS.wandb_project,
        group=group,
        name=f'{run_name}_H{horizon}_N{num_samples}_{planning_score}_x{plan_execute_steps}',
        id=run_id,
        resume='allow',
        config=dict(
            env_name=train_flags['env_name'],
            train_run_group=train_flags['run_group'],
            train_run_name=run_name,
            train_seed=train_flags['seed'],
            run_dir=os.path.abspath(run_dir),
            agent=train_flags['agent'],
            plan_horizon=horizon,
            plan_samples=num_samples,
            planning_score=planning_score,
            plan_execute_steps=plan_execute_steps,
            eval_seed=FLAGS.seed,
            eval_episodes=FLAGS.eval_episodes,
        ),
    )


def evaluate_run(run_dir, train_flags, config, eval_env, example_batch, group, setting, output, rows):
    """Evaluate the checkpoints of a training run that are not yet in the results CSV with the planning setting.

    New rows are appended to `rows`, which are written to `output` after each checkpoint.
    """
    horizon, num_samples, planning_score, plan_execute_steps = setting
    done_keys = {tuple(row[k] for k in KEY_FIELDS) for row in rows}

    steps = get_checkpoint_steps(run_dir, FLAGS.steps)
    steps = [step for step in steps if tuple(str(k) for k in (run_dir, step, *setting)) not in done_keys]
    if not steps:
        print(f'All checkpoints of {run_dir} are already evaluated with this setting.')
        return
    agent = make_agent(train_flags, config, example_batch)

    # Keep one W&B run open for all checkpoints (logged points are uploaded as they are logged).
    with init_wandb(train_flags, run_dir, group, setting) if FLAGS.wandb else contextlib.nullcontext() as run:
        if run is not None:
            logging.info(
                'W&B run: %s (project %s, group %s, name %s, %s).',
                run.url,
                run.project,
                run.group,
                run.name,
                'resumed' if run.resumed else 'new',
            )
        else:
            logging.info('W&B logging is disabled, so videos are not rendered.')

        for step in steps:
            print(
                f'{run_dir}, step {step}: H={horizon}, N={num_samples}, score={planning_score}, '
                f'execute={plan_execute_steps}'
            )

            step_agent = restore_agent(agent, run_dir, step)
            cell_agent = with_plan_settings(step_agent, *setting)
            # Videos are only logged to W&B, so skip rendering without it.
            num_video_episodes = FLAGS.video_episodes if FLAGS.wandb else 0
            results, renders = evaluate_planner(
                cell_agent, eval_env, FLAGS.eval_episodes, FLAGS.seed, num_video_episodes, FLAGS.video_frame_skip
            )

            rows.append(
                {**dict(zip(KEY_FIELDS, (run_dir, step, *setting))), 'eval_episodes': FLAGS.eval_episodes, **results}
            )
            write_rows(output, rows)
            if run is not None:
                metrics = {f'evaluation/{k}': v for k, v in results.items()}
                if renders:
                    metrics['video'] = get_wandb_video(renders=renders)
                run.log(metrics, step=step)
            logging.info(
                'Step %d: wrote a row to %s%s.',
                step,
                output,
                f' and logged {"metrics and a video" if renders else "metrics"} to W&B at step {step}' if run else '',
            )
            print({k: v for k, v in results.items() if 'success' in k or 'return' in k or k == 'plan_time_per_step'})


def main(_):
    run_dirs = [os.path.abspath(run_dir) for run_dir in FLAGS.run_dirs]
    run_flags = [load_run_flags(run_dir) for run_dir in run_dirs]

    # The training runs share the evaluation environment (and the default W&B group prefix).
    first_flags = run_flags[0][0]
    for run_dir, (train_flags, _) in zip(run_dirs, run_flags):
        for key in ['env_name', 'frame_stack'] + ([] if FLAGS.wandb_group_prefix else ['run_group']):
            assert train_flags[key] == first_flags[key], (
                f'{run_dir} has {key}={train_flags[key]}, but {run_dirs[0]} has {key}={first_flags[key]}.'
            )
    if FLAGS.wandb and FLAGS.video_episodes > 0:
        assert 'singletask' in first_flags['env_name'], 'Rendering is currently only supported for OGBench environments.'

    setting = get_plan_setting(FLAGS.plan_horizon, FLAGS.plan_samples, FLAGS.planning_score, FLAGS.plan_execute_steps)
    horizon, num_samples = setting[:2]
    group_prefix = FLAGS.wandb_group_prefix or f'{first_flags["run_group"]}-eval'
    group = f'{group_prefix}_H{horizon}_N{num_samples}'

    # The results of all training runs go to one CSV in the first run directory. Rows from before the CSV had a run_dir
    # column were written by the run directory that the CSV is in.
    output = os.path.join(run_dirs[0], FLAGS.output)
    rows = [{'run_dir': run_dirs[0], **row} if not row.get('run_dir') else row for row in read_rows(output)]
    logging.info(
        'Evaluating checkpoints from %d training runs:\n%s', len(run_dirs), '\n'.join(f'  {d}' for d in run_dirs)
    )
    logging.info('Results CSV: %s (%d existing rows).', output, len(rows))

    eval_env, example_batch = make_env(first_flags)
    for run_dir, (train_flags, config) in zip(run_dirs, run_flags):
        evaluate_run(run_dir, train_flags, config, eval_env, example_batch, group, setting, output, rows)


if __name__ == '__main__':
    app.run(main)
