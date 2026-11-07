import torch

from navigation_env.NavigationEnv import NavEnv


## Curriculum manager calls this only when one or more envs are reset, so the internal
## episode_length_buf contains the lengths of the episodes just before reset.

def _total_env_steps(env: NavEnv) -> int:
    """ simulation steps collectd by all robots together since start of training run """
    return int(env.common_step_counter) * env.num_envs

def _schedule_coeff(current: float, start: float, end: float) -> float:
    if current < start:
        return 1.0
    if current > end: 
        return 0.0
    return 1.0 - (current - start)/(end - start)

def update_collision_weight(
    env: NavEnv, 
    env_ids: torch.Tensor, 
    weight_step_size: float, 
    max_weight: float, 
    episode_start: int = 0, 
    step_start: float | None = None,
    update_every_steps: float | None = None,
) -> float:
    
    if not hasattr(env, "_running_mean_var"):  # ensure attribute exists
        env._running_mean_var = 0.0
        env._last_update_step = 0

    env.path_manager.update_episode_durations(env_ids, env.episode_length_buf[env_ids])

    term_cfg = env.reward_manager.get_term_cfg("collision")

    if step_start is not None:
        current_step = _total_env_steps(env)
        interval = update_every_steps if update_every_steps is not None else 1000 * env.num_envs
        not_started = current_step < step_start
    else:
        current_step = env.common_step_counter
        interval = 1000
        not_started = env.episode_counter.mean().item() < episode_start

    # check curriculum conditions once every 1000 steps
    if current_step - env._last_update_step < interval or not_started:
        return term_cfg.weight
    env._last_update_step = current_step

    history = env.path_manager.duration_history
    if torch.any(history == 0.0):
        # not enough data yet to compute variance, skip update
        return term_cfg.weight

    duration_variance = history.var(dim=-1)
    average_variance = duration_variance.mean()

    if env._running_mean_var == 0.0:  # initialize running mean variance on first update
        env._running_mean_var = average_variance
    else:
        env._running_mean_var = 0.8 * env._running_mean_var + 0.2 * average_variance

    if average_variance < env._running_mean_var * 0.90:
        term_cfg.weight += weight_step_size

    # Ensure the weight doesn't exceed the maximum allowed value
    term_cfg.weight = min(term_cfg.weight, max_weight)

    return term_cfg.weight


def _progress_coeff(
        env: NavEnv, 
        episode_start: int, 
        episode_end: int,
        step_start: float | None = None,
        step_end: float | None = None,
        ) -> float:
    if step_start is not None and step_end is not None:
        return _schedule_coeff(_total_env_steps(env), step_start, step_end)
    current_episode = env.episode_counter.mean().item() * 0.75  # TODO: check correction factor
    return _schedule_coeff(current_episode, episode_start, episode_end)

def collision_termination_threshold(
        env: NavEnv, 
        env_ids: torch.Tensor, 
        episode_start: int = 50, 
        episode_end: int = 250, 
        step_start: float | None = None, 
        step_end: float | None = None,
        ) -> float:
    termination_coeff = _progress_coeff(env, episode_start, episode_end, step_start, step_end)
    env.collision_termination_thresh = max(termination_coeff, 0.01)
    return env.collision_termination_thresh


def nominal_policy_weight(
        env: NavEnv, 
        env_ids: torch.Tensor, 
        episode_start: int = 50, 
        episode_end: int = 250, 
        step_start: float | None = None,
        step_end: float | None = None,
        ) -> float:
    # linearly decay nominal policy weight from 1.0 to 0.0 over the ep interval
    env.nominal_weight = _progress_coeff(env, episode_start, episode_end, step_start, step_end)
    return env.nominal_weight

def total_env_steps(env: NavEnv, env_ids: torch.Tensor) -> float:
    return float(_total_env_steps(env))

def add_path_obstacles(
        env: NavEnv, 
        env_ids: torch.Tensor, 
        episode_start: int = 50, 
        episode_end: int = 250,
        step_start: float | None = None,
        step_end: float | None = None,
        ) -> None:
    env.obstacle_prob = 1 - _progress_coeff(env, episode_start, episode_end, step_start, step_end)
    return env.obstacle_prob
