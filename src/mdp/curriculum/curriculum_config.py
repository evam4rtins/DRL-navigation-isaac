from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass

import mdp.curriculum.helper_functions as curriculum


@configclass
class CurriculumCfg:
    obstacle_weight = CurriculumTermCfg(
        func=curriculum.update_collision_weight,
        params={
            "weight_step_size": 0.05,
            "max_weight": 1.5,
            "episode_start": 20,  # start curriculum at 20 episodes
        },
    )

    collision_thresh = CurriculumTermCfg(
        func=curriculum.collision_termination_threshold,
        params={
            "episode_start": 100,  # start increasing threshold at 100 episodes
            "episode_end": 300,  # end increasing threshold at 300 episodes
        },
    )

    nominal_weight = CurriculumTermCfg(
        func=curriculum.nominal_policy_weight,
        params={
            "episode_start": 25,  # start decaying at 25 episodes
            "episode_end": 125,  # end decaying at 125 episodes
        },
    )

# @configclass
# class PathObstaclesCurriculumCfg(CurriculumCfg):
#     """Curriculum config that adds path obstacles after a certain number of episodes"""

#     add_obstacles = CurriculumTermCfg(
#         func=curriculum.add_path_obstacles,
#         params={
#             "episode_start": 100,  # add obstacles after n episodes
#             "episode_end": 600,  # full obstacle probability
#         }
#     )

@configclass
class CurriculumStepsCfg:
    """ same curriculum as CurriculumCfg scheduled in total simulation steps instead of episodes per robot """
    
    obstacle_weight = CurriculumTermCfg(
        func=curriculum.update_collision_weight,
        params={
            "weight_step_size": 0.05,
            "max_weight": 1.5,
            "step_start": 0.4e6,
            "update_every_steps": 0.256e6,
        }
    )

    collision_thresh = CurriculumTermCfg(
        func=curriculum.collision_termination_threshold,
        params={
            "step_start": 2.56e6,
            "step_end": 7.68e6,
        }
    )

    nominal_weight = CurriculumTermCfg(
        func=curriculum.nominal_policy_weight,
        params={
            "step_start": 0.64e6,
            "step_end": 3.2e6,
        }
    )

    total_steps = CurriculumTermCfg(func=curriculum.total_env_steps)