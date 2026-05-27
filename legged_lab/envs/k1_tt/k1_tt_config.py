# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg
import legged_lab.mdp as mdp
from legged_lab.assets.booster import BOOSTER_K1_TT_CFG
from legged_lab.assets.table_tennis.table import TABLE_CFG
from legged_lab.assets.table_tennis.ball import BALL_CFG
from legged_lab.envs.base.tt_env_config import (  # noqa:F401
    TTAgentCfg,
    TTEnvCfg,
    BaseSceneCfg,
    DomainRandCfg,
    HeightScannerCfg,
    PhysxCfg,
    RewardCfg,
    CurriculumCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG


@configclass
class K1TableTennisRewardCfg(RewardCfg):
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    ang_vel_z_l2 = RewTerm(func=mdp.ang_vel_z_l2, weight=-0.02)
    energy = RewTerm(func=mdp.energy, weight=-1.5e-3)
    energy_ankle = RewTerm(
        func=mdp.energy,
        weight=-2e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Ankle_Pitch", ".*_Ankle_Roll"])},
    )
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.025)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-80.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*_foot_link*).*"), "threshold": 1.0},
    )
    penalty_robot_table_proximity_x = RewTerm(
        func=mdp.penalty_robot_table_proximity_x,
        weight=-20.0,
        params={"min_distance": 0.15, "std": 0.07},
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-2.5,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot_link*"), "threshold": 1.0},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.5)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1000.0)

    hit_unstable_support = RewTerm(
        func=mdp.hit_unstable_support,
        weight=-10,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot_link*")},
    )

    feet_orientation_L = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-4.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="left_foot_link*")},
    )
    feet_orientation_R = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-4.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="right_foot_link*")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot_link*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot_link*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    # K1's head body is named "Head_2" (T1 used "H2"), so the regex differs from T1.
    paddel_head_too_near = RewTerm(
        func=mdp.paddel_too_near_humanoid,
        weight=-100,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["Head_2"]), "threshold": 0.3},
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_foot_link*"]), "threshold": 0.2},
    )
    feet_really_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-10,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_foot_link*"]), "threshold": 0.15},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_foot_link*"])},
    )

    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Yaw", ".*_Hip_Roll"])},
    )

    # K1 shoulder pitch is named ALeft_Shoulder_Pitch; the broader Left_Shoulder_.* regex
    # only matches Left_Shoulder_Roll. We add an explicit shoulder pitch term below.
    joint_deviation_left_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["Left_Shoulder_.*", "Left_Elbow_.*", "ALeft_Shoulder_Pitch"]
            )
        },
    )

    joint_deviation_left_shoulder_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Left_Shoulder_Roll"])},
    )

    joint_deviation_right_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["Right_Shoulder_.*", "Right_Elbow_.*", "ARight_Shoulder_Pitch"]
            )
        },
    )

    # K1 has no Waist joint; the T1 joint_deviation_torso term is intentionally omitted.

    reward_contact = RewTerm(
        func=mdp.reward_contact,
        weight=150.0,
    )

    reward_future_dis_ee = RewTerm(
        func=mdp.reward_future_ee_target,
        weight=2.0,
        params={"std_ee": 0.5, "threshold": 0.15},
    )

    reward_future_dis_ro = RewTerm(
        func=mdp.reward_future_body_target,
        weight=5.0,
        params={"std_ro": 0.5, "threshold": 0.05},
    )

    reward_future_vel_base = RewTerm(
        func=mdp.reward_future_vel_target,
        weight=5.0,
        params={"vel_std": 1.2, "threshold": 0.1},
    )

    reward_future_landing_dis = RewTerm(
        func=mdp.reward_future_landing_dis,
        weight=60.0,
        params={"threshold": 3.0},
    )

    reward_future_pass_net = RewTerm(
        func=mdp.reward_future_pass_net,
        params={"std_h": 0.4, "z_target": 0.76 + 0.35},
        weight=100.0,
    )

    reward_table_success = RewTerm(
        func=mdp.reward_table_success,
        weight=100.0,
    )


# K1 actuated joint list for the table-tennis policy (20 joints; head excluded).
# Order matches T1's TT layout where possible: arms (L then R), then legs (L then R).
# K1 shoulder pitch joints use the `ALeft_*` / `ARight_*` URDF naming.
_K1_TT_POLICY_JOINTS = [
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]


@configclass
class K1TableTennisEnvCfg(TTEnvCfg):

    reward = K1TableTennisRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        # Match the T1 TT timing (kept here so the K1 cfg is fully self-describing).
        self.sim.dt = 0.002
        self.sim.decimation = 10  # 50 Hz
        self.scene.height_scanner.prim_body_name = "Trunk"
        self.scene.robot = BOOSTER_K1_TT_CFG
        self.scene.table = TABLE_CFG
        self.scene.ball = BALL_CFG
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = None
        self.robot.terminate_contacts_body_names = ["Trunk"]
        self.robot.feet_body_names = [".*_foot_link"]
        self.robot.num_actions = len(_K1_TT_POLICY_JOINTS)
        self.robot.num_joints = len(_K1_TT_POLICY_JOINTS)
        # K1 receives the T1 paddle subtree via tools/add_t1_paddle_to_k1.py, which
        # parents the marker/visuals/collisions under a `paddle_adapter` Xform so
        # K1's own articulation is untouched. The runtime resolves the actual
        # paddle hit point from this prim's USD transform at startup; the
        # numeric fallback below is only used if that lookup fails.
        self.robot.paddle_marker_subpath = "paddle_adapter/marker_ball"
        self.robot.paddle_local_offset = (0.0, -0.345, 0.0)
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ["Trunk"]
        # K1 has no Waist joint, so override the locomotion-joint reset event regex list.
        self.domain_rand.events.reset_locomotion_joints.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                ".*_Hip_.*",
                ".*_Knee_.*",
                ".*_Ankle_.*",
                "Left_Elbow_.*",
                "Left_Shoulder_.*",
                "ALeft_Shoulder_Pitch",
                "AAHead_yaw",
                "Head_pitch",
            ],
        )
        # Mirror the T1 manipulation-joint reset event but include K1's ARight_Shoulder_Pitch.
        self.domain_rand.events.reset_manipulation_joints.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=["Right_Elbow_.*", "Right_Shoulder_.*", "ARight_Shoulder_Pitch"],
        )
        self.observations.joint_names = list(_K1_TT_POLICY_JOINTS)
        self.actions.joint_names = list(_K1_TT_POLICY_JOINTS)


@configclass
class K1TT_EvalEnvCfg(K1TableTennisEnvCfg):
    """Eval variant: identical to K1TableTennisEnvCfg but with extended episode length."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.max_episode_length_s = 99999999999  # prevent frequent reset
        self.domain_rand.events.reset_base.params["pose_range"] = {
            "x": (-0.41, -0.4),
            "y": (0.3, 0.4),
            "yaw": (-0.1, 0.1),
        }
        self.domain_rand.events.reset_base.params["velocity_range"] = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (-0.02, 0.02),
            "roll": (-0.02, 0.02),
            "pitch": (-0.02, 0.02),
            "yaw": (-0.02, 0.02),
        }
        # serving range — matches paper evaluation protocol (T1 eval uses same values)
        self.ball.ball_speed_x_range = (-6.5, -5.2)
        self.ball.ball_speed_y_range = (-0.6, 0.2)
        self.ball.ball_speed_z_range = (1.5, 1.9)


@configclass
class K1TableTennisAgentCfg(TTAgentCfg):
    experiment_name: str = "k1_table_tennis"
    logger = "tensorboard"
    save_interval = 250
    max_iterations = 20000

    # Auxiliary predictor configuration used by OnPolicyPredictorRegressionRunner
    # Ignored by the standard OnPolicyRunner.
    predictor = {
        "history_len": 5,
        "traj_max_len": 128,
        "hidden_sizes": [64, 64],
        "lr": 0.5e-3,
        "epochs_per_update": 1,
        "batch_size": 1024,
        "train_until_iters": 20,
    }
