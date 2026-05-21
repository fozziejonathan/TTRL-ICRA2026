# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR

BOOSTER_T1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/booster/t1/t1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.72),
        joint_pos={
            # Head
            "AAHead_yaw": 0.0,
            "Head_pitch": 0.0,
            # Arm
            ".*_Shoulder_Pitch": 0.2,
            "Left_Shoulder_Roll": -1.35,
            "Right_Shoulder_Roll": 1.35,
            ".*_Elbow_Pitch": 0.0,
            "Left_Elbow_Yaw": -0.5,
            "Right_Elbow_Yaw": 0.5,
            # Waist
            "Waist": 0.0,
            # Leg
            ".*_Hip_Pitch": -0.20,
            ".*_Hip_Roll": 0.0,
            ".*_Hip_Yaw": 0.0,
            ".*_Knee_Pitch": 0.42,
            ".*_Ankle_Pitch": -0.23,
            ".*_Ankle_Roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
                "Waist",
            ],
            effort_limit={
                ".*_Hip_Pitch": 45.0,
                ".*_Hip_Roll": 30.0,
                ".*_Hip_Yaw": 30.0,
                ".*_Knee_Pitch": 60.0,
                "Waist": 30.0,
            },
            velocity_limit={
                ".*_Hip_Pitch": 12.5,
                ".*_Hip_Roll": 10.9,
                ".*_Hip_Yaw": 10.9,
                ".*_Knee_Pitch": 11.7,
                "Waist": 10.88,
            },
            stiffness=200.0,
            damping=5.0,
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            effort_limit={".*_Ankle_Pitch": 24, ".*_Ankle_Roll": 15},
            velocity_limit={".*_Ankle_Pitch": 18.8, ".*_Ankle_Roll": 12.4},
            stiffness=50.0,
            damping=1.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            effort_limit=18.0,
            velocity_limit=18.8,
            stiffness=40.0,
            damping=10.0,
            armature=0.01,
        ),
    },
)
"""Configuration for the Booster T1 Humanoid robot."""




BOOSTER_T1_TT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/booster/T1_TT/T1_TT.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-1.6, 0.0, 0.72),
        joint_pos={
            # Head
            "AAHead_yaw": 0.0,
            "Head_pitch": 0.0,
            # Arm
            ".*_Shoulder_Pitch": 0.2,
            "Left_Shoulder_Roll": -1.35,
            "Right_Shoulder_Roll": 1.35,
            ".*_Elbow_Pitch": 0.0,
            "Left_Elbow_Yaw": -0.5,
            "Right_Elbow_Yaw": 0.5,
            # Waist
            "Waist": 0.0,
            # Leg
            ".*_Hip_Pitch": -0.20,
            ".*_Hip_Roll": 0.0,
            ".*_Hip_Yaw": 0.0,
            ".*_Knee_Pitch": 0.42,
            ".*_Ankle_Pitch": -0.23,
            ".*_Ankle_Roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
                "Waist",
            ],
            effort_limit={
                ".*_Hip_Pitch": 45.0,
                ".*_Hip_Roll": 30.0,
                ".*_Hip_Yaw": 30.0,
                ".*_Knee_Pitch": 60.0,
                "Waist": 30.0,
            },
            velocity_limit={
                ".*_Hip_Pitch": 12.5,
                ".*_Hip_Roll": 10.9,
                ".*_Hip_Yaw": 10.9,
                ".*_Knee_Pitch": 11.7,
                "Waist": 10.88,
            },
            stiffness=200.0, #default 200
            damping=5.0, #default 5
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            effort_limit={".*_Ankle_Pitch": 24, ".*_Ankle_Roll": 15},
            velocity_limit={".*_Ankle_Pitch": 18.8, ".*_Ankle_Roll": 12.4},
            stiffness=50.0,#default 50
            damping=1.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            effort_limit=18.0,
            velocity_limit=18.8,
            stiffness=40.0,
            damping=10.0,
            armature=0.01,
        ),
    },
)
"""Configuration for the Booster T1 Humanoid robot for Table Tennis."""

BOOSTER_T1_TT_P_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/booster/T1_TT/T1_TT.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-1.6, 0.0, 0.72),
        joint_pos={
            # Head
            "AAHead_yaw": 0.0,
            "Head_pitch": 0.0,
            # Arm
            ".*_Shoulder_Pitch": 0.2,
            "Left_Shoulder_Roll": -1.35,
            "Right_Shoulder_Roll": 0.0,
            "Left_Elbow_Pitch": 0.0,
            "Right_Elbow_Pitch": 0.4,
            "Left_Elbow_Yaw": -0.5,
            "Right_Elbow_Yaw": 0.2,
            # Waist
            "Waist": 0.0,
            # Leg
            ".*_Hip_Pitch": -0.20,
            ".*_Hip_Roll": 0.0,
            ".*_Hip_Yaw": 0.0,
            ".*_Knee_Pitch": 0.42,
            ".*_Ankle_Pitch": -0.23,
            ".*_Ankle_Roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
                "Waist",
            ],
            effort_limit={
                ".*_Hip_Pitch": 45.0,
                ".*_Hip_Roll": 30.0,
                ".*_Hip_Yaw": 30.0,
                ".*_Knee_Pitch": 60.0,
                "Waist": 30.0,
            },
            velocity_limit={
                ".*_Hip_Pitch": 12.5,
                ".*_Hip_Roll": 10.9,
                ".*_Hip_Yaw": 10.9,
                ".*_Knee_Pitch": 11.7,
                "Waist": 10.88,
            },
            stiffness=200.0,
            damping=5.0,
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            effort_limit={".*_Ankle_Pitch": 24, ".*_Ankle_Roll": 15},
            velocity_limit={".*_Ankle_Pitch": 18.8, ".*_Ankle_Roll": 12.4},
            stiffness=50.0,
            damping=1.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            effort_limit=18.0,
            velocity_limit=18.8,
            stiffness=40.0,
            damping=10.0,
            armature=0.01,
        ),
    },
)
"""Configuration for the Booster T1 Humanoid robot for Table Tennis, changed default joint positions"""


BOOSTER_K1_TT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/booster/K1_TT/K1_TT.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-1.6, 0.0, 0.72),
        joint_pos={
            "AAHead_yaw": 0.0,
            "Head_pitch": 0.0,
            ".*_Shoulder_Pitch": 0.2,
            "Left_Shoulder_Roll": -1.35,
            "Right_Shoulder_Roll": -0.1,
            "Left_Elbow_Pitch": 0.0,
            "Right_Elbow_Pitch": 0.2,
            "Left_Elbow_Yaw": -0.5,
            "Right_Elbow_Yaw": 0.5,
            ".*_Hip_Pitch": -0.20,
            ".*_Hip_Roll": 0.0,
            ".*_Hip_Yaw": 0.0,
            ".*_Knee_Pitch": 0.42,
            ".*_Ankle_Pitch": -0.23,
            ".*_Ankle_Roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
            ],
            effort_limit={
                ".*_Hip_Pitch": 30.0,
                ".*_Hip_Roll": 35.0,
                ".*_Hip_Yaw": 20.0,
                ".*_Knee_Pitch": 40.0,
            },
            velocity_limit={
                ".*_Hip_Pitch": 7.1,
                ".*_Hip_Roll": 12.9,
                ".*_Hip_Yaw": 18.1,
                ".*_Knee_Pitch": 12.5,
            },
            # K1 leg effort limits are ~67% of T1 (30-40 Nm vs 45-60 Nm).
            # Halving stiffness from T1's 200 keeps PD saturation onset at a
            # similar joint-error angle and reduces ankle tracking error, which
            # the PACE paper identifies as a primary source of end-effector error.
            stiffness=100.0,
            damping=3.0,
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            effort_limit={".*_Ankle_Pitch": 20.0, ".*_Ankle_Roll": 20.0},
            velocity_limit={".*_Ankle_Pitch": 18.1, ".*_Ankle_Roll": 18.1},
            stiffness=30.0,
            damping=2.0,
            armature=0.01,
        ),
        # K1 shoulder pitch joints are named ALeft_Shoulder_Pitch / ARight_Shoulder_Pitch.
        # ".*_Shoulder_Pitch" matches both A-prefixed names AND the roll/elbow variants by
        # accident is avoided here because only Shoulder_Pitch ends with that suffix.
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            effort_limit=14.0,
            velocity_limit=18.0,
            # With 14 Nm limit, stiffness=40 saturated at only 0.35 rad error,
            # giving no torque headroom during a swing. Reducing to 25 gives
            # ~0.56 rad before saturation. Damping cut from 10 to 3 so fast
            # swing motions are not actively braked by the PD controller.
            stiffness=25.0,
            damping=3.0,
            armature=0.01,
        ),
        # Head joints are not policy-controlled but we hold them at default with a stiff PD
        # so the head doesn't flop around. Drop this group if you'd rather leave them passive.
        "head": ImplicitActuatorCfg(
            joint_names_expr=["AAHead_yaw", "Head_pitch"],
            effort_limit=6.0,
            velocity_limit=18.0,
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
    },
)
"""Configuration for the Booster K1 (22-DoF) Humanoid robot for Table Tennis.

Joint set differs from T1: no Waist, shoulder pitch joints are prefixed `ALeft_*` /
`ARight_*` in the URDF/USD, and head joints (`AAHead_yaw`, `Head_pitch`) live in their
own actuator group because the K1 TT policy outputs only 20 joints (head excluded).

Effort/velocity limits taken from booster_assets/robots/K1/K1_22dof/K1_22dof.urdf.
Stiffness/damping copied from T1 as a starting point; tune per joint group after the
first training run.
"""


BOOSTER_T1_TT_P2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/booster/T1_TT/T1_TT.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-1.6, 0.0, 0.72),
        joint_pos={
            # Head
            # "AAHead_yaw": 0.0,
            # "Head_pitch": 0.0,
            # Arm
            "Left_Shoulder_Pitch": 0.2,
            "Right_Shoulder_Pitch": 0.0,
            "Left_Shoulder_Roll": -1.35,
            "Right_Shoulder_Roll": -0.1,
            "Left_Elbow_Pitch": 0.0,
            "Right_Elbow_Pitch": 0.2,
            "Left_Elbow_Yaw": -0.5,
            "Right_Elbow_Yaw": 0.5,
            # Waist
            "Waist": 0.0,
            # Leg
            ".*_Hip_Pitch": -0.20,
            ".*_Hip_Roll": 0.0,
            ".*_Hip_Yaw": 0.0,
            ".*_Knee_Pitch": 0.42,
            ".*_Ankle_Pitch": -0.23,
            ".*_Ankle_Roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
                "Waist",
            ],
            effort_limit={
                ".*_Hip_Pitch": 45.0,
                ".*_Hip_Roll": 30.0,
                ".*_Hip_Yaw": 30.0,
                ".*_Knee_Pitch": 60.0,
                "Waist": 30.0,
            },
            velocity_limit={
                ".*_Hip_Pitch": 12.5,
                ".*_Hip_Roll": 10.9,
                ".*_Hip_Yaw": 10.9,
                ".*_Knee_Pitch": 11.7,
                "Waist": 10.88,
            },
            stiffness=100.0, #default 200
            damping=3.0, #default 5
            armature=0.01,
            min_delay=0,
            max_delay=3,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            effort_limit={".*_Ankle_Pitch": 20, ".*_Ankle_Roll": 15},
            velocity_limit={".*_Ankle_Pitch": 18.8, ".*_Ankle_Roll": 12.4},
            stiffness=30.0, #default 50
            damping=3.0,
            armature=0.01,
            min_delay=0,
            max_delay=3,
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            effort_limit=18.0,
            velocity_limit=18.84,
            stiffness=30.0,#default 20
            damping=3,#default 0.5
            armature=0.01,
            min_delay=0,
            max_delay=3,
        ),
    },
)
"""Configuration for the Booster T1 Humanoid robot for Table Tennis, changed default joint positions"""
