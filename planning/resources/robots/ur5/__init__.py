from pathlib import Path

from grapeshot.assets.robot_asset import RobotAsset

UR5 = RobotAsset(Path(__file__).parent, "ur5.urdf", "ur5.srdf", "manipulator")
UR5_REAL = RobotAsset(Path(__file__).parent, "ur5_real.urdf", "ur5.srdf", "manipulator")
