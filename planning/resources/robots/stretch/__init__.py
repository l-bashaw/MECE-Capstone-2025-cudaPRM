from pathlib import Path

from grapeshot.assets.robot_asset import RobotAsset

STRETCH = RobotAsset(Path(__file__).parent, "stretch.urdf", "stretch.srdf", "stretch_arm")
