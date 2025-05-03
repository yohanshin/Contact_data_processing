import os
username = os.getenv("USER")
import sys
sys.path.append(f"/home/{username}/Codes/projects/DeepGaitLab_beta")
from lib.models.detector.utils.transform import get_affine_transform