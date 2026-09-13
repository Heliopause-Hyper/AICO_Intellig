import os
import sys
sys.path.append("/home/ycl/AICO-Intellig")
from src.optimizer import DeepFreeFEMOptimizer
import json

config = {
  "problem_type": "corca_state",
  "template_name": "default",
  "simulator_type": "corca_state",
  "parameters": {"pos1: R, K 04 ": (0, 225)},
  "objectives": ["keff"],
  "primary_objective": "minimize_keff",
  "target_values": {"keff": 1.0},
  "budget": {"max_iterations": 2}
}
opt = DeepFreeFEMOptimizer()
res = opt.optimize(config, ["pos1: R, K 04 "], "Nelder-Mead", "SciPy")
print(res)
