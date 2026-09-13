import json
import sys
sys.path.append("/home/ycl/AICO-Intellig")
from src.problem_config import ProblemConfig
from scripts.generate_classification_data import _auto_build_instances

pc = ProblemConfig()
cfg = {
    "auto_max_total_instances": 10
}
instances = _auto_build_instances(pc.problem_types, cfg)
print(json.dumps(instances, indent=2))
