import os
import sys
import random

repo_root = os.path.abspath("/home/ycl/AICO-Intellig")
exec_state_path = os.path.join(repo_root, "folderA", "preciseFZ", "apply", "exec_state")
hdf5_file_path = os.path.join(repo_root, "folderA", "preciseFZ", "databank", "COMRES_last", "define_rod_01_001.hdf5")

sys.path.append(os.path.join(repo_root, "src"))
from corcasim_simulator import run_corcasim_simulation

for i in range(5):
    params = {
        "Prk:": int(random.uniform(0, 100)),
        "Pp:": round(random.uniform(10.0, 20.0), 3),
        "Tin:": round(random.uniform(280.0, 320.0), 3),
        "bore_ppm": int(random.uniform(0, 2000))
    }
    print(f"Testing {params}")
    result = run_corcasim_simulation(
        template_name="default",
        params=params,
        exec_state_path=exec_state_path,
        hdf5_file_path=hdf5_file_path,
    )
    print(result.get('success'))
