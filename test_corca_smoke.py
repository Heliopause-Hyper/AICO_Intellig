import os
import sys

repo_root = os.path.abspath("/home/ycl/AICO-Intellig")
exec_state_path = os.path.join(repo_root, "folderA", "preciseFZ", "apply", "exec_state")
hdf5_file_path = os.path.join(repo_root, "folderA", "preciseFZ", "databank", "COMRES_last", "define_rod_01_001.hdf5")

sys.path.append(os.path.join(repo_root, "src"))
from corcasim_simulator import run_corcasim_simulation

result = run_corcasim_simulation(
    template_name="default",
    params={"Prk:": 50, "Pp:": 15.5, "Tin:": 292.1, "bore_ppm": 1000},
    exec_state_path=exec_state_path,
    hdf5_file_path=hdf5_file_path,
)
print(result)
