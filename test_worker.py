import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from simulator_interface import SimulatorFactory

sim = SimulatorFactory.create_simulator("corca_state")
bore_l, bore_r = 0, 2000
bore_steps = 10
found = False
last_result = {"success": False}
params_dict = {'Pp:': 15.0, 'Prk:': 50.0, 'Tin:': 300.0}
for _ in range(bore_steps):
    if bore_l > bore_r:
        break
    bore_mid = (bore_l + bore_r) // 2
    params_dict["bore_ppm"] = bore_mid
    params_dict["Prk:"] = int(params_dict.get("Prk:", 50))
    print(f"Testing bore_ppm={bore_mid}")
    result = sim.run_simulation("default", params_dict)
    print(result)
    last_result = result
    if not isinstance(result, dict) or not result.get("success"):
        break
    k = float(result.get("keff", 1.0))
    if k < 1.033:
        bore_r = bore_mid - 1
    elif k > 1.035:
        bore_l = bore_mid + 1
    else:
        found = True
        break
