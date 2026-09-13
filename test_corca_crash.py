import sys, os, random
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from simulator_interface import SimulatorFactory

sim = SimulatorFactory.create_simulator("corca_state")
for i in range(100):
    params_dict = {
        "Pp:": random.uniform(10.0, 20.0),
        "Prk:": int(random.uniform(0, 100)),
        "Tin:": random.uniform(280.0, 320.0)
    }
    bore_l, bore_r = 0, 2000
    bore_steps = 10
    found = False
    last_result = {"success": False}
    for _ in range(bore_steps):
        if bore_l > bore_r: continue
        bore_mid = (bore_l + bore_r) // 2
        params_dict["bore_ppm"] = bore_mid
        params_dict["Prk:"] = int(params_dict.get("Prk:", 50))
        result = sim.run_simulation("default", params_dict)
        last_result = result
        if not isinstance(result, dict) or not result.get("success"):
            print(f"FAILED on bore_ppm={bore_mid}, params={params_dict}, result={result}")
            continue
        k = float(result.get("keff", 1.0))
        if k < 1.033: bore_r = bore_mid - 1
        elif k > 1.035: bore_l = bore_mid + 1
        else:
            found = True
            continue
    print(f"Iter {i} finished. found={found}, keff={last_result.get('keff')}, FQ={last_result.get('FQ')}")
    if not last_result.get("success"):
        print("Crash detected!")
        continue
