import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from simulator_interface import SimulatorFactory

print("Testing CORCA Evol (Burnup)...")
evol_sim = SimulatorFactory.create_simulator("corca_evol")
result = evol_sim.run_simulation("burnup", {
    "sequence_count": 3,
    "burnup_steps_1": 50, "rod_1": 225,
    "burnup_steps_2": 20, "rod_2": -1,
    "burnup_steps_3": 10, "rod_3": 150
})
print(result)
