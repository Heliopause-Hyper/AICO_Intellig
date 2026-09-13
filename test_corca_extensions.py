import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from simulator_interface import SimulatorFactory

print("Testing CORCA Evol (Burnup)...")
evol_sim = SimulatorFactory.create_simulator("corca_evol")
result = evol_sim.run_simulation("burnup", {"burnup_steps": 5})
print(result)

print("\nTesting CORCA Xenon...")
xenon_sim = SimulatorFactory.create_simulator("corca_xenon")
result = xenon_sim.run_simulation("xenon", {"xenon_steps": 2, "power_1": 0.5, "rod_1": 200, "power_2": 0.8, "rod_2": 150})
print(result)
