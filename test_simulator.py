import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from simulator_interface import SimulatorFactory

sim = SimulatorFactory.create_simulator("corca_state")
res = sim.run_simulation("default", {"Pp:": 15, "Prk:": 50, "Tin:": 300, "bore_ppm": 1000})
print(res)
