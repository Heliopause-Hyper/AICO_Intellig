import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from simulator_interface import SimulatorFactory

sim = SimulatorFactory.create_simulator("corca_state")
print(sim.exec_state_path)
