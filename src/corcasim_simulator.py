import os
import sys
from typing import Dict


def run_corcasim_simulation(
    template_name: str,
    params: dict,
    exec_state_path: str,
    hdf5_file_path: str,
    template_dir: str = "templates/corcasim",
) -> Dict:
    try:
        try:
            from .simulator_interface import CORCAStateSimulator
        except ImportError:
            try:
                from src.simulator_interface import CORCAStateSimulator
            except ImportError:
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from simulator_interface import CORCAStateSimulator

        simulator = CORCAStateSimulator(
            exec_state_path=exec_state_path,
            hdf5_file_path=hdf5_file_path,
            template_dir=template_dir,
        )
        return simulator.run_simulation(template_name, params)
    except Exception as e:
        return {"error": str(e), "success": False}

