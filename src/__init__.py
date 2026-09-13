"""
AICO优化系统包初始化文件
"""
from .optimizer import DeepFreeFEMOptimizer
from .freefem_simulator import (
    run_freefem_simulation,
    run_freefem_tool,
    run_thermal_fins_tool,
    run_convection_flow_tool,
    get_available_templates
)
from .corcasim_simulator import run_corcasim_simulation
from .utils import save_result_tool

__all__ = [
    'DeepFreeFEMOptimizer',
    'run_freefem_simulation',
    'run_freefem_tool',
    'run_thermal_fins_tool', 
    'run_convection_flow_tool',
    'get_available_templates',
    'run_corcasim_simulation',
    'save_result_tool'
]
