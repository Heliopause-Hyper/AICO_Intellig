"""
FreeFEM仿真模块
负责所有与FreeFEM相关的仿真调用
"""
import os
import re
import subprocess
import tempfile
import uuid
import glob
import numpy as np
from typing import Dict


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _templates_dir() -> str:
    return os.path.join(_repo_root(), "templates")


def run_freefem_simulation(template_name: str, params: dict) -> dict:
    """
    通用的FreeFEM仿真函数，基于模板名称自动处理输入输出文件
    """
    try:
        # 尝试导入SimulatorInterface
        import sys
        # 处理相对导入和绝对导入
        try:
            from .simulator_interface import FreeFEMSimulator
        except ImportError:
            try:
                from src.simulator_interface import FreeFEMSimulator
            except ImportError:
                # 如果都在src下，直接导入
                import simulator_interface
                FreeFEMSimulator = simulator_interface.FreeFEMSimulator
            
        simulator = FreeFEMSimulator()
        simulator.template_dir = _templates_dir()
        return simulator.run_simulation(template_name, params)
        
    except Exception as e:
        print(f"❌ FreeFEM仿真出错: {e}")
        return {"error": str(e), "success": False}


def get_available_templates() -> list:
    """获取所有可用的FreeFEM模板"""
    template_dir = _templates_dir()
    edp_files = glob.glob(os.path.join(template_dir, "*.edp"))
    return [os.path.basename(f).replace('.edp', '') for f in edp_files]


def get_template_parameters(template_name: str) -> dict:
    """获取模板的默认参数"""
    input_file = os.path.join(_templates_dir(), f"{template_name}_input.txt")
    
    # 预定义的参数范围（作为备用）
    default_ranges = {
        'iaea': {
            'D11': 1.5, 'D12': 1.5, 'D13': 1.5, 'D14': 2.0,
            'D21': 0.4, 'D22': 0.4, 'D23': 0.4, 'D24': 0.3,
            'Sigmaa11': 0.01, 'Sigmaa12': 0.01, 'Sigmaa13': 0.01, 'Sigmaa14': 0.0025,
            'Sigmaa21': 0.08, 'Sigmaa22': 0.085, 'Sigmaa23': 0.13, 'Sigmaa24': 0.01,
            'musigmaf11': 0.005, 'musigmaf12': 0.005, 'musigmaf13': 0.005, 'musigmaf14': 0.005,
            'musigmaf21': 0.135, 'musigmaf22': 0.135, 'musigmaf23': 0.135, 'musigmaf24': 0.005,
            'chi11': 1.0, 'chi12': 1.0, 'chi13': 1.0, 'chi14': 1.0,
            'chi21': 0.05, 'chi22': 0.05, 'chi23': 0.05, 'chi24': 0.05,
            'Sigmas121': 0.02, 'Sigmas122': 0.02, 'Sigmas123': 0.02, 'Sigmas124': 0.04
        },
        'thmf': {
            'k1': 0.5, 'k2': 0.5, 'k3': 0.5,
            'k4': 0.5, 'k5': 0.5, 'Bi': 0.5
        },
        'sns': {
            'uMax': 10.0, 'Mu': 0.1, 'nn': 10,
            'L': 5.0, 'D': 1.0, 'R': 0.2
        }
    }
    
    parameters = {}
    
    # 首先尝试从输入文件解析
    if os.path.exists(input_file):
        try:
            with open(input_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        param_name, param_value = line.split('=', 1)
                        param_name = param_name.strip()
                        param_value = param_value.strip()
                        
                        try:
                            # 尝试解析数值
                            default_value = float(param_value)
                            parameters[param_name] = default_value
                            
                        except ValueError:
                            # 如果不是数值，跳过
                            continue
        except Exception as e:
            print(f"⚠️ 解析输入文件失败: {e}")
    
    # 如果解析失败或参数为空，使用预定义范围
    if not parameters and template_name in default_ranges:
        parameters = default_ranges[template_name].copy()
        print(f"📋 使用预定义参数: {template_name}")
    
    # 如果仍然为空，生成基本参数
    if not parameters:
        print(f"⚠️ 无法获取 {template_name} 的参数，使用默认参数")
        parameters = {
            'param1': 1.0,
            'param2': 1.0,
            'param3': 1.0
        }
    
    return parameters


def get_template_outputs(template_name: str) -> list:
    """
    从模板的输出文件中获取输出变量名称
    
    Args:
        template_name: 模板名称
    
    Returns:
        输出变量名称列表
    """
    output_file = os.path.join(_templates_dir(), f"{template_name}_output.txt")
    outputs = []
    
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key = line.split('=', 1)[0].strip()
                    if key not in outputs:
                        outputs.append(key)
    
    return outputs


def get_template_pde_type(template_name: str) -> str:
    mapping = {
        "sns": "elliptic",
        "thmf": "elliptic",
        "iaea": "elliptic",
        "ex_poisson_adapt_indicator": "elliptic",
        "ex_optimcontrol_inverse": "elliptic",
        "ex_vi_obstacle": "elliptic",
        "ex_lapeigen": "elliptic",
        "ex_heat_time": "parabolic",
        "ex_blackscholes2d": "parabolic",
        "ex_advection2d": "hyperbolic",
        "colorbar": "unknown",
    }
    if template_name in mapping:
        return str(mapping[template_name])

    edp_path = os.path.join(_templates_dir(), f"{template_name}.edp")
    try:
        with open(edp_path, "r", encoding="utf-8") as f:
            s = f.read()
    except Exception:
        return "unknown"

    if "EigenValue(" in s:
        return "elliptic"
    if "convect(" in s and ("dt" in s or "for (" in s):
        if "dx(" not in s and "dy(" not in s and "int2d" not in s:
            return "hyperbolic"
        return "parabolic"
    if "dx(" in s or "dy(" in s or "int2d" in s:
        return "elliptic"
    return "unknown"


# 向后兼容的包装器函数
def run_freefem_tool(params: dict) -> dict:
    """运行FreeFEM中子扩散仿真，输入参数字典，返回keff和FQ"""
    return run_freefem_simulation("iaea", params)


def run_thermal_fins_tool(params: dict) -> dict:
    """运行thermal fins FreeFEM仿真，输入参数字典，返回传热效率等指标"""
    return run_freefem_simulation("thmf", params)


def run_convection_flow_tool(params: dict) -> dict:
    """运行对流问题FreeFEM仿真，输入参数字典，返回压力降和平均压力"""
    return run_freefem_simulation("sns", params)
