"""
问题配置模块
动态构建问题类型和参数配置
"""
import os
import sys
from typing import Dict

# 处理导入路径
try:
    from .freefem_simulator import get_available_templates, get_template_parameters, get_template_outputs
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from freefem_simulator import get_available_templates, get_template_parameters, get_template_outputs


class ProblemConfig:
    """问题配置管理器"""
    
    def __init__(self):
        self.problem_types = self._build_dynamic_problem_types()
    
    def _build_dynamic_problem_types(self) -> Dict:
        """动态构建问题类型配置"""
        available_templates = get_available_templates()
        problem_types = {}
        
        # 预定义的别名映射和参数范围（作为默认值，如果自动生成失败则使用）
        template_configs = {
            'iaea': {
                'problem_type': 'neutron_diffusion',
                'name': 'IAEA-2D',
                'default_param_ranges': {
                    'D11': (1.35, 1.65), 'D12': (1.35, 1.65), 'D13': (1.35, 1.65), 'D14': (1.8, 2.2),
                    'D21': (0.35, 0.45), 'D22': (0.35, 0.45), 'D23': (0.35, 0.45), 'D24': (0.25, 0.35),
                    'Sigmaa11': (0.005, 0.015), 'Sigmaa12': (0.005, 0.015), 'Sigmaa13': (0.005, 0.015), 'Sigmaa14': (0.0, 0.005),
                    'Sigmaa21': (0.06, 0.10), 'Sigmaa22': (0.07, 0.10), 'Sigmaa23': (0.10, 0.16), 'Sigmaa24': (0.005, 0.015),
                    'musigmaf11': (0.0, 0.01), 'musigmaf12': (0.0, 0.01), 'musigmaf13': (0.0, 0.01), 'musigmaf14': (0.0, 0.01),
                    'musigmaf21': (0.12, 0.15), 'musigmaf22': (0.12, 0.15), 'musigmaf23': (0.12, 0.15), 'musigmaf24': (0.0, 0.01),
                    'chi11': (0.9, 1.1), 'chi12': (0.9, 1.1), 'chi13': (0.9, 1.1), 'chi14': (0.9, 1.1),
                    'chi21': (0.0, 0.1), 'chi22': (0.0, 0.1), 'chi23': (0.0, 0.1), 'chi24': (0.0, 0.1),
                    'Sigmas121': (0.015, 0.025), 'Sigmas122': (0.015, 0.025), 'Sigmas123': (0.015, 0.025), 'Sigmas124': (0.03, 0.05)
                }
            },
            'thmf': {
                'problem_type': 'thermal_fins',
                'name': 'HEAT-COND',
                'default_param_ranges': {
                    'k1': (0.1, 1.0), 'k2': (0.1, 1.0), 'k3': (0.1, 1.0),
                    'k4': (0.1, 1.0), 'k5': (0.1, 1.0), 'Bi': (0.01, 1.0)
                }
            },
            'sns': {
                'problem_type': 'convection_flow',
                'name': 'SNS-FLOW',
                'default_param_ranges': {
                    'uMax': (7.5, 12.5), 'Mu': (0.05, 0.15), 'nn': (5, 15),
                    'L': (2.5, 7.5), 'D': (0.5, 1.5), 'R': (0.1, 0.3)
                }
            },
            'ex_heat_time': {
                'problem_type': 'ex_heat_time',
                'name': 'HEAT-TIME',
            },
            'ex_blackscholes2d': {
                'problem_type': 'ex_blackscholes2d',
                'name': 'BLACK-SCHOLES2D',
            },
            'ex_advection2d': {
                'problem_type': 'ex_advection2d',
                'name': 'ADVECTION2D',
            },
            'ex_optimcontrol_inverse': {
                'problem_type': 'ex_optimcontrol_inverse',
                'name': 'OPTIMCONTROL-INVERSE',
                'default_param_ranges': {
                    'z0': (0.1, 5.0),
                    'z1': (0.1, 5.0),
                    'z2': (0.1, 5.0)
                }
            },
            'ex_vi_obstacle': {
                'problem_type': 'ex_vi_obstacle',
                'name': 'VI-OBSTACLE',
                'default_param_ranges': {
                    'c': (1.0, 20.0),
                    'gmax': (0.01, 0.2),
                    'f': (0.5, 5.0)
                }
            }
        }
        
        for template_name in available_templates:
            # 获取模板信息
            template_params = get_template_parameters(template_name)
            template_outputs = get_template_outputs(template_name)
            
            # 获取配置信息
            config = template_configs.get(template_name, {})
            problem_type = config.get('problem_type', template_name)
            name = config.get('name', template_name.replace('_', ' ').title())
            
            # 设置参数范围 - 强制使用预定义范围确保参数不为空
            param_ranges = {}
            default_ranges = config.get('default_param_ranges', {})
            
            # 优先使用预定义的参数范围
            if default_ranges:
                param_ranges = default_ranges.copy()
                print(f"📋 使用预定义参数范围: {template_name} ({len(param_ranges)}个参数)")
            else:
                # 如果没有预定义范围，尝试从模板参数生成
                for param, default_value in template_params.items():
                    if isinstance(default_value, (int, float)) and default_value != 0:
                        # 自动生成范围（±50%）
                        if default_value > 0:
                            min_val = default_value * 0.5
                            max_val = default_value * 1.5
                        else:
                            min_val = default_value * 1.5
                            max_val = default_value * 0.5
                        param_ranges[param] = (min_val, max_val)
                    else:
                        # 对于零值或非数值，设置合理的默认范围
                        param_ranges[param] = (0.0, 1.0)
            
            # 确保参数范围不为空
            if not param_ranges:
                print(f"⚠️ {template_name} 参数范围为空，使用默认参数")
                param_ranges = {
                    'param1': (0.1, 1.0),
                    'param2': (0.1, 1.0),
                    'param3': (0.1, 1.0)
                }
            
            problem_types[problem_type] = {
                'name': name,
                'template_name': template_name,
                'objectives': template_outputs,
                'parameters': param_ranges,
                'simulator_type': 'freefem'
            }

        if "convection_flow" in problem_types and "sns_flow_family" not in problem_types:
            base = problem_types.get("convection_flow") or {}
            problem_types["sns_flow_family"] = {
                "name": "SNS-FLOW",
                "template_name": base.get("template_name", "sns"),
                "objectives": base.get("objectives", []),
                "parameters": (base.get("parameters") or {}).copy(),
                "simulator_type": base.get("simulator_type", "freefem"),
            }
            problem_types["sns_flow_family_v2"] = {
                "name": "SNS-FLOW-V2",
                "template_name": base.get("template_name", "sns"),
                "objectives": base.get("objectives", []),
                "parameters": (base.get("parameters") or {}).copy(),
                "simulator_type": base.get("simulator_type", "freefem"),
            }
            problem_types["sns_flow_family_v3"] = {
                "name": "SNS-FLOW-V3",
                "template_name": base.get("template_name", "sns"),
                "objectives": base.get("objectives", []),
                "parameters": (base.get("parameters") or {}).copy(),
                "simulator_type": base.get("simulator_type", "freefem"),
            }

        if "thermal_fins" in problem_types and "thermal_fins_family_v1" not in problem_types:
            base = problem_types.get("thermal_fins") or {}
            problem_types["thermal_fins_family_v1"] = {
                "name": "HEAT-COND-V1",
                "template_name": base.get("template_name", "thmf"),
                "objectives": base.get("objectives", []),
                "parameters": (base.get("parameters") or {}).copy(),
                "simulator_type": base.get("simulator_type", "freefem"),
            }

        if "neutron_diffusion" in problem_types and "neutron_diffusion_family_v1" not in problem_types:
            base = problem_types.get("neutron_diffusion") or {}
            problem_types["neutron_diffusion_family_v1"] = {
                "name": "IAEA-2D-V1",
                "template_name": base.get("template_name", "iaea"),
                "objectives": base.get("objectives", []),
                "parameters": (base.get("parameters") or {}).copy(),
                "simulator_type": base.get("simulator_type", "freefem"),
            }

        if "ex_heat_time" in problem_types and "ex_heat_time_family_v1" not in problem_types:
            base = problem_types.get("ex_heat_time") or {}
            problem_types["ex_heat_time_family_v1"] = {
                "name": "HEAT-TIME-V1",
                "template_name": base.get("template_name", "ex_heat_time"),
                "objectives": base.get("objectives", []),
                "parameters": (base.get("parameters") or {}).copy(),
                "simulator_type": base.get("simulator_type", "freefem"),
            }

        if "ex_blackscholes2d" in problem_types and "ex_blackscholes2d_family_v1" not in problem_types:
            base = problem_types.get("ex_blackscholes2d") or {}
            problem_types["ex_blackscholes2d_family_v1"] = {
                "name": "BLACK-SCHOLES2D-V1",
                "template_name": base.get("template_name", "ex_blackscholes2d"),
                "objectives": base.get("objectives", []),
                "parameters": (base.get("parameters") or {}).copy(),
                "simulator_type": base.get("simulator_type", "freefem"),
            }

        if "ex_advection2d" in problem_types and "ex_advection2d_family_v1" not in problem_types:
            base = problem_types.get("ex_advection2d") or {}
            problem_types["ex_advection2d_family_v1"] = {
                "name": "ADVECTION2D-V1",
                "template_name": base.get("template_name", "ex_advection2d"),
                "objectives": base.get("objectives", []),
                "parameters": (base.get("parameters") or {}).copy(),
                "simulator_type": base.get("simulator_type", "freefem"),
            }
            
        if "ex_optimcontrol_inverse" in problem_types and "ex_optimcontrol_inverse_family_v1" not in problem_types:
            base = problem_types.get("ex_optimcontrol_inverse") or {}
            problem_types["ex_optimcontrol_inverse_family_v1"] = {
                "name": "OPTIMCONTROL-INVERSE-V1",
                "template_name": base.get("template_name", "ex_optimcontrol_inverse"),
                "objectives": base.get("objectives", []),
                "parameters": (base.get("parameters") or {}).copy(),
                "simulator_type": base.get("simulator_type", "freefem"),
            }

        if "ex_vi_obstacle" in problem_types and "ex_vi_obstacle_family_v1" not in problem_types:
            base = problem_types.get("ex_vi_obstacle") or {}
            problem_types["ex_vi_obstacle_family_v1"] = {
                "name": "VI-OBSTACLE-V1",
                "template_name": base.get("template_name", "ex_vi_obstacle"),
                "objectives": base.get("objectives", []),
                "parameters": (base.get("parameters") or {}).copy(),
                "simulator_type": base.get("simulator_type", "freefem"),
            }
        
        problem_types['corca_state'] = {
            'name': 'CORCA 仿真（临界/硼浓度/棒位）',
            'template_name': 'default',
            'objectives': ['keff', 'FQ', 'FDH', 'AO', 'deltai', 'bore', 'burnup'],
            'parameters': {
                'bore_ppm': (0, 2000),
                'Pp:': (13.0, 17.0),
                'Prk:': (0, 100),
                'Tin:': (285.0, 305.0),
                'pos1: R, K 04 ': (0, 225),
                'pos2: R, F 04 ': (0, 225),
                'pos3: R, M 06 ': (0, 225),
                'pos4: R, J 06 ': (0, 225),
                'pos5: R, G 06 ': (0, 225),
                'pos6: R, D 06 ': (0, 225),
                'pos7: R, K 07 ': (0, 225),
                'pos8: R, F 07 ': (0, 225),
                'pos9: R, K 09 ': (0, 225),
                'pos10: R, F 09 ': (0, 225),
                'pos11: R, M 10 ': (0, 225),
                'pos12: R, J 10 ': (0, 225),
                'pos13: R, G 10 ': (0, 225),
                'pos14: R, D 10 ': (0, 225),
                'pos15: R, K 12 ': (0, 225),
                'pos16: R, F 12 ': (0, 225)
            },
            'simulator_type': 'corca_state'
        }
        
        problem_types['corca_evol'] = {
            'name': 'CORCA 燃耗仿真',
            'template_name': 'burnup',
            'objectives': ['keff', 'FQ', 'burnup_finished'],
            'parameters': {
                'sequence_count': (1, 3),
                'burnup_steps_1': (10, 100),
                'rod_1': (0, 225),
                'burnup_steps_2': (10, 100),
                'rod_2': (0, 225),
                'burnup_steps_3': (10, 100),
                'rod_3': (0, 225)
            },
            'simulator_type': 'corca_evol'
        }
        
        problem_types['corca_xenon'] = {
            'name': 'CORCA 氙瞬态计算',
            'template_name': 'xenon',
            'objectives': ['keff', 'xenon_finished'],
            'parameters': {
                'xenon_steps': (1, 10),
                'power_1': (0.1, 1.0),
                'rod_1': (0, 225)
            },
            'simulator_type': 'corca_xenon'
        }
        
        return problem_types
    
    def get_problem_type_by_template(self, template_name: str) -> str:
        """根据模板名称获取问题类型"""
        if template_name in self.problem_types:
            return template_name
        for problem_type, config in self.problem_types.items():
            if config['template_name'] == template_name:
                return problem_type
        return list(self.problem_types.keys())[0] if self.problem_types else 'default'
    
    def get_template_by_problem_type(self, problem_type: str) -> str:
        """根据问题类型获取模板名称"""
        return self.problem_types.get(problem_type, {}).get('template_name', 'iaea')
    
    def get_available_problem_types(self) -> list:
        """获取所有可用的问题类型"""
        return list(self.problem_types.keys())
    
    def get_problem_config(self, problem_type: str) -> dict:
        """获取指定问题类型的配置"""
        return self.problem_types.get(problem_type, {})


# 优化器配置
OPTIMIZER_CONFIG = {
    'SciPy': {
        'name': 'scipy',
        'methods': ['Nelder-Mead', 'Powell', 'DE', 'COBYLA', 'SA']
    },
    'Optuna': {
        'name': 'optuna',
        'methods': ['Grid', 'TPE', 'CMA-ES', 'Random']
    },
    'Hyperopt': {
        'name': 'hyperopt',
        'methods': ['TPE', 'Random', 'AdaptiveTPE']
    },
    'Nevergrad': {
        'name': 'nevergrad',
        'methods': ['DE', 'CMA', 'PSO', 'OnePlusOne', 'TwoPointsDE', 'NGOpt']
    }
}


def get_optimizer_config() -> dict:
    """获取优化器配置"""
    return OPTIMIZER_CONFIG


if __name__ == "__main__":
    try:
        config = ProblemConfig()
        print("✅ 问题配置初始化成功")
        print(f"可用问题类型: {config.get_available_problem_types()}")
        for problem_type in config.get_available_problem_types():
            problem_config = config.get_problem_config(problem_type)
            print(f"- {problem_type}: {problem_config.get('name', 'Unknown')}")
            print(f"  参数数量: {len(problem_config.get('parameters', {}))}")
            params = list(problem_config.get('parameters', {}).keys())[:5]
            print(f"  示例参数: {params}")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
