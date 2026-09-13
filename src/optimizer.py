"""
主优化器模块
整合所有组件，提供统一的优化接口
"""
import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# 处理导入路径
# 在文件顶部添加导入
try:
    from .freefem_simulator import run_freefem_simulation, get_template_pde_type
    from .algorithm_library import AlgorithmLibrary  # 更新引用
    from .problem_detector import ProblemDetector
    from .problem_config import ProblemConfig
    from .utils import save_result_tool
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from freefem_simulator import run_freefem_simulation, get_template_pde_type
    from algorithm_library import AlgorithmLibrary  # 更新引用
    from problem_detector import ProblemDetector
    from problem_config import ProblemConfig
    from utils import save_result_tool


class DeepFreeFEMOptimizer:
    """主优化器类"""
    
    def __init__(self):
        load_dotenv()
        
        # API配置
        self.api_key = os.getenv("LLM_API_KEY")
        self.api_base = os.getenv("LLM_API_BASE", "https://api.deepseek.com")
        
        if not self.api_key:
            raise ValueError("请设置环境变量 LLM_API_KEY")

        if OpenAI is None:
            raise RuntimeError("缺少依赖 openai：请安装 openai 包后再运行 CLI")
            
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
        
        # 初始化各个组件
        self.problem_config = ProblemConfig()
        self.problem_detector = ProblemDetector(self.client)
        self.algorithm_library = AlgorithmLibrary()  # 更新类名
        
        # 初始化历史数据库
        try:
            from .optimization_history import OptimizationHistory
        except ImportError:
            import sys
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from optimization_history import OptimizationHistory
        
        self.history_db = OptimizationHistory()
        
        # 当前问题类型
        self.current_problem_type = None
    
    def optimize(self, analysis_config: dict, opt_params: List[str], method: str, optimizer: str) -> dict:
        """执行优化 - 支持复杂目标函数"""
        start_time = time.time()
        verbose = not bool((analysis_config or {}).get("brief_logs", False))
        
        # 使用传递的问题类型，并进行映射转换
        if 'problem_type' in analysis_config:
            detected_type = analysis_config['problem_type']
            # 如果检测到的是模板名，转换为问题类型
            self.current_problem_type = self.problem_config.get_problem_type_by_template(detected_type)
            if not self.current_problem_type or self.current_problem_type == detected_type:
                # 如果转换失败，直接使用检测到的类型
                self.current_problem_type = detected_type
        elif not hasattr(self, 'current_problem_type') or self.current_problem_type is None:
            self.current_problem_type = list(self.problem_config.problem_types.keys())[0]
        
        print(f"🔍 检测到问题类型: {analysis_config.get('problem_type', 'unknown')}")
        print(f"🎯 使用问题配置: {self.current_problem_type}")
        
        problem_config = self.problem_config.get_problem_config(self.current_problem_type)
        simulator_type = problem_config.get('simulator_type', 'freefem')
        corca_exec_state_path = None
        corca_hdf5_file_path = None
        if simulator_type == 'corca_state':
            corca_exec_state_path = os.getenv("CORCA_EXEC_STATE_PATH")
            corca_hdf5_file_path = os.getenv("CORCA_HDF5_FILE_PATH")
            if not corca_exec_state_path or not corca_hdf5_file_path:
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                corca_root = os.path.join(repo_root, "folderA", "preciseFZ")
                fallback_exec = os.path.join(corca_root, "apply", "exec_state")
                fallback_hdf5 = os.path.join(corca_root, "databank", "COMRES_last", "define_rod_01_001.hdf5")
                if os.path.exists(fallback_exec) and os.path.exists(fallback_hdf5):
                    corca_exec_state_path = corca_exec_state_path or fallback_exec
                    corca_hdf5_file_path = corca_hdf5_file_path or fallback_hdf5
                else:
                    return {"success": False, "message": "CORCA-sim 需要环境变量 CORCA_EXEC_STATE_PATH/CORCA_HDF5_FILE_PATH，或在 AICO1/folderA/preciseFZ 下提供 exec_state 与 databank/COMRES_last/*.hdf5"}
        
        # 调试信息
        print(f"📊 问题配置内容: {list(problem_config.keys()) if problem_config else 'None'}")
        
        # 获取可用参数
        available_params = problem_config.get('parameters', {})
        print(f"📋 可用参数数量: {len(available_params)}")
        
        # 如果参数为空，尝试其他问题类型
        if not available_params:
            print("⚠️ 当前问题类型参数为空，尝试查找匹配的问题类型...")
            detected_template = analysis_config.get('problem_type', '')
            
            # 尝试直接使用模板名作为问题类型
            for problem_type, config in self.problem_config.problem_types.items():
                if (config.get('template_name') == detected_template or 
                    problem_type == detected_template):
                    print(f"🔄 切换到问题类型: {problem_type}")
                    self.current_problem_type = problem_type
                    problem_config = self.problem_config.get_problem_config(problem_type)
                    available_params = problem_config.get('parameters', {})
                    break
        
        print(f"📋 最终可用参数: {list(available_params.keys())}")
        
        # 智能参数匹配（大小写不敏感）
        matched_params = []
        missing_params = []
        
        print(f"🔍 要匹配的参数: {opt_params}")
        print(f"📋 可用参数: {list(available_params.keys())}")
        
        for param in opt_params:
            # 首先尝试精确匹配
            if param in available_params:
                matched_params.append(param)
                print(f"✅ 精确匹配: '{param}'")
            else:
                # 尝试大小写不敏感匹配
                found = False
                for available_param in available_params.keys():
                    if param.lower() == available_param.lower():
                        matched_params.append(available_param)  # 使用正确的参数名
                        print(f"🔄 大小写匹配: '{param}' → '{available_param}'")
                        found = True
                        break
                
                # 尝试部分匹配（如 umax → uMax）
                if not found:
                    for available_param in available_params.keys():
                        if (param.lower().replace('max', '') in available_param.lower() or
                            available_param.lower().replace('max', '') in param.lower() or
                            param.lower() in available_param.lower() or
                            available_param.lower() in param.lower()):
                            matched_params.append(available_param)
                            print(f"🔄 模糊匹配: '{param}' → '{available_param}'")
                            found = True
                            break
                
                if not found:
                    missing_params.append(param)
                    print(f"❌ 未找到匹配: '{param}'")
        
        print(f"🎯 匹配结果: {matched_params}")
        print(f"❌ 未匹配: {missing_params}")
        
        # 检查是否有未匹配的参数
        if missing_params:
            print(f"❌ 以下参数不在可用参数列表中: {missing_params}")
            print(f"📋 所有可用参数: {list(available_params.keys())}")
            
            # 提供相似参数建议
            suggestions = []
            for missing_param in missing_params:
                for available_param in available_params.keys():
                    # 简单的相似度检查
                    if (missing_param.lower() in available_param.lower() or 
                        available_param.lower() in missing_param.lower()):
                        suggestions.append(f"'{missing_param}' → '{available_param}'")
            
            if suggestions:
                print(f"💡 参数建议: {', '.join(suggestions)}")
            
            return {
                "success": False, 
                "message": f"参数 {missing_params} 不存在。可用参数: {list(available_params.keys())[:10]}..."
            }
        
        # 使用匹配后的参数（这里是关键！）
        opt_params = matched_params  # 只使用匹配的参数

        def _resolve_param_ranges() -> Dict[str, Tuple[float, float]]:
            override = analysis_config.get("param_ranges_override")
            if override is None:
                override = analysis_config.get("param_ranges")
            override = override or {}
            out: Dict[str, Tuple[float, float]] = {}
            for p in opt_params:
                if p in override:
                    lo, hi = override[p]
                    out[p] = (float(lo), float(hi))
                else:
                    lo, hi = problem_config["parameters"][p]
                    out[p] = (float(lo), float(hi))
            return out

        param_ranges = _resolve_param_ranges()
        
        print(f"🚀 最终优化参数: {opt_params}")
        print(f"📊 参数范围: {param_ranges}")
        
        template_name = problem_config['template_name']
        simulator_type = problem_config.get('simulator_type', 'freefem')

        def _simulate(params_dict: dict) -> dict:
            if simulator_type == 'freefem':
                return run_freefem_simulation(template_name, params_dict)
            if simulator_type == 'corca_state':
                try:
                    from .corcasim_simulator import run_corcasim_simulation
                except ImportError:
                    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                    from corcasim_simulator import run_corcasim_simulation
                return run_corcasim_simulation(
                    template_name=template_name,
                    params=params_dict,
                    exec_state_path=corca_exec_state_path,
                    hdf5_file_path=corca_hdf5_file_path,
                )
            raise ValueError(f"未知仿真器类型: {simulator_type}")

        def _make_objective(history: list, verbose: bool):
            def f(params_list):
                params_dict = {name: value for name, value in zip(opt_params, params_list)}
                result = _simulate(params_dict)
                if verbose:
                    print(f"📊 Simulation result for {params_dict}: {result}")
                objective_value = self.algorithm_library.calculate_objective(
                    result, analysis_config, self.current_problem_type
                )
                try:
                    objective_value = float(objective_value)
                except Exception:
                    pass
                history.append({'params': params_dict, 'objective': objective_value, 'result': result})
                return objective_value
            return f

        iteration_history = []
        objective_function = _make_objective(iteration_history, verbose=verbose)

        def _warm_features(values: List[float], succ: List[int], warmup_k: int) -> Dict[str, float]:
            k = int(warmup_k)
            if k <= 0 or not values:
                return {
                    "warm_n": 0,
                    "warm_best": 0.0,
                    "warm_first": 0.0,
                    "warm_last": 0.0,
                    "warm_improve": 0.0,
                    "warm_slope": 0.0,
                    "warm_std": 0.0,
                    "warm_fail_rate": 1.0,
                    "warm_best_pos": 0.0,
                }
            seg = values[:k]
            n = len(seg)
            best = min(seg)
            first = seg[0]
            last = seg[-1]
            improve = float(first - best)
            slope = float((last - first) / float(max(1, n - 1)))
            mean = sum(seg) / float(n)
            var = sum((v - mean) ** 2 for v in seg) / float(max(1, n - 1))
            std = var ** 0.5
            best_pos = float(seg.index(best) / float(max(1, n - 1)))
            if succ:
                sseg = succ[:k]
                fail_rate = float(1.0 - (sum(sseg) / float(len(sseg)))) if sseg else 1.0
            else:
                fail_rate = 1.0
            return {
                "warm_n": int(n),
                "warm_best": float(best),
                "warm_first": float(first),
                "warm_last": float(last),
                "warm_improve": float(improve),
                "warm_slope": float(slope),
                "warm_std": float(std),
                "warm_fail_rate": float(fail_rate),
                "warm_best_pos": float(best_pos),
            }

        def _maybe_recommend_algorithm(current_optimizer: str, current_method: str) -> Tuple[str, str, Optional[dict]]:
            reccfg = analysis_config.get("algo_recommender") or {}
            if not reccfg or not bool(reccfg.get("enabled", False)):
                return current_optimizer, current_method, None

            user_specified = analysis_config.get('user_specified_algorithm')
            if user_specified and '-' in str(user_specified):
                return current_optimizer, current_method, {"skipped": True, "reason": "user_specified_algorithm"}

            model_path = str(reccfg.get("model_path") or "").strip()
            if not model_path:
                model_path = "/home/ycl/AICO-Intellig/results/model_algo_ranker_pool9_warm10.joblib"
            if not os.path.exists(model_path):
                return current_optimizer, current_method, {"skipped": True, "reason": "model_not_found", "model_path": model_path}

            warmup_iterations = int(reccfg.get("warmup_iterations", 10))
            warmup_time_s = float(reccfg.get("warmup_time_s", 30.0))
            warmup_k = int(reccfg.get("warmup_k", warmup_iterations))
            seed = int(reccfg.get("seed", 0))

            try:
                import joblib
                payload = joblib.load(model_path)
                model = payload.get("model")
                candidates = payload.get("candidates") or []
            except Exception as e:
                return current_optimizer, current_method, {"skipped": True, "reason": "model_load_failed", "error": str(e)}

            if not model or not candidates:
                return current_optimizer, current_method, {"skipped": True, "reason": "invalid_model_payload"}

            orig_iter = int(getattr(self.algorithm_library, "max_iterations", 100))
            orig_time = float(getattr(self.algorithm_library, "time_limit", 3600))
            orig_seed = int(getattr(self.algorithm_library, "random_seed", 42))

            scored = []
            try:
                self.algorithm_library.max_iterations = int(warmup_iterations)
                self.algorithm_library.time_limit = float(warmup_time_s)
                self.algorithm_library.random_seed = int(seed)

                base_feat = {}
                base_feat["problem_type"] = str(self.current_problem_type)
                base_feat["template_name"] = str(template_name)
                base_feat["pde_type"] = str(get_template_pde_type(str(template_name)) or "")
                if str(template_name) == "sns":
                    base_feat["physics_domain"] = "fluid"
                elif str(template_name) == "thmf":
                    base_feat["physics_domain"] = "heat"
                elif str(template_name) == "iaea":
                    base_feat["physics_domain"] = "neutronics"
                else:
                    base_feat["physics_domain"] = "unknown"
                base_feat["n_params"] = int(len(opt_params))
                for p in opt_params:
                    base_feat[f"param_{p}"] = 1
                widths = [float(hi) - float(lo) for (lo, hi) in param_ranges.values()]
                if widths:
                    base_feat["range_w_mean"] = float(sum(widths) / len(widths))
                    base_feat["range_w_max"] = float(max(widths))
                    base_feat["range_w_min"] = float(min(widths))
                else:
                    base_feat["range_w_mean"] = 0.0
                    base_feat["range_w_max"] = 0.0
                    base_feat["range_w_min"] = 0.0
                base_feat["budget_iter"] = int(warmup_iterations)
                base_feat["budget_time"] = float(warmup_time_s)
                expr = str(analysis_config.get('objective_expression') or "")
                base_feat["expr_len"] = int(len(expr))
                base_feat["has_abs"] = int("abs(" in expr)
                import re
                for k in re.findall(r"result\.get\(\s*'([^']+)'", expr):
                    base_feat[f"expr_key_{k}"] = 1

                for cand in candidates:
                    o = str(cand.get("optimizer"))
                    m = str(cand.get("method"))
                    if not o or not m:
                        continue
                    warm_hist = []
                    warm_vals: List[float] = []
                    warm_succ: List[int] = []
                    warm_obj = _make_objective(warm_hist, verbose=False)

                    def wrapped(params_list):
                        v = warm_obj(params_list)
                        try:
                            warm_vals.append(float(v))
                        except Exception:
                            pass
                        try:
                            if warm_hist and isinstance(warm_hist[-1], dict):
                                ok = warm_hist[-1].get("result", {}).get("success", True)
                                warm_succ.append(1 if bool(ok) else 0)
                        except Exception:
                            pass
                        return v

                    t0 = time.time()
                    if o == 'SciPy':
                        res = self.algorithm_library.run_scipy(wrapped, param_ranges, m)
                    elif o == 'Hyperopt':
                        res = self.algorithm_library.run_hyperopt(wrapped, param_ranges, m)
                    elif o == 'Optuna':
                        res = self.algorithm_library.run_optuna(wrapped, param_ranges, m)
                    elif o == 'Nevergrad':
                        res = self.algorithm_library.run_nevergrad(wrapped, param_ranges, m)
                    else:
                        continue
                    warm_time = float(time.time() - t0)

                    feat = dict(base_feat)
                    feat["optimizer"] = o
                    feat["method"] = m
                    feat["success"] = 1
                    feat.update(_warm_features(warm_vals, warm_succ, warmup_k))
                    try:
                        score = float(model.predict([feat])[0])
                    except Exception:
                        continue
                    scored.append(
                        {
                            "candidate": f"{o}-{m}",
                            "optimizer": o,
                            "method": m,
                            "pred_rank": score,
                            "warm_best": feat.get("warm_best"),
                            "warm_n": feat.get("warm_n"),
                            "warm_fail_rate": feat.get("warm_fail_rate"),
                            "warm_time_s": warm_time,
                            "warm_success": bool(res.get("success", False)),
                        }
                    )
            finally:
                self.algorithm_library.max_iterations = orig_iter
                self.algorithm_library.time_limit = orig_time
                self.algorithm_library.random_seed = orig_seed

            if not scored:
                return current_optimizer, current_method, {"skipped": True, "reason": "no_scores"}

            scored.sort(key=lambda x: float(x.get("pred_rank", 1e18)))
            best = scored[0]
            return best["optimizer"], best["method"], {
                "model_path": model_path,
                "warmup_iterations": warmup_iterations,
                "warmup_time_s": warmup_time_s,
                "warmup_k": warmup_k,
                "seed": seed,
                "chosen": best.get("candidate"),
                "topk": scored[: min(9, len(scored))],
            }

        optimizer, method, recommender_info = _maybe_recommend_algorithm(optimizer, method)
        
        print(f"\n🚀 开始优化: {optimizer} - {method}")
        print(f"📊 目标类型: {analysis_config.get('objective_type', 'simple')}")
        if analysis_config.get('objective_expression'):
            print(f"🎯 目标函数: {analysis_config['objective_expression']}")
        
        # 根据优化器选择执行路径
        if optimizer == 'SciPy':
            result = self.algorithm_library.run_scipy(objective_function, param_ranges, method)
        elif optimizer == 'Hyperopt':
            result = self.algorithm_library.run_hyperopt(objective_function, param_ranges, method)
        elif optimizer == 'Optuna':
            result = self.algorithm_library.run_optuna(objective_function, param_ranges, method)
        elif optimizer == 'Nevergrad':
            result = self.algorithm_library.run_nevergrad(objective_function, param_ranges, method)
        else:
            return {"success": False, "message": f"不支持的优化器: {optimizer}"}
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 统一输出格式
        final_result = {
            "optimizer": optimizer,
            "method": method,
            "execution_time": execution_time,
            "iteration_history": iteration_history,
            "analysis_config": analysis_config,
            "matched_params": opt_params,  # 记录实际使用的参数名
            "recommender": recommender_info,
            **result
        }
        
        self._save_results(final_result)
        return final_result

    def find_feasible_solutions(
        self,
        analysis_config: dict,
        opt_params: List[str],
        requested_count: int,
        max_attempts: int = None,
        timeout_s: float = None,
        seed: int = None,
    ) -> dict:
        start_time = time.time()
        import re
        import numpy as np

        if requested_count is None:
            requested_count = int(analysis_config.get("feasible_count", 1) or 1)
        requested_count = max(1, int(requested_count))

        if timeout_s is None:
            timeout_s = float(analysis_config.get("feasible_timeout", 3600) or 3600)
        timeout_s = max(1.0, float(timeout_s))

        constraints = analysis_config.get('constraints', []) or []
        if not constraints:
            return {"success": False, "message": "未提供约束条件，无法搜索可行解", "feasible_solutions": []}

        if 'problem_type' in analysis_config:
            detected_type = analysis_config['problem_type']
            self.current_problem_type = self.problem_config.get_problem_type_by_template(detected_type)
            if not self.current_problem_type or self.current_problem_type == detected_type:
                self.current_problem_type = detected_type
        elif not hasattr(self, 'current_problem_type') or self.current_problem_type is None:
            self.current_problem_type = list(self.problem_config.problem_types.keys())[0]

        problem_config = self.problem_config.get_problem_config(self.current_problem_type)
        if not problem_config:
            return {"success": False, "message": "未找到问题配置", "feasible_solutions": []}

        simulator_type = problem_config.get('simulator_type', 'freefem')
        if simulator_type != 'corca_state':
            return {"success": False, "message": "可行解搜索目前仅支持CORCA-sim (corca_state)", "feasible_solutions": []}

        corca_exec_state_path = os.getenv("CORCA_EXEC_STATE_PATH")
        corca_hdf5_file_path = os.getenv("CORCA_HDF5_FILE_PATH")
        if not corca_exec_state_path or not corca_hdf5_file_path:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            corca_root = os.path.join(repo_root, "folderA", "preciseFZ")
            fallback_exec = os.path.join(corca_root, "apply", "exec_state")
            fallback_hdf5 = os.path.join(corca_root, "databank", "COMRES_last", "define_rod_01_001.hdf5")
            if os.path.exists(fallback_exec) and os.path.exists(fallback_hdf5):
                corca_exec_state_path = corca_exec_state_path or fallback_exec
                corca_hdf5_file_path = corca_hdf5_file_path or fallback_hdf5
            else:
                return {"success": False, "message": "CORCA-sim 需要环境变量 CORCA_EXEC_STATE_PATH/CORCA_HDF5_FILE_PATH", "feasible_solutions": []}

        available_params = problem_config.get('parameters', {})
        if not opt_params:
            opt_params = list(available_params.keys())

        matched_params = []
        missing_params = []
        available_keys = list(available_params.keys())
        lower_map = {a.lower(): a for a in available_keys}
        for p in opt_params:
            if p in available_params:
                matched_params.append(p)
                continue
            lp = str(p).strip().lower()
            if lp in lower_map:
                matched_params.append(lower_map[lp])
                continue
            found = None
            for a in available_keys:
                la = a.lower()
                if lp in la or la in lp:
                    found = a
                    break
            if found:
                matched_params.append(found)
            else:
                missing_params.append(p)

        if missing_params:
            return {"success": False, "message": f"参数 {missing_params} 不存在", "feasible_solutions": []}

        opt_params = matched_params
        param_ranges = {p: problem_config['parameters'][p] for p in opt_params}
        template_name = problem_config['template_name']

        keff_lower = None
        keff_upper = None
        for c in constraints:
            if not isinstance(c, dict):
                continue
            if str(c.get('metric', '')).strip().lower() != 'keff':
                continue
            ctype = str(c.get('type', '')).strip().lower()
            val = c.get('value')
            if val is None:
                continue
            try:
                v = float(val)
            except Exception:
                continue
            if ctype in ['>=', 'ge']:
                keff_lower = v if keff_lower is None else max(keff_lower, v)
            elif ctype in ['<=', 'le']:
                keff_upper = v if keff_upper is None else min(keff_upper, v)

        if (keff_lower is not None or keff_upper is not None) and 'bore_ppm' in problem_config.get('parameters', {}):
            if 'bore_ppm' not in param_ranges:
                param_ranges['bore_ppm'] = problem_config['parameters']['bore_ppm']
                if 'bore_ppm' not in opt_params:
                    opt_params = [*opt_params, 'bore_ppm']

        safe_globals = {
            'result': None,
            'abs': abs, 'min': min, 'max': max, 'pow': pow,
            'sqrt': lambda x: x**0.5, 'float': float, '__builtins__': {}
        }

        def constraint_report(result: dict) -> dict:
            safe_globals['result'] = result
            penalty_weight = float(analysis_config.get('penalty_weight', 10000.0))
            total_penalty = 0.0
            violations = []
            for c in constraints:
                try:
                    if isinstance(c, dict):
                        metric = c.get('metric', '')
                        ctype = str(c.get('type', '')).lower()
                        val = c.get('value')
                        tol = float(c.get('tol', c.get('epsilon', 0.0)))
                        w = float(c.get('penalty_weight', 1000.0))

                        mv = result.get(metric) if metric else None
                        if mv is None and metric:
                            try:
                                mv = float(eval(metric, safe_globals))
                            except Exception:
                                mv = None
                        if mv is None:
                            continue

                        lower = c.get('lower', c.get('min', None))
                        upper = c.get('upper', c.get('max', None))
                        vio = 0.0
                        if ctype in ['between', 'range', 'in'] or (lower is not None and upper is not None):
                            if lower is None and isinstance(val, (list, tuple)) and len(val) == 2:
                                lower, upper = val
                            if lower is None and isinstance(val, dict):
                                lower = val.get('lower', val.get('min', None))
                                upper = val.get('upper', val.get('max', None))
                            if lower is not None and upper is not None:
                                mvf = float(mv)
                                lf = float(lower)
                                uf = float(upper)
                                vio = max(max(0.0, lf - mvf - tol), max(0.0, mvf - uf - tol))
                                if vio > 0:
                                    violations.append({"metric": metric, "type": "range", "lower": lf, "upper": uf, "actual": mvf, "tol": tol, "violation": vio})
                        else:
                            mvf = float(mv)
                            if val is None:
                                continue
                            vf = float(val)
                            if ctype in ['<=', 'le']:
                                vio = max(0.0, mvf - vf - tol)
                            elif ctype in ['>=', 'ge']:
                                vio = max(0.0, vf - mvf - tol)
                            elif ctype in ['==', 'eq']:
                                vio = max(0.0, abs(mvf - vf) - tol)
                            elif ctype == '<':
                                vio = max(0.0, mvf - vf)
                            elif ctype == '>':
                                vio = max(0.0, vf - mvf)
                            else:
                                vio = 0.0
                            if vio > 0:
                                violations.append({"metric": metric, "type": ctype, "target": vf, "actual": mvf, "tol": tol, "violation": vio})

                        total_penalty += float(w) * float(vio)
                    else:
                        s = str(c).strip()
                        vio = 0.0
                        if '<=' in s:
                            l, r = s.split('<=', 1)
                            vio = max(0.0, float(eval(l, safe_globals)) - float(eval(r, safe_globals)))
                        elif '>=' in s:
                            l, r = s.split('>=', 1)
                            vio = max(0.0, float(eval(r, safe_globals)) - float(eval(l, safe_globals)))
                        elif '==' in s:
                            l, r = s.split('==', 1)
                            vio = max(0.0, abs(float(eval(l, safe_globals)) - float(eval(r, safe_globals))))
                        elif '<' in s:
                            l, r = s.split('<', 1)
                            vio = max(0.0, float(eval(l, safe_globals)) - float(eval(r, safe_globals)))
                        elif '>' in s:
                            l, r = s.split('>', 1)
                            vio = max(0.0, float(eval(r, safe_globals)) - float(eval(l, safe_globals)))
                        if vio > 0:
                            violations.append({"expr": s, "violation": vio})
                        total_penalty += penalty_weight * float(vio)
                except Exception:
                    continue
            return {"feasible": float(total_penalty) <= 0.0, "penalty": float(total_penalty), "violations": violations}

        if max_attempts is None:
            max_attempts = int(analysis_config.get('feasible_max_attempts', 0) or 0)
        if not max_attempts:
            max_attempts = max(1000, requested_count * 200)

        rng = np.random.default_rng(seed)
        feasible_solutions = []
        all_samples = []
        all_samples_seen = 0
        attempts = 0

        try:
            from .corcasim_simulator import run_corcasim_simulation
        except ImportError:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from corcasim_simulator import run_corcasim_simulation

        while attempts < max_attempts and (time.time() - start_time) < timeout_s and len(feasible_solutions) < requested_count:
            attempts += 1
            params_dict = {}
            for name, (lo, hi) in param_ranges.items():
                if name == 'bore_ppm' and (keff_lower is not None or keff_upper is not None):
                    continue
                if str(name).strip().lower().startswith("pos") or (isinstance(lo, int) and isinstance(hi, int)):
                    params_dict[name] = int(rng.integers(int(lo), int(hi) + 1))
                else:
                    params_dict[name] = float(rng.uniform(float(lo), float(hi)))

            def _run_with(params_run: dict):
                return run_corcasim_simulation(
                    template_name=template_name,
                    params=params_run,
                    exec_state_path=corca_exec_state_path,
                    hdf5_file_path=corca_hdf5_file_path,
                )

            result = None
            if (keff_lower is not None or keff_upper is not None) and 'bore_ppm' in param_ranges:
                bore_lo, bore_hi = param_ranges.get('bore_ppm', (0, 2000))
                try:
                    bore_lo_i = int(bore_lo)
                    bore_hi_i = int(bore_hi)
                except Exception:
                    bore_lo_i, bore_hi_i = 0, 2000
                target = analysis_config.get('target_values', {}).get('keff')
                if target is None:
                    if keff_lower is not None and keff_upper is not None:
                        target = 0.5 * (float(keff_lower) + float(keff_upper))
                    elif keff_lower is not None:
                        target = float(keff_lower)
                    else:
                        target = float(keff_upper)
                target = float(target)

                bore_l = bore_lo_i
                bore_r = bore_hi_i
                found = False
                bore_selected = None
                for _ in range(int(analysis_config.get('bore_search_steps', 8) or 8)):
                    if bore_l > bore_r:
                        break
                    bore_mid = int((bore_l + bore_r) // 2)
                    bore_selected = bore_mid
                    trial_params = dict(params_dict)
                    trial_params['bore_ppm'] = bore_mid
                    r = _run_with(trial_params)
                    if not isinstance(r, dict) or not r.get('success'):
                        result = None
                        break
                    k = r.get('keff')
                    if k is None:
                        result = None
                        break
                    k = float(k)
                    result = r
                    if keff_lower is not None and k < float(keff_lower):
                        bore_r = bore_mid - 1
                        continue
                    if keff_upper is not None and k > float(keff_upper):
                        bore_l = bore_mid + 1
                        continue
                    found = True
                    break
                    
                    
                if not found:
                    continue
                if bore_selected is not None:
                    params_dict['bore_ppm'] = int(bore_selected)
            else:
                result = _run_with(params_dict)
            if not isinstance(result, dict) or not result.get('success'):
                continue
            rep = constraint_report(result)
            try:
                record_max = int(analysis_config.get('feasible_record_max', 50000) or 50000)
            except Exception:
                record_max = 50000
            record_max = max(0, int(record_max))
            record_all = analysis_config.get('feasible_record_all', True)
            record_all = bool(record_all) if not isinstance(record_all, str) else record_all.strip().lower() not in ['0', 'false', 'no']

            if record_all and record_max != 0:
                k = result.get('keff')
                fq = result.get('FQ')
                sample_result = {
                    'success': True,
                    'keff': float(k) if k is not None else None,
                    'FQ': float(fq) if fq is not None else None,
                    'feasible': bool(rep.get('feasible')),
                    'penalty': float(rep.get('penalty', 0.0)),
                    'violations': rep.get('violations', []) or [],
                }
                entry = {'params': dict(params_dict), 'objective': float(rep.get('penalty', 0.0)), 'result': sample_result, 'step': int(attempts)}
                all_samples_seen += 1
                if len(all_samples) < record_max:
                    all_samples.append(entry)
                else:
                    j = int(rng.integers(1, all_samples_seen + 1))
                    if j <= record_max:
                        all_samples[j - 1] = entry
            if rep.get('feasible'):
                feasible_solutions.append({"params": params_dict, "result": result, "constraints": rep})

        execution_time = time.time() - start_time
        success = len(feasible_solutions) >= requested_count
        final_result = {
            "optimizer": "FeasibleSearch",
            "method": "Random",
            "execution_time": execution_time,
            "iterations": attempts,
            "success": success,
            "message": "可行解搜索完成" if success else "未在预算内找到足够可行解",
            "analysis_config": {**analysis_config, "objective_type": "feasible", "primary_objective": "feasible"},
            "matched_params": opt_params,
            "feasible_requested": requested_count,
            "feasible_found": len(feasible_solutions),
            "feasible_solutions": feasible_solutions,
            "best_value": None,
            "best_params": feasible_solutions[0]["params"] if feasible_solutions else {},
            "iteration_history": all_samples,
        }
        self._save_results(final_result)
        return final_result
    
    def optimize_with_scheduler(self, analysis_config: dict, opt_params: List[str], 
                               initial_method: str, initial_optimizer: str) -> dict:
        """使用动态调度器的优化"""
        import sys
        import os
        
        try:
            from .dynamic_scheduler import DynamicScheduler
        except ImportError:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from dynamic_scheduler import DynamicScheduler
        
        # 获取完整的算法组合（18种）
        try:
            from .problem_config import get_optimizer_config
        except ImportError:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from problem_config import get_optimizer_config
        
        optimizer_config = get_optimizer_config()
        
        # 构建所有可用算法列表
        available_algorithms = []
        for optimizer_name, config in optimizer_config.items():
            for method in config['methods']:
                available_algorithms.append(f"{optimizer_name}-{method}")
        
        print(f"📋 可用算法总数: {len(available_algorithms)}")
        print(f"🔧 算法列表: {', '.join(available_algorithms)}")
        
        # 初始化调度器
        scheduler = DynamicScheduler(
            client=self.client,
            eval_interval=15,
            progress_protect_ratio=0.7,  # 提高到70%后保护
            time_protect=120,
            max_iter=300,  # 增加到300次
            confidence_threshold=0.6
        )
        
        current_optimizer = initial_optimizer
        current_method = initial_method
        current_algorithm = f"{current_optimizer}-{current_method}"
        
        print(f"🤖 启动智能调度优化: {current_algorithm}")
        print(f"📊 调度参数: 每{scheduler.eval_interval}次评估, {scheduler.progress_protect_ratio*100}%后保护")
        
        # 开始优化循环
        best_result = None
        total_iterations = 0
        algorithm_attempts = 0
        max_algorithm_attempts = 3  # 最多尝试3种算法
        
        while total_iterations < scheduler.max_iter and algorithm_attempts < max_algorithm_attempts:
            print(f"\n🔄 第{algorithm_attempts + 1}轮优化，使用算法: {current_algorithm}")
            
            # 执行当前算法的优化
            acfg = dict(analysis_config or {})
            if "algo_recommender" in acfg:
                try:
                    acfg["algo_recommender"] = dict(acfg.get("algo_recommender") or {})
                    acfg["algo_recommender"]["enabled"] = False
                except Exception:
                    pass
            partial_result = self.optimize(acfg, opt_params, current_method, current_optimizer)
            
            if not partial_result.get('success'):
                print(f"❌ 算法 {current_algorithm} 优化失败")
                break
            
            # 更新最佳结果
            if best_result is None or partial_result['best_value'] < best_result['best_value']:
                best_result = partial_result
            
            total_iterations += partial_result.get('iterations', 0)
            algorithm_attempts += 1
            
            # 记录损失到调度器
            scheduler.record_loss(total_iterations, partial_result['best_value'])
            
            # 检查是否达到目标
            if partial_result.get('target_achieved', False):
                print(f"🎯 目标已达成！停止优化")
                break
            
            # 检查是否建议切换算法
            if partial_result.get('should_switch_algorithm', False) and not scheduler.is_protected(total_iterations):
                print(f"🤔 当前算法效果不佳，考虑切换算法...")
                
                # 使用LLM决策下一个算法
                decision = scheduler.evaluate_and_decide(
                    total_iterations, current_algorithm, available_algorithms
                )
                
                if decision['action'] == 'switch' and decision['algorithm'] != current_algorithm:
                    # 切换算法
                    new_optimizer, new_method = decision['algorithm'].split('-', 1)
                    
                    print(f"🔄 切换算法: {current_algorithm} → {decision['algorithm']}")
                    print(f"   切换原因: {decision['reason']}")
                    
                    current_optimizer = new_optimizer
                    current_method = new_method
                    current_algorithm = decision['algorithm']
                else:
                    print(f"✅ LLM建议继续使用当前算法")
                    break
            else:
                # 不需要切换，结束优化
                break
        
        # 保存调度日志
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = f"results/scheduler_log_{timestamp}.json"
        scheduler.save_log(log_file)
        
        # 添加调度信息到结果
        if best_result:
            best_result['scheduler_summary'] = scheduler.get_decision_summary()
            best_result['scheduler_log_file'] = log_file
            best_result['total_algorithm_attempts'] = algorithm_attempts
            best_result['total_iterations'] = total_iterations
            best_result['available_algorithms'] = available_algorithms
        
        return best_result or {"success": False, "message": "优化失败"}
    
    def analyze_results(self, results: dict, primary_objective: str, target_values: dict) -> str:
        """分析优化结果"""
        if not results.get('success', False):
            return "优化失败，无法进行结果分析。"
        
        try:
            # 获取基本信息
            best_value = results.get('best_value', 'N/A')
            best_params = results.get('best_params', {})
            iterations = results.get('iterations', 0)
            execution_time = results.get('execution_time', 0)
            convergence_quality = results.get('convergence_quality', '未知')
            
            # 获取仿真结果
            iteration_history = results.get('iteration_history', [])
            if iteration_history:
                final_simulation = iteration_history[-1].get('result', {})
            else:
                final_simulation = {}
            
            # 分析目标达成情况
            target_achieved = results.get('target_achieved', False)
            
            analysis = f"""
### 📊 优化结果分析

#### 1. 优化成功度评估
**优化成功'**  
- **最优目标值**: {best_value}
- **收敛质量**: {convergence_quality}
- **迭代次数**: {iterations}次
- **执行时间**: {execution_time:.2f}秒

#### 2. 最优参数分析
"""
            
            for param, value in best_params.items():
                analysis += f"- **{param}**: {value:.6f}\n"
            
            analysis += f"""
#### 3. 目标函数分析
**主要目标**: {primary_objective}
**最优值**: {best_value}
"""
            
            # 分析目标值
            if 'pressuredrop' in primary_objective:
                analysis += f"""
- **压力降**: {final_simulation.get('pressuredrop', 'N/A')}
- **平均压力**: {final_simulation.get('avgpressure', 'N/A')}
- **入口压力**: {final_simulation.get('inletpressure', 'N/A')}
- **出口压力**: {final_simulation.get('outletpressure', 'N/A')}
"""
            elif 'keff' in primary_objective:
                target_keff = target_values.get('keff', 1.0)
                actual_keff = final_simulation.get('keff', 'N/A')
                if isinstance(actual_keff, (int, float)):
                    error = abs(float(actual_keff) - target_keff)
                    analysis += f"""
- **目标keff**: {target_keff}
- **实际keff**: {actual_keff}
- **误差**: {error:.6f}
"""
                else:
                    analysis += f"""
- **目标keff**: {target_keff}
- **实际keff**: {actual_keff}
"""
            elif 'efficiency' in primary_objective:
                analysis += f"""
- **效率**: {final_simulation.get('efficiency', 'N/A')}
"""
            
            analysis += f"""
#### 4. 收敛性分析
**算法表现**: {results.get('optimizer', 'Unknown')} - {results.get('method', 'Unknown')}
- **收敛质量**: {convergence_quality}
- **是否达到目标**: {'是' if target_achieved else '否'}
- **迭代效率**: {iterations/execution_time:.1f} 次/秒

#### 5. 改进建议
"""
            
            if convergence_quality in ['优秀', '良好']:
                analysis += "- ✅ 优化结果良好，参数设置合理\n"
                analysis += "- 💡 可以考虑进一步精细化参数范围以获得更精确的解\n"
            elif convergence_quality == '一般':
                analysis += "- ⚠️ 优化结果一般，建议尝试其他算法或增加迭代次数\n"
                analysis += "- 💡 可以考虑使用智能调度器自动切换算法\n"
            else:
                analysis += "- ❌ 优化结果不理想，建议检查参数范围和目标函数设置\n"
                analysis += "- 💡 尝试使用全局优化算法如DE或CMA-ES\n"
            
            if execution_time < 1:
                analysis += "- ⚡ 优化速度很快，可以考虑增加迭代次数以获得更好的解\n"
            elif execution_time > 300:
                analysis += "- 🐌 优化时间较长，可以考虑减少迭代次数或使用更快的算法\n"
            
            analysis += """
### 📋 总结
优化过程已完成，建议根据实际需求和约束条件评估结果的可行性。如需进一步优化，可以调整参数范围或尝试不同的优化算法。
"""
            
            return analysis.strip()
            
        except Exception as e:
            return f"结果分析出错: {str(e)}"
    
    def _save_results(self, result_data: dict):
        """保存结果到文件和历史数据库"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON结果
        json_filename = f"results/opt_result_{timestamp}.json"
        os.makedirs("results", exist_ok=True)
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存CSV格式的迭代历史
        if 'iteration_history' in result_data and result_data['iteration_history']:
            csv_filename = f"results/search_results_{timestamp}.csv"
            save_result_tool(result_data['iteration_history'], csv_filename)
        
        # 保存到历史数据库
        try:
            self.history_db.add_optimization_record(result_data)
            print(f"📊 历史记录已更新")
        except Exception as e:
            print(f"⚠️ 历史记录保存失败: {e}")
        
        print(f"✅ 结果已保存到: {json_filename}")


if __name__ == "__main__":
    # 测试代码
    try:
        optimizer = DeepFreeFEMOptimizer()
        print("✅ 优化器初始化成功")
        print(f"可用问题类型: {list(optimizer.problem_config.problem_types.keys())}")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
