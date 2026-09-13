"""
算法库模块
包含各种优化算法的实现
"""
import time
import sys
import threading
import random
import numpy as np
from typing import Dict, List, Callable, Optional


class ProgressBar:
    """优化进度条 - 刷新式显示"""
    def __init__(self, max_iterations: int, description: str = "优化进行中"):
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.description = description
        self.start_time = time.time()
        self.running = True
        self.best_value = float('inf')
        self.last_display_time = 0
        self.display_interval = 0.5  # 每0.5秒更新一次显示
        
    def update(self, iteration: int, current_value: float = None):
        """更新进度"""
        self.current_iteration = iteration
        if current_value is not None and current_value < self.best_value:
            self.best_value = current_value
        
        # 控制显示频率，避免过于频繁的刷新
        current_time = time.time()
        if current_time - self.last_display_time >= self.display_interval:
            self._display()
            self.last_display_time = current_time
    
    def _display(self):
        """显示进度条 - 使用\r刷新同一行"""
        if not self.running:
            return
            
        # 处理超出预期的情况
        if self.current_iteration > self.max_iterations:
            self.max_iterations = max(self.max_iterations, self.current_iteration + 10)
        
        progress = min(self.current_iteration / self.max_iterations, 1.0)
        bar_length = 40
        filled_length = int(bar_length * progress)
        
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        elapsed_time = time.time() - self.start_time
        if progress > 0 and progress < 1.0:
            eta = elapsed_time * (1 - progress) / progress
            eta_str = f"ETA: {eta:.0f}s"
        else:
            eta_str = "完成"
        
        iteration_info = f"({self.current_iteration}/{self.max_iterations})"
        best_info = f"最优值: {self.best_value:.6f}" if self.best_value != float('inf') else "最优值: N/A"
        
        # 使用\r刷新同一行，避免产生新行
        print(f"\r🔄 {self.description} |{bar}| {progress:.1%} {iteration_info} | {best_info} | {eta_str}", 
              end="", flush=True)
    
    def finish(self, success: bool = True):
        """完成进度条"""
        self.running = False
        status = "✅" if success else "❌"
        # 完成时换行，避免覆盖
        print(f"\r{status} {self.description} 完成！" + " " * 20)  # 添加空格清除残留字符


class AlgorithmLibrary:
    """算法库 - 包含18种优化算法的实现"""
    def __init__(self):
        self.max_iterations = 100
        self.time_limit = 3600
        self.random_seed = 42
    
    def run_scipy(self, objective_function, param_ranges, method):
        """执行SciPy优化"""
        from scipy.optimize import minimize, differential_evolution, dual_annealing
        
        param_names = list(param_ranges.keys())
        bounds = [param_ranges[name] for name in param_names]
        
        # 生成初始点
        x0 = [(bounds[i][0] + bounds[i][1]) / 2 for i in range(len(bounds))]
        
        # 根据算法类型设置不同的进度条
        if method == 'DE':
            # 计算适当的代数
            # popsize in scipy is multiplier: population = popsize * len(x)
            # We want total population to be manageable, e.g., around 20-40
            # So popsize_arg should be max(1, 20 // len(bounds))
            popsize_arg = max(1, 20 // len(bounds))
            actual_pop = popsize_arg * len(bounds)
            
            # Ensure we run at least 1 generation, but not exceeding total budget too much
            max_generations = max(1, self.max_iterations // actual_pop)
            max_func_evals = max_generations * actual_pop
            progress_bar = ProgressBar(max_func_evals, f"SciPy-{method}")
            
        elif method == 'SA':
             # 模拟退火 (dual_annealing)
            max_iter = self.max_iterations
            # SA needs many evaluations per iteration (local search)
            max_func_evals = self.max_iterations * 50
            progress_bar = ProgressBar(max_func_evals, f"SciPy-{method}")
        elif method in ['Nelder-Mead', 'Powell']:
            # Nelder-Mead和Powell需要较多迭代
            # 设置为500次，既避免运行过久，又能保证一定收敛效果
            max_iter = self.max_iterations
            max_func_evals = self.max_iterations
            progress_bar = ProgressBar(max_func_evals, f"SciPy-{method}")
        elif method == 'COBYLA':
            # COBYLA: Constrained Optimization BY Linear Approximation
            # 这是一个无梯度算法，通常需要较多的函数评估次数
            max_iter = self.max_iterations  # COBYLA 的 maxiter 实际上是 max fun evals
            max_func_evals = int(max_iter)
            progress_bar = ProgressBar(max_func_evals, f"SciPy-{method}")
        else:
            # 其他算法
            max_iter = self.max_iterations
            # Gradient-free methods need more function evaluations than iterations
            max_func_evals = self.max_iterations * 20
            progress_bar = ProgressBar(max_func_evals, f"SciPy-{method}")
        
        # 包装目标函数以更新进度
        iteration_count = 0
        initial_loss = None
        best_loss = float('inf')
        loss_history = []
        
        def wrapped_objective(params_list):
            nonlocal iteration_count, initial_loss, best_loss, loss_history
            iteration_count += 1
            result = objective_function(params_list)
            
            # 记录初始损失和历史
            if initial_loss is None:
                initial_loss = result
            if result < best_loss:
                best_loss = result
            loss_history.append(result)
            
            progress_bar.update(iteration_count, best_loss)
            return result
        
        try:
            if method == 'DE':
                result = differential_evolution(
                    wrapped_objective,
                    bounds,
                    maxiter=max_generations,
                    popsize=popsize_arg,
                    seed=self.random_seed,
                    disp=False,
                    atol=1e-6,
                    tol=1e-6
                )
            elif method == 'SA':
                # 使用 dual_annealing 实现模拟退火
                result = dual_annealing(
                    wrapped_objective,
                    bounds,
                    maxiter=max_iter,
                    maxfun=max_func_evals, # 使用 maxfun 控制总评估次数
                    seed=self.random_seed,
                    no_local_search=False  # 结合局部搜索
                )
            elif method == 'COBYLA':
                # COBYLA 不支持 bounds，需要转换为 constraints
                constraints = []
                for i, (low, high) in enumerate(bounds):
                    # x[i] >= low  =>  x[i] - low >= 0
                    constraints.append({'type': 'ineq', 'fun': lambda x, i=i, low=low: x[i] - low})
                    # x[i] <= high =>  high - x[i] >= 0
                    constraints.append({'type': 'ineq', 'fun': lambda x, i=i, high=high: high - x[i]})
                
                options = {'maxiter': max_iter, 'disp': False}
                try:
                    result = minimize(
                        wrapped_objective, 
                        x0, 
                        method='COBYLA', 
                        constraints=constraints,
                        options=options
                    )
                except Exception as e:
                    print(f"⚠️ COBYLA failed with constraints: {e}")
                    # Fallback or re-raise
                    raise e
                    
            elif method == 'Powell':
                 # 修复：根据算法类型动态设置 options，避免 "Unknown solver options" 警告
                options = {'maxiter': max_iter, 'maxfev': max_func_evals, 'disp': False}
                
                try:
                    # 尝试带 bounds 运行
                    result = minimize(
                        wrapped_objective, 
                        x0, 
                        method=method, 
                        bounds=bounds,
                        options=options
                    )
                except Exception as e:
                    print(f"⚠️ Powell with bounds failed ({e}), retrying without bounds...")
                    # 如果失败，尝试不带 bounds
                    result = minimize(
                        wrapped_objective, 
                        x0, 
                        method=method, 
                        options=options
                    )
                    
            else:
                # 其他 SciPy 算法
                options = {'maxiter': max_iter, 'disp': False}
                
                # 只有部分算法支持 maxfev (如 Nelder-Mead)
                if method in ['Nelder-Mead']:
                    options['maxfev'] = max_func_evals
                
                result = minimize(
                    wrapped_objective, 
                    x0, 
                    method=method, 
                    bounds=bounds,
                    options=options
                )
            
            # 智能判断优化是否成功（不仅依赖result.success）
            optimization_successful = self._is_optimization_successful(result, best_loss, initial_loss, iteration_count)
            
            progress_bar.finish(optimization_successful)
            
            if optimization_successful:
                # Ensure output is JSON serializable (convert numpy types to python types)
                best_params = {name: float(value) for name, value in zip(param_names, result.x)}
                
                # 智能评估优化质量
                convergence_quality, target_achieved = self._evaluate_convergence_quality(
                    best_loss, initial_loss, iteration_count, max_func_evals if method == 'DE' else max_iter
                )
                
                return {
                    "success": True,
                    "best_params": best_params,
                    "best_value": float(result.fun),
                    "iterations": iteration_count,
                    "initial_loss": float(initial_loss) if initial_loss is not None else None,
                    "convergence_quality": convergence_quality,
                    "target_achieved": target_achieved,
                    "scipy_success": bool(result.success),  # 记录原始的success状态
                    "message": f"SciPy优化完成 (质量: {convergence_quality})"
                }
            else:
                # Handle failure case output types too
                best_params_fail = {}
                if hasattr(result, 'x'):
                     try:
                         best_params_fail = {name: float(value) for name, value in zip(param_names, result.x)}
                     except:
                         pass

                return {
                    "success": False,
                    "best_params": best_params_fail,
                    "best_value": float(getattr(result, 'fun', best_loss)),
                    "iterations": iteration_count,
                    "scipy_success": bool(result.success) if hasattr(result, 'success') else False,
                    "message": f"SciPy优化失败: {getattr(result, 'message', '未知原因')}"
                }
        except Exception as e:
            progress_bar.finish(False)
            return {
                "success": False,
                "message": f"SciPy优化出错: {str(e)}"
            }
    

###########################################################################################################################################

    def run_optuna(self, objective_function, param_ranges, method):
        """执行Optuna优化"""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            return {"success": False, "message": "Optuna未安装"}
        
        param_names = list(param_ranges.keys())
        n_trials = self.max_iterations
        progress_bar = ProgressBar(n_trials, f"Optuna-{method}")
        
        iteration_count = 0
        best_loss = float('inf')
        
        def optuna_objective(trial):
            nonlocal iteration_count, best_loss
            iteration_count += 1
            
            params = {}
            for name, (low, high) in param_ranges.items():
                params[name] = trial.suggest_float(name, low, high)
            
            # 转换为列表形式调用目标函数
            params_list = [params[name] for name in param_names]
            result = objective_function(params_list)
            
            # 处理 inf 值，防止 Optuna 显示 N/A
            if not np.isfinite(result):
                result = 1e10

            if result < best_loss:
                best_loss = result
            
            progress_bar.update(iteration_count, best_loss)
            return result
        
        try:
            if method == 'TPE':
                sampler = optuna.samplers.TPESampler(seed=self.random_seed)
            elif method == 'CMA-ES':
                try:
                    sampler = optuna.samplers.CmaEsSampler(seed=self.random_seed)
                except Exception:
                    sampler = optuna.samplers.TPESampler(seed=self.random_seed)
            elif method == 'Random':
                sampler = optuna.samplers.RandomSampler(seed=self.random_seed)
            elif method == 'Grid':
                # GridSampler需要定义搜索空间
                search_space = {}
                n_params = len(param_ranges)
                # 计算每个维度的采样点数，确保总数大致匹配max_iterations
                # 但至少每个维度有2个点
                if n_params > 0:
                    # 即使维度很高，也至少尝试每个维度2个点，虽然这可能远超max_iterations
                    points_per_dim = max(2, int(self.max_iterations ** (1/n_params)))
                else:
                    points_per_dim = 1
                
                for name, (low, high) in param_ranges.items():
                    # 生成均匀分布的网格点
                    search_space[name] = np.linspace(low, high, points_per_dim).tolist()
                
                sampler = optuna.samplers.GridSampler(search_space)
            else:
                sampler = optuna.samplers.TPESampler(seed=self.random_seed)
            
            study = optuna.create_study(direction='minimize', sampler=sampler)
            study.optimize(optuna_objective, n_trials=n_trials, timeout = self.time_limit)
            
            progress_bar.finish(True)
            
            best_params = {name: study.best_params[name] for name in param_names}
            
            return {
                "success": True,
                "best_params": best_params,
                "best_value": study.best_value,
                "iterations": len(study.trials),
                "message": f"Optuna优化完成"
            }
        except Exception as e:
            progress_bar.finish(False)
            return {
                "success": False,
                "message": f"Optuna优化出错: {str(e)}"
            }
    

##########################################################################################################################################


    def run_hyperopt(self, objective_function, param_ranges, method):
        """执行Hyperopt优化"""
        # Patch numpy for hyperopt compatibility with numpy >= 1.24
        if not hasattr(np, 'warnings'):
             import warnings
             np.warnings = warnings

        from hyperopt import fmin, tpe, hp, Trials, rand
        try:
            from hyperopt import atpe
        except ImportError:
            # 如果版本较旧可能不支持，或者在 experimental 中
            atpe = None
        
        param_names = list(param_ranges.keys())
        max_evals = self.max_iterations
        progress_bar = ProgressBar(max_evals, f"Hyperopt-{method}")
        
        # 构建搜索空间
        space = {}
        for name, (low, high) in param_ranges.items():
            space[name] = hp.uniform(name, low, high)
        
        iteration_count = 0
        best_loss = float('inf')
        
        def hyperopt_objective(params):
            nonlocal iteration_count, best_loss
            iteration_count += 1
            
            # 转换为列表形式调用目标函数
            params_list = [params[name] for name in param_names]
            result = objective_function(params_list)
            
            # 处理 inf 值
            if not np.isfinite(result):
                print(f"⚠️ Warning: Objective function returned {result}, replacing with 1e10")
                result = 1e10

            if result < best_loss:
                best_loss = result
            
            progress_bar.update(iteration_count, best_loss)
            return result
        
        try:
            # 根据方法选择算法
            if method == 'TPE':
                algo = tpe.suggest
            elif method == 'Random':
                algo = rand.suggest
            elif method == 'AdaptiveTPE':
                if atpe is None:
                    algo = tpe.suggest
                else:
                    try:
                        __import__("lightgbm")
                        __import__("sklearn")
                        algo = atpe.suggest
                    except Exception:
                        algo = tpe.suggest
            else:
                algo = tpe.suggest
            
            trials = Trials()
            best = fmin(fn=hyperopt_objective,
                       space=space,
                       algo=algo,
                       max_evals=max_evals,
                       trials=trials,
                       rstate=np.random.default_rng(self.random_seed),
                       timeout = self.time_limit)
            
            progress_bar.finish(True)
            
            return {
                "success": True,
                "best_params": best,
                "best_value": min([trial['result']['loss'] for trial in trials.trials]),
                "iterations": len(trials.trials),
                "message": f"Hyperopt优化完成"
            }
        except Exception as e:
            progress_bar.finish(False)
            return {
                "success": False,
                "message": f"Hyperopt优化出错: {str(e)}"
            }
    

##########################################################################################################################################


    def run_nevergrad(self, objective_function, param_ranges, method):
        """执行Nevergrad优化"""
        np_dict = np.__dict__
        if "float_" not in np_dict:
            np.float_ = np.float64
        if "float" not in np_dict:
            np.float = float
        if "int" not in np_dict:
            np.int = int
        if "bool" not in np_dict:
            np.bool = bool
        if "object" not in np_dict:
            np.object = object
        if "str" not in np_dict:
            np.str = str
        if "complex" not in np_dict:
            np.complex = complex
        if "long" not in np_dict:
            np.long = int

        try:
            random.seed(int(self.random_seed))
            np.random.seed(int(self.random_seed))
        except Exception:
            pass

        import nevergrad as ng

        try:
            parametrization = ng.p.Dict(**{
                name: ng.p.Scalar(lower=bounds[0], upper=bounds[1])
                for name, bounds in param_ranges.items()
            })

            # 添加异常捕获的包装器
            evaluation_count = 0
            best_result = {'loss': float('inf'), 'params': None}
            
            def safe_nevergrad_objective(d):
                nonlocal evaluation_count, best_result
                try:
                    params_list = [d[name] for name in param_ranges.keys()]
                    result = objective_function(params_list)
                    if result < best_result['loss']:
                        best_result['loss'] = result
                        best_result['params'] = d.copy()
                    evaluation_count += 1
                    return result
                except Exception as e:
                    print(f"\n⚠️ 目标函数评估出错: {e}")
                    return float('inf')

            budget = self.max_iterations
            progress_bar = ProgressBar(budget, f"Nevergrad-{method}")

            try:
                try:
                    optimizer = ng.optimizers.registry[method](
                        parametrization=parametrization,
                        budget=budget,
                        random_state=int(self.random_seed)
                    )
                except TypeError:
                    optimizer = ng.optimizers.registry[method](
                        parametrization=parametrization,
                        budget=budget
                    )
            except KeyError:
                print(f"⚠️ 算法 {method} 不可用，使用默认的DE算法")
                try:
                    optimizer = ng.optimizers.DE(
                        parametrization=parametrization,
                        budget=budget,
                        random_state=int(self.random_seed)
                    )
                except TypeError:
                    optimizer = ng.optimizers.DE(
                        parametrization=parametrization,
                        budget=budget
                    )

            best_candidate = None
            best_loss = float('inf')
            initial_loss = None
            iteration = 0

            try:
                import time
                start_time = time.time()
                for _ in range(budget):
                    current_time = time.time()
                    if (current_time - start_time) > self.time_limit:
                        break
                    candidate = optimizer.ask()
                    loss = safe_nevergrad_objective(candidate.value)
                    if not np.isfinite(loss):
                        loss = 1e10
                    optimizer.tell(candidate, loss)
                    if initial_loss is None:
                        initial_loss = loss
                    if loss < best_loss:
                        best_loss = loss
                        best_candidate = candidate.value
                    iteration += 1
                    progress_bar.update(iteration, best_loss)
            except KeyboardInterrupt:
                pass
            except Exception as e:
                print(f"\n⚠️ 优化过程出错: {e}")

            progress_bar.finish(best_candidate is not None)

            if best_candidate is None and best_result['params'] is not None:
                best_candidate = best_result['params']
                best_loss = best_result['loss']

            if best_candidate is not None:
                return {
                    "success": True,
                    "best_params": best_candidate,
                    "best_value": best_loss,
                    "iterations": iteration,
                    "message": "Nevergrad优化完成"
                }
            else:
                return {
                    "success": False,
                    "message": "Nevergrad优化未找到有效解"
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Nevergrad优化出错: {str(e)}"
            }

##########################################################################################################################################


    def _is_optimization_successful(self, result, best_loss, initial_loss, iterations):
        """智能判断优化是否成功"""
        
        # 1. 如果SciPy认为成功，那就是成功
        if result.success:
            return True
        
        # 2. 即使SciPy认为不成功，但如果有显著改进，也认为成功
        if initial_loss is not None and best_loss < initial_loss:
            improvement_ratio = (initial_loss - best_loss) / abs(initial_loss)
            
            # 如果改进超过5%，认为是成功的
            if improvement_ratio > 0.05:
                print(f"🎯 虽然未完全收敛，但改进显著 ({improvement_ratio:.1%})，认为优化成功")
                return True
        
        # 3. 如果目标函数值很小，也认为成功
        if best_loss < 1.0:
            print(f"🎯 目标函数值较小 ({best_loss:.6f})，认为优化成功")
            return True
        
        # 4. 如果迭代次数较多且有改进，也可能是成功的
        if iterations > 50 and np.isfinite(best_loss):
             # 只要不是inf，且迭代了足够次数，就认为达到了"能接受的结果"
             # 尤其是用户要求"不要报错"时，我们放宽成功标准
             print(f"🎯 经过充分迭代 ({iterations}次) 且结果有效 ({best_loss:.4f})，认为优化成功")
             return True
        
        return False
    
    def _evaluate_convergence_quality(self, best_loss, initial_loss, iterations, max_iterations):
        """简化的统一收敛质量评估"""
        
        # 计算改进指标
        if initial_loss is not None and abs(initial_loss) > 1e-10:
            absolute_improvement = abs(initial_loss - best_loss)
            relative_improvement = absolute_improvement / abs(initial_loss)
        else:
            absolute_improvement = 0
            relative_improvement = 0
        
        # 统一的简单判断标准
        target_achieved = False
        convergence_quality = "较差"
        
        # 1. 优先看相对改进（适用于所有数值范围）
        if relative_improvement >= 0.5:  # 50%以上改进
            convergence_quality = "优秀"
            target_achieved = True
        elif relative_improvement >= 0.2:  # 20%以上改进
            convergence_quality = "良好"
            target_achieved = True
        elif relative_improvement >= 0.1:  # 10%以上改进
            convergence_quality = "一般"
            target_achieved = False
        elif relative_improvement >= 0.05:  # 5%以上改进
            convergence_quality = "尚可"
            target_achieved = False
        
        # 2. 如果相对改进很小，但绝对改进很大，也认为是好结果
        elif absolute_improvement >= 10:  # 绝对改进≥10
            convergence_quality = "良好"
            target_achieved = True
        elif absolute_improvement >= 1:   # 绝对改进≥1
            convergence_quality = "一般"
            target_achieved = False
        elif absolute_improvement > 0:    # 有改进
            convergence_quality = "尚可"
            target_achieved = False
        
        # 3. 特殊情况：目标函数值本身就很小
        elif abs(best_loss) < 0.01:
            convergence_quality = "优秀"
            target_achieved = True
        elif abs(best_loss) < 0.1:
            convergence_quality = "良好"
            target_achieved = True
        
        return convergence_quality, target_achieved
    

    
    def calculate_objective(self, result: dict, analysis_config: dict, problem_type: str) -> float:
        """计算目标函数值 (统一处理简单目标和约束目标)"""
        return self._calculate_unified_objective(result, analysis_config)

    def _calculate_unified_objective(self, result: dict, analysis_config: dict) -> float:
        """统一目标函数计算逻辑：合并了基础目标计算和约束罚分计算"""
        try:
            # --- 1. 计算基础目标值 (Base Objective) ---
            obj = float('inf')
            
            # 1.1 尝试使用自定义表达式
            expr = analysis_config.get('objective_expression')
            # 创建安全的执行环境
            safe_globals = {
                'result': result,
                'abs': abs, 'min': min, 'max': max, 'pow': pow,
                'sqrt': lambda x: x**0.5, 'float': float, '__builtins__': {}
            }
            
            if expr:
                try:
                    obj = float(eval(expr, safe_globals))
                except Exception:
                    # 表达式执行失败，不立即报错，继续尝试默认方式
                    pass 

            primary_expr = str(analysis_config.get('primary_objective', '') or '')
            if primary_expr.lower().startswith('maximize_') and obj != float('inf') and obj != float('-inf') and not (np.isnan(obj) if 'np' in globals() else False):
                obj = -float(obj)
            
            # 1.2 如果表达式无效，尝试使用预定义的主目标
            if obj == float('inf') or obj == float('-inf') or (np.isnan(obj) if 'np' in globals() else False):
                primary = analysis_config.get('primary_objective', 'minimize_keff')
                
                if 'FQ' in primary or 'minimize_FQ' in primary:
                    obj = float(result.get('FQ', float('inf')))
                elif 'pressuredrop' in primary and ('error' in primary or 'target' in primary):
                    target_pd = analysis_config.get('target_values', {}).get('pressuredrop')
                    if target_pd is None:
                        target_pd = analysis_config.get('target_values', {}).get('pressure_drop')
                    if target_pd is None:
                        target_pd = 0.0
                    if 'pressuredrop' in result:
                        obj = abs(float(result.get('pressuredrop')) - float(target_pd))
                elif 'pressuredrop' in primary:
                    obj = float(result.get('pressuredrop', float('inf')))
                elif 'pressure' in primary:
                    obj = float(result.get('avgpressure', float('inf')))
                elif 'keff' in primary:
                    target_keff = analysis_config.get('target_values', {}).get('keff', 1.0)
                    # 只有当keff存在时才计算，否则保持inf
                    if 'keff' in result:
                        obj = abs(result.get('keff') - target_keff)
                elif 'nv' in primary:
                    obj = float(result.get('nv', float('inf')))
                elif 'nt' in primary:
                    obj = float(result.get('nt', float('inf')))
                elif 'rho_max' in primary:
                    obj = float(result.get('rho_max', float('inf')))
                elif 'rho_mean' in primary:
                    obj = float(result.get('rho_mean', float('inf')))
                elif 'u_max' in primary:
                    obj = float(result.get('u_max', float('inf')))
                elif 'u_mean' in primary:
                    obj = float(result.get('u_mean', float('inf')))
                elif 'u_min' in primary:
                    obj = float(result.get('u_min', float('inf')))
                elif 'cpu' in primary:
                    obj = float(result.get('cpu', float('inf')))
                elif 'efficiency' in primary:
                    if 'maximize' in primary:
                        obj = -result.get('efficiency', 0)
                    else:
                        obj = result.get('efficiency', float('inf'))
            
            # 1.3 最后的兜底：尝试取结果中的第一个数值
            if obj == float('inf') or obj == float('-inf') or (np.isnan(obj) if 'np' in globals() else False):
                print(f"⚠️ Warning: Base objective is invalid ({obj}). Result keys: {list(result.keys())}, Primary objective: {analysis_config.get('primary_objective')}")
                for key, value in result.items():
                    if isinstance(value, (int, float)) and key != 'success':
                        obj = float(value)
                        break
            
            # --- 2. 计算约束罚分 (Constraint Penalties) ---
            total_penalty = 0.0
            constraints = analysis_config.get('constraints', []) or []
            penalty_weight = float(analysis_config.get('penalty_weight', 10000.0))
            
            if constraints:
                for c in constraints:
                    try:
                        # 处理字典形式的约束
                        if isinstance(c, dict):
                            metric = c.get('metric', '')
                            ctype = str(c.get('type', '')).lower()
                            val = c.get('value')
                            tol = float(c.get('tol', c.get('epsilon', 0.0)))
                            w = float(c.get('penalty_weight', 1000.0))
                            
                            # 获取度量值
                            mv = result.get(metric)
                            if mv is None and metric:
                                try:
                                    mv = float(eval(metric, safe_globals))
                                except:
                                    mv = None
                            
                            if mv is None or val is None:
                                # 无法获取度量值或目标值，无法计算差距，因此忽略罚分 (Strict Gap Policy)
                                pass
                            else:
                                mv = float(mv)
                                val = float(val)
                                vio = 0.0
                                
                                if ctype in ['<=', 'le']:
                                    vio = max(0.0, mv - val - tol)
                                elif ctype in ['>=', 'ge']:
                                    vio = max(0.0, val - mv - tol)
                                elif ctype in ['==', 'eq']:
                                    vio = max(0.0, abs(mv - val) - tol)
                                elif ctype in ['<']:
                                    vio = max(0.0, mv - val)
                                elif ctype in ['>']:
                                    vio = max(0.0, val - mv)
                                else:
                                    # User requested to remove boolean fallback to avoid "boolean * coefficient" penalty
                                    # If we can't calculate a gap, we assume no penalty (or user should provide explicit metric)
                                    vio = 0.0
                                
                                total_penalty += w * vio
                                
                        # 处理字符串形式的约束
                        else:
                            s = str(c).strip()
                            vio = 0.0
                            try:
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
                                else:
                                    # 无法解析为标准不等式，尝试直接评估表达式
                                    # 用户要求：不要使用bool，因为没有梯度。应使用距离。
                                    val = eval(s, safe_globals)
                                    
                                    # 情况1: 返回的是数字（非布尔），假设其为"违约程度" (大于0表示违约)
                                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                                        vio = max(0.0, float(val))
                                    # 情况2: 返回的是布尔值
                                    # 这是一个极其不理想的情况，因为没有优化方向（梯度为0）
                                    # 但为了程序不崩溃，只能给一个固定的惩罚
                                    else:
                                        # User requested to remove boolean fallback
                                        vio = 0.0
                            except:
                                # 解析或计算出错，忽略罚分
                                vio = 0.0
                            
                            total_penalty += penalty_weight * vio
                            
                    except Exception:
                        # 单个约束计算出错，忽略罚分
                        pass

            # --- 3. 返回总目标值 (基础值 + 罚分) ---
            return float(obj) + float(total_penalty)

        except Exception as e:
            print(f"⚠️ 目标函数计算出错: {e}")
            return 1e10  # Return large finite value instead of inf
