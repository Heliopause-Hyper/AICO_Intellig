"""
问题检测器模块
使用LLM分析用户输入，识别问题类型并构建复杂目标函数
"""
import json
import re
import os
import sys
from typing import Dict, List, Any, Optional  # 添加这行导入

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

class ProblemDetector:
    """问题检测和分析器"""
    
    def __init__(self, client):
        self.client = client
    
    def detect_problem_type(self, user_input: str, problem_config) -> str:
        """检测问题类型：覆盖全部 ProblemConfig.problem_types；支持显式指定/直接提到类型名或模板名。"""
        user_input_lower = (user_input or "").lower()
        problem_types = getattr(problem_config, "problem_types", {}) or {}
        if not problem_types:
            return "iaea"

        alias_to_type = {}
        for ptype, cfg in problem_types.items():
            if not ptype:
                continue
            alias_to_type[str(ptype).lower()] = ptype
            try:
                tname = str((cfg or {}).get("template_name") or "").strip()
            except Exception:
                tname = ""
            if tname:
                alias_to_type[tname.lower()] = ptype
            try:
                display = str((cfg or {}).get("name") or "").strip()
            except Exception:
                display = ""
            if display:
                alias_to_type[display.lower()] = ptype

        m = re.search(r"(?:^|\\b)problem_type\\s*[:=]\\s*([a-zA-Z0-9_]+)\\b", user_input_lower)
        if m:
            key = m.group(1).lower()
            if key in alias_to_type:
                detected = alias_to_type[key]
                print(f"规则检测结果: {detected}")
                return detected

        for alias_l, ptype in alias_to_type.items():
            if not alias_l:
                continue
            if re.search(rf"(?:^|\\b){re.escape(alias_l)}(?:\\b|$)", user_input_lower):
                print(f"规则检测结果: {ptype}")
                return ptype
        for alias_l, ptype in alias_to_type.items():
            if alias_l and alias_l in user_input_lower:
                print(f"规则检测结果: {ptype}")
                return ptype

        keyword_alias = {
            "iaea": ['中子扩散', 'sigmaa', 'neutron', 'diffusion', 'reactor', '反应堆', 'd11', 'd21'],
            "thmf": ['传热', '翅片', 'thermal', 'fins', 'heat', 'transfer', 'efficiency', '效率', 'bi', 'k1', 'k2', 'k3', 'k4', 'k5'],
            "sns": ['对流', '流体', 'convection', 'flow', 'pressure', '压力', 'umax', 'mu', 'fluid', '压力降', 'pressuredrop'],
            "corca_state": ['corca', '硼', 'boron', 'ppm', '棒位', 'rod', '临界硼', 'critical', 'keff', 'fq', 'fdh'],
            "ex_poisson_adapt_indicator": [
                "poisson",
                "自适应",
                "网格",
                "adapt",
                "indicator",
                "anisomax",
                "err0",
                "niter",
                "nv",
                "rho",
                "rho_mean",
                "rho_max",
            ],
        }

        scores = {}
        for alias, words in keyword_alias.items():
            score = sum(1 for word in words if str(word).lower() in user_input_lower)
            if score <= 0:
                continue
            mapped = None
            try:
                mapped = problem_config.get_problem_type_by_template(alias)
            except Exception:
                mapped = None
            if mapped and mapped in problem_types:
                scores[mapped] = max(scores.get(mapped, 0), score)
            elif alias in problem_types:
                scores[alias] = max(scores.get(alias, 0), score)

        if scores:
            detected = max(scores, key=scores.get)
            print(f"规则检测结果: {detected}")
            return detected

        return next(iter(problem_types.keys()))
    
#     def smart_select_optimizer_and_method(self, problem_type: str, user_input: str, 
#                                         opt_params: List[str], problem_config) -> Dict[str, Any]:
#         """智能选择优化器和方法，参考历史数据"""
        
#         # 获取历史洞察
#         try:
#             from .optimization_history import OptimizationHistory
#             history_db = OptimizationHistory()
#             historical_context = history_db.generate_llm_context(problem_type, opt_params)
#         except Exception as e:
#             print(f"⚠️ 历史数据获取失败: {e}")
#             historical_context = "无历史数据可参考"
        
#         # 获取可用的优化算法
#         try:
#             from .problem_config import get_optimizer_config
#         except ImportError:
#             sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#             from problem_config import get_optimizer_config
        
#         optimizer_config = get_optimizer_config()
        
#         # 构建所有可用算法列表
#         all_algorithms = []
#         for optimizer_name, config in optimizer_config.items():
#             for method in config['methods']:
#                 all_algorithms.append(f"{optimizer_name}-{method}")
        
#         # 构建包含历史信息的prompt
#         prompt = f"""你是优化算法专家。请根据问题特征和历史经验选择最合适的优化算法。

# {historical_context}

# 当前问题:
# - 问题类型: {problem_type}
# - 用户描述: {user_input}
# - 优化参数: {opt_params}
# - 参数数量: {len(opt_params)}

# 可选算法: {', '.join(all_algorithms)}

# 请综合考虑历史成功经验，各优化库和算法自身特点和当前问题特点，选择最合适的算法。要求同时考虑算法效率（收敛速度）和算法精度。

# 返回JSON格式:
# {{
#     "optimizer": "推荐的优化器",
#     "method": "推荐的方法",
#     "reason": "选择理由(包含历史经验参考)"
# }}
# """
        
#         try:
#             response = self.client.chat.completions.create(
#                 model="deepseek-chat",
#                 messages=[
#                     {"role": "system", "content": "你是优化算法专家，根据问题特征和历史经验智能选择算法。"},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.1,
#                 max_tokens=500
#             )
            
#             response_content = response.choices[0].message.content.strip()
#             print(f"LLM算法选择原始回复: {response_content[:200]}...")
            
#             # 清理响应内容，提取JSON部分
#             if '```json' in response_content:
#                 json_start = response_content.find('```json') + 7
#                 json_end = response_content.find('```', json_start)
#                 response_content = response_content[json_start:json_end].strip()
#             elif '```' in response_content:
#                 json_start = response_content.find('```') + 3
#                 json_end = response_content.find('```', json_start)
#                 response_content = response_content[json_start:json_end].strip()
#             elif response_content.startswith('{') and response_content.endswith('}'):
#                 pass
#             else:
#                 json_start = response_content.find('{')
#                 json_end = response_content.rfind('}') + 1
#                 if json_start >= 0 and json_end > json_start:
#                     response_content = response_content[json_start:json_end]
            
#             try:
#                 result = json.loads(response_content)
                
#                 # 验证算法是否可用
#                 algorithm_key = f"{result['optimizer']}-{result['method']}"
#                 if algorithm_key not in all_algorithms:
#                     # 回退到默认算法
#                     result = {
#                         "optimizer": "SciPy",
#                         "method": "Nelder-Mead",
#                         "reason": f"推荐算法不可用，使用默认算法"
#                     }
                
#                 print(f"智能选择结果: {result['optimizer']} - {result['method']}")
#                 print(f"选择理由: {result['reason']}")
#                 return result
                
#             except json.JSONDecodeError as e:
#                 print(f"算法选择失败: JSON解析错误 - {e}")
                
#         except Exception as e:
#             print(f"算法选择失败: {e}")
        
#         # 默认回退
#         return {
#             "optimizer": "SciPy",
#             "method": "Nelder-Mead",
#             "reason": "LLM选择失败，使用默认算法"
#         }
    
    def llm_analyse(self, user_input: str, problem_config, problem_type: str = None) -> str:
        """增强的用户输入分析，支持复杂目标函数构建和算法指定"""
        # 如果没有提供问题类型，则检测
        if problem_type is None:
            problem_type = self.detect_problem_type(user_input, problem_config)

        mapped_problem_type = None
        try:
            mapped_problem_type = problem_config.get_problem_type_by_template(problem_type)
        except Exception:
            mapped_problem_type = None

        if mapped_problem_type and mapped_problem_type in getattr(problem_config, 'problem_types', {}):
            problem_type_for_config = mapped_problem_type
        else:
            problem_type_for_config = problem_type
        
        # 获取对应的模板名称
        template_name = problem_config.get_template_by_problem_type(problem_type_for_config)
        
        # 从 ProblemConfig 获取参数/输出定义（对 FreeFEM/CORCA 统一）
        template_params = problem_config.get_problem_config(problem_type_for_config).get('parameters', {})
        template_outputs = problem_config.get_problem_config(problem_type_for_config).get('objectives', [])
        input_file_content = self._read_template_input_file(problem_type, template_name)
        output_file_content = self._read_template_output_file(problem_type, template_name)
        
        # 获取历史优化数据
        try:
            try:
                from .optimization_history import OptimizationHistory
            except ImportError:
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from optimization_history import OptimizationHistory
            
            history_db = OptimizationHistory()
            # 由于此时还未提取具体参数，我们传入空列表获取通用建议
            historical_context = history_db.generate_llm_context(problem_type, [])
        except Exception as e:
            print(f"⚠️ 历史数据获取失败: {e}")
            historical_context = "无历史数据可参考"

        # 获取可用的优化算法
        try:
            from .problem_config import get_optimizer_config
        except ImportError:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from problem_config import get_optimizer_config
        
        optimizer_config = get_optimizer_config()
        
        # 构建所有可用算法列表
        all_algorithms = []
        for optimizer_name, config in optimizer_config.items():
            for method in config['methods']:
                all_algorithms.append(f"{optimizer_name}-{method}")
        
        # 构建增强的prompt，包含算法识别
        prompt = f"""请分析以下用户优化需求，提取详细的优化信息：

参考历史数据：
{historical_context}

用户输入: {user_input}

问题类型: {problem_type}
模板名称: {template_name}

可用参数及默认值:
{input_file_content[:700]}...

可用输出指标:
{output_file_content[:400]}...

可用优化算法:
{', '.join(all_algorithms)}

请从用户输入中提取以下信息，注意参数名称的大小写要与可用参数匹配：
1. 目标类型 (simple/constrained)
   - 只有明确有"使得...在...范围内"或"约束..."时才选constrained
   - 如果是constrained，主目标应该是最终要最小化的指标，而不是约束条件本身
2. 主要目标 (如 minimize_keff_error, maximize_efficiency, minimize_pressuredrop, minimize_FQ 等)
3. 优化参数列表（请使用准确的参数名称，注意大小写）
   - 只包含用户明确提到或强相关的参数，不要把“所有可用参数”都列进去
4. 目标函数的Python表达式 (objective_expression)
   - 这里的表达式仅计算主目标的值（例如 "result.get('FQ', float('inf'))"）
   - 不要在这里包含约束条件的惩罚逻辑！约束逻辑会由系统单独处理。
5. 约束条件 (constraints)
   - 列表格式，每个约束是一个对象：{{"metric": "指标名", "type": ">=/<=/==", "value": 阈值}}
   - 例如: [{{"metric": "keff", "type": ">=", "value": 1.033}}, {{"metric": "keff", "type": "<=", "value": 1.035}}]
6. 目标数值
7. 用户指定的优化算法（如果有）
8. 推荐的优化器和方法（如果用户未指定）
9. 优化器选择的理由以及算法选择的理由（algorithm_reason）

参数名称匹配规则：
- 如果用户说"umax"，应该匹配为"uMax"
- 如果用户说"k1,k2,k3"，应该匹配为"k1","k2","k3"
- 请根据可用参数列表选择正确的参数名称

算法选择要求：
请综合考虑历史成功经验，四种优化库（SciPy，Hyperopt，Optuna，Nevergrad）和算法自身特点和当前问题特点，选择最合适的算法。
要求考虑算法效率（收敛速度）和算法精度。
注意！！！不考虑算法成熟性或是否传统，也不考虑历史案例的时间先后，尤其是不要受到最近几次优化的影响，只考虑统计数据。

返回JSON格式示例（IAEA约束优化）：
{{
    "objective_type": "constrained",
    "primary_objective": "minimize_FQ",
    "opt_params": ["Sigmaa11", "D11"],
    "objective_expression": "result.get('FQ', float('inf'))",
    "constraints": [
        {{"metric": "keff", "type": ">=", "value": 1.033}},
        {{"metric": "keff", "type": "<=", "value": 1.035}}
    ],
    "target_values": {{"keff": 1.034}},
    "user_specified_algorithm": null,
    "optimizer": "SciPy",
    "method": "COBYLA",
    "algorithm_reason": "..."
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是优化问题分析专家，能够理解复杂的优化需求并识别用户指定的算法。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            response_content = response.choices[0].message.content.strip()
            if str(os.getenv("AICO_LLM_DEBUG", "")).strip().lower() in {"1", "true", "yes", "y"}:
                print(f"LLM原始回复: {response_content[:200]}...")
            
            # 清理响应内容，提取JSON部分
            if '```json' in response_content:
                json_start = response_content.find('```json') + 7
                json_end = response_content.find('```', json_start)
                response_content = response_content[json_start:json_end].strip()
            elif '```' in response_content:
                json_start = response_content.find('```') + 3
                json_end = response_content.find('```', json_start)
                response_content = response_content[json_start:json_end].strip()
            elif response_content.startswith('{') and response_content.endswith('}'):
                # 已经是JSON格式
                pass
            else:
                # 尝试找到JSON部分
                json_start = response_content.find('{')
                json_end = response_content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    response_content = response_content[json_start:json_end]
                else:
                    raise ValueError("无法找到有效的JSON内容")
            
            # 验证JSON格式
            try:
                cfg = json.loads(response_content)
                cfg = self._sanitize_analysis_config(cfg, problem_type, template_outputs)
                return json.dumps(cfg, ensure_ascii=False)
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}")
                print(f"清理后的内容: {response_content}")
                raise
            
        except Exception as e:
            print(f"LLM分析失败，使用默认配置: {e}")
            # 返回默认配置，针对用户输入进行简单解析
            default_config = {
                "objective_type": "constrained",
                "primary_objective": "minimize_keff_error",
                "opt_params": ["Sigmaa11", "Sigmaa12", "Sigmaa13", "Sigmaa21", "Sigmaa22", "Sigmaa23"],
                "objective_expression": "abs(result['keff'] - 1.012) * 1000",
                "constraints": [{"metric": "keff", "type": "equal", "value": 1.012}],
                "target_values": {"keff": 1.012},
                "user_specified_algorithm": "Nevergrad-OnePlusOne" if "oneplusone" in user_input.lower() else None,
                "optimizer": "Nevergrad" if "oneplusone" in user_input.lower() else "SciPy",
                "method": "OnePlusOne" if "oneplusone" in user_input.lower() else "Nelder-Mead",
                "algorithm_reason": "LLM分析失败，使用默认算法"
            }
            return json.dumps(default_config)

    def _sanitize_analysis_config(self, cfg: dict, problem_type: str, template_outputs: list) -> dict:
        if not isinstance(cfg, dict):
            return cfg

        outputs = [str(x) for x in (template_outputs or [])]
        outputs_lower = {x.lower() for x in outputs}

        primary = str(cfg.get("primary_objective") or "")
        primary_lower = primary.lower()
        expr = str(cfg.get("objective_expression") or "")
        expr_lower = expr.lower()

        def drop_expr():
            if "objective_expression" in cfg:
                cfg.pop("objective_expression", None)

        def parse_primary(p: str):
            pl = p.lower().strip()
            direction = None
            metric = None
            if pl.startswith("maximize_"):
                direction = "maximize"
                metric = pl[len("maximize_"):]
            elif pl.startswith("minimize_"):
                direction = "minimize"
                metric = pl[len("minimize_"):]
            if metric:
                metric = metric.replace("-", "_").strip()
                if metric.endswith("_error"):
                    metric = metric[: -len("_error")]
            return direction, metric

        def extract_expr_metrics(e: str):
            metrics = set()
            try:
                for m in re.findall(r"result\.get\(\s*['\"]([^'\"]+)['\"]", e):
                    metrics.add(str(m).strip())
                for m in re.findall(r"result\[\s*['\"]([^'\"]+)['\"]\s*\]", e):
                    metrics.add(str(m).strip())
            except Exception:
                pass
            return metrics

        direction, metric = parse_primary(primary)
        if metric and (metric not in outputs_lower):
            if outputs:
                preferred = None
                for cand in ["efficiency", "pressuredrop", "avgpressure", "keff", "fq"]:
                    if cand in outputs_lower:
                        preferred = cand
                        break
                if preferred is None:
                    preferred = outputs[0].lower()
                if preferred == "keff" and direction == "minimize":
                    cfg["primary_objective"] = "minimize_keff_error"
                else:
                    if direction in ["maximize", "minimize"]:
                        cfg["primary_objective"] = f"{direction}_{preferred}"
                    else:
                        cfg["primary_objective"] = f"minimize_{preferred}"
                drop_expr()

        expr_metrics = {m.lower() for m in extract_expr_metrics(expr) if m}
        if expr_metrics and any((m not in outputs_lower) for m in expr_metrics):
            drop_expr()

        expr2 = str(cfg.get("objective_expression") or "").strip()
        primary2 = str(cfg.get("primary_objective") or "")
        primary2_lower = primary2.lower()
        if not expr2 and ("_error" not in primary2_lower):
            d2, m2 = parse_primary(primary2)
            if m2 and (m2.lower() in outputs_lower):
                cfg["objective_expression"] = f"result.get('{m2}', float('inf'))"

        return cfg
    
    def _read_template_input_file(self, problem_type: str, template_name: str) -> str:
        """读取模板输入文件内容"""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if problem_type == 'corca_state':
            config_path = os.path.join(repo_root, "templates", "corcasim", f"{template_name}.json")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return f.read()
            return "CORCA-sim 模板缺失，建议在 templates/corcasim/<name>.json 中定义参数与输出。"

        input_file = os.path.join(repo_root, "templates", f"{template_name}_input.txt")
        if os.path.exists(input_file):
            with open(input_file, 'r', encoding='utf-8') as f:
                return f.read()
        return "无输入文件信息"
    
    def _read_template_output_file(self, problem_type: str, template_name: str) -> str:
        """读取模板输出文件内容"""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if problem_type == 'corca_state':
            return "keff, FQ, FDH, AO, deltai, bore, burnup"

        output_file = os.path.join(repo_root, "templates", f"{template_name}_output.txt")
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                return f.read()
        return "无输出文件信息"
