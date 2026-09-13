import os
import sys
import json
from datetime import datetime

# 处理导入路径
try:
    from .optimizer import DeepFreeFEMOptimizer
    from .optimization_history import OptimizationHistory
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from optimizer import DeepFreeFEMOptimizer
    from optimization_history import OptimizationHistory


def show_history(limit=500):
    # 显示最近的优化历史记录
    try:
        history = OptimizationHistory()
        conn = history._init_database()
        import sqlite3
        conn = sqlite3.connect(history.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, timestamp, problem_type, optimizer, method, best_value, success, execution_time
            FROM optimization_records 
            ORDER BY timestamp DESC LIMIT ?
        ''', (limit,))
        
        records = cursor.fetchall()
        conn.close()
        
        if not records:
            print("\n📭 暂无历史记录")
            return
            
        print(f"\n📚 最近 {len(records)} 条优化记录:")
        print("="*100)
        print(f"{'ID':<5} {'时间':<20} {'类型':<10} {'算法':<25} {'结果':<15} {'耗时(s)':<10} {'状态':<10}")
        print("-"*100)
        
        for row in records:
            try:
                id_, timestamp, p_type, opt, method, val, success, time_ = row
                
                # 确保所有字段都是字符串
                def safe_str(s):
                    if isinstance(s, bytes):
                        return s.decode('utf-8', errors='ignore')
                    return str(s)

                # 格式化时间
                try:
                    ts_str = safe_str(timestamp)
                    dt = datetime.fromisoformat(ts_str)
                    time_str = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    time_str = safe_str(timestamp)[:16]
                
                # 处理其他字段
                p_type = safe_str(p_type)
                opt = safe_str(opt)
                method = safe_str(method)
                
                status = "✅ 成功" if success else "❌ 失败"
                best_val = f"{val:.4e}" if val is not None else "N/A"
                alg = f"{opt}-{method}"
                
                # 格式化耗时
                try:
                    if time_ is not None:
                        time_str_val = f"{float(time_):.1f}"
                    else:
                        time_str_val = "N/A"
                except:
                    time_str_val = safe_str(time_)

                print(f"{id_:<5} {time_str:<20} {p_type:<10} {alg:<25} {best_val:<15} {time_str_val:<10} {status:<10}")
            except Exception as row_e:
                print(f"{row[0]:<5} <数据解析错误: {row_e}>")
                continue
        print("="*100)
        print("\n")
        
    except Exception as e:
        print(f"❌ 获取历史记录失败: {e}")


def handle_history_command(command_parts):
    """处理历史记录相关命令"""
    if not command_parts:
        return
        
    action = command_parts[0].lower()
    history = OptimizationHistory()
    
    if action in ['list', 'ls', 'show']:
        show_history()
        
    elif action == 'delete':
        if len(command_parts) < 2:
            print("⚠️ 请指定要删除的记录ID (支持格式: 123 或 1,2,3 或 1-5)")
            return
        
        ids_to_delete = set()
        raw_args = " ".join(command_parts[1:])  # 合并所有参数处理
        
        # 处理逗号分隔
        parts = raw_args.replace(',', ' ').split()
        
        for part in parts:
            try:
                if '-' in part:
                    # 处理范围 1-5
                    start, end = map(int, part.split('-'))
                    ids_to_delete.update(range(start, end + 1))
                else:
                    # 单个ID
                    ids_to_delete.add(int(part))
            except ValueError:
                print(f"⚠️ 跳过无效格式: {part}")
                continue
                
        if not ids_to_delete:
            print("❌ 未识别到有效的ID")
            return
            
        count = history.delete_records(list(ids_to_delete))
        if count > 0:
            print(f"✅ 成功删除 {count} 条记录")
        else:
            print(f"⚠️ 未找到可删除的记录 (尝试删除ID: {sorted(list(ids_to_delete))})")
            
    elif action == 'note':
        if len(command_parts) < 3:
            print("⚠️ 请指定ID和备注内容 (例如: note 123 测试数据)")
            return
        try:
            record_id = int(command_parts[1])
            note = " ".join(command_parts[2:])
            if history.update_record_note(record_id, note):
                print(f"✅ 记录 {record_id} 备注已更新")
            else:
                print(f"❌ 更新失败：记录 {record_id} 不存在")
        except ValueError:
            print("❌ ID必须是整数")
            
    elif action == 'status':
        if len(command_parts) < 3:
            print("⚠️ 请指定ID和状态 (例如: status 123 1 表示成功)")
            return
        try:
            record_id = int(command_parts[1])
            status_val = command_parts[2].lower()
            success = status_val in ['1', 'true', 'yes', 'success', 'ok']
            if history.update_record_status(record_id, success):
                print(f"✅ 记录 {record_id} 状态已更新为 {'成功' if success else '失败'}")
            else:
                print(f"❌ 更新失败：记录 {record_id} 不存在")
        except ValueError:
            print("❌ ID必须是整数")
    else:
        print(f"❌ 未知历史命令: {action}")


def main():
    """纯自然语言对话模式的主程序"""
    try:
        print("\n=== AICO 智能优化助手 ===")
        print("欢迎使用AICO优化系统！我可以帮您解决以下类型的优化问题：")
        print("• 中子扩散问题 - 核反应堆参数优化")
        print("• 传热翅片问题 - 热传导效率优化")
        print("• 对流问题 - 流体动力学优化")
        print("\n请直接用自然语言描述您的优化需求，我会自动识别问题类型并选择最合适的优化方法。")
        print("例如：'优化中子扩散参数D11和Sigmaa21，使keff达到1.01'")
        print("支持复杂目标：'优化所有参数使得keff达到1.034时FQ最小'")
        print("\n📚 历史记录管理:")
        print("• history list       - 查看最近记录")
        print("• history delete ID  - 删除指定记录")
        print("• history note ID 文本 - 添加/修改备注")
        print("• history status ID 1/0 - 修改成功状态")
        print("\n输入 'exit' 或 'quit' 退出程序\n")
        
        optimizer = DeepFreeFEMOptimizer()
        
        while True:
            try:
                user_input = input("🤖 请描述您的优化需求: ").strip()
                
                if user_input.lower() in ['exit', 'quit', '退出', '结束']:
                    print("感谢使用AICO优化系统！再见！")
                    break
                
                if not user_input:
                    print("请输入您的优化需求。")
                    continue
                    
                # 检查是否是查看历史记录的命令
                if user_input.lower().startswith('history') or user_input.lower().startswith('历史'):
                    parts = user_input.split()
                    if len(parts) == 1:
                        show_history()
                    else:
                        handle_history_command(parts[1:])
                    continue
                
                if user_input.lower() in ['ls', 'list', 'log']:
                    show_history()
                    continue
                
                print(f"\n📝 正在分析您的需求...")
                
                # 调试信息：显示当前问题配置
                # problem_type = optimizer.problem_detector.detect_problem_type(user_input, optimizer.problem_config)
                # problem_config = optimizer.problem_config.get_problem_config(problem_type)
                # available_params = list(problem_config.get('parameters', {}).keys())
                # print(f"🔍 调试信息:")
                # print(f"   问题类型: {problem_type}")
                # print(f"   可用参数数量: {len(available_params)}")
                # print(f"   前10个参数: {available_params[:10]}")
                # if 'Sigmaa23' in available_params:
                #     print(f"   ✅ Sigmaa23 参数存在")
                # else:
                #     print(f"   ❌ Sigmaa23 参数不存在")
                
                # 使用增强的LLM分析，支持复杂目标函数
                problem_type = optimizer.problem_detector.detect_problem_type(user_input, optimizer.problem_config)
                
                # 传递问题类型给分析方法，避免重复检测
                analysis_result = optimizer.problem_detector.llm_analyse(user_input, optimizer.problem_config, problem_type)
                
                try:
                    analysis_config = json.loads(analysis_result)
                except json.JSONDecodeError as e:
                    print(f"❌ 分析结果解析失败: {e}")
                    print(f"原始结果: {analysis_result[:200]}...")
                    continue
                
                # 确保问题类型一致
                analysis_config['problem_type'] = problem_type
                analysis_config['user_input'] = user_input
                
                def clarify_inputs(acfg):
                    pconf = optimizer.problem_config.get_problem_config(acfg.get('problem_type', ''))
                    available_params = list(pconf.get('parameters', {}).keys())
                    simulator_type = pconf.get('simulator_type', 'freefem')
                    opt_params0 = acfg.get('opt_params', []) or []
                    if simulator_type == 'corca_state' and available_params:
                        ui = (acfg.get("user_input") or "").lower()
                        mention_boron = any(k in ui for k in ["硼", "boron", "ppm", "临界硼", "硼浓度"])
                        mention_rod = any(k in ui for k in ["棒位", "控制棒", "rod", "棒", "插棒", "提棒"])
                        if mention_boron and "bore_ppm" in available_params and "bore_ppm" not in opt_params0:
                            opt_params0 = ["bore_ppm", *opt_params0]
                        if not mention_rod and opt_params0:
                            opt_params0 = [p for p in opt_params0 if not str(p).strip().lower().startswith("pos")]
                        acfg["opt_params"] = opt_params0
                    if available_params and opt_params0:
                        lower_map = {a.lower(): a for a in available_params}
                        new_params = []
                        for p in opt_params0:
                            if p in available_params:
                                new_params.append(p)
                                continue
                            candidates = []
                            lp = p.lower()
                            if lp in lower_map:
                                candidates.append(lower_map[lp])
                            else:
                                for a in available_params:
                                    la = a.lower()
                                    if lp in la or la in lp:
                                        candidates.append(a)
                            print(f"\n⚠️ 未匹配参数: {p}")
                            if candidates:
                                print(f"候选参数: {', '.join(candidates[:5])}")
                            repl = input("请输入替换的参数名，或直接回车跳过: ").strip()
                            if repl and repl in available_params:
                                new_params.append(repl)
                            elif not repl:
                                pass
                            else:
                                print("参数名无效，已跳过。")
                        acfg['opt_params'] = new_params if new_params else opt_params0
                    if simulator_type == 'corca_state':
                        if not os.getenv("CORCA_EXEC_STATE_PATH"):
                            path1 = input("请输入 CORCA_EXEC_STATE_PATH 路径，或直接回车跳过: ").strip()
                            if path1:
                                os.environ["CORCA_EXEC_STATE_PATH"] = path1
                        if not os.getenv("CORCA_HDF5_FILE_PATH"):
                            path2 = input("请输入 CORCA_HDF5_FILE_PATH 路径，或直接回车跳过: ").strip()
                            if path2:
                                os.environ["CORCA_HDF5_FILE_PATH"] = path2
                    return acfg
                
                analysis_config = clarify_inputs(analysis_config)

                use_rec = str(os.getenv("AICO_USE_RECOMMENDER", "")).strip().lower() in {"1", "true", "yes", "y"}
                if use_rec:
                    analysis_config["algo_recommender"] = {
                        "enabled": True,
                        "model_path": os.getenv("AICO_RECOMMENDER_MODEL", "").strip()
                        or "/home/ycl/AICO-Intellig/results/model_algo_ranker_pool9_warm10.joblib",
                        "warmup_iterations": int(os.getenv("AICO_RECOMMENDER_WARMUP_ITERS", "10")),
                        "warmup_time_s": float(os.getenv("AICO_RECOMMENDER_WARMUP_TIME", "30")),
                        "warmup_k": int(os.getenv("AICO_RECOMMENDER_WARMUP_K", "10")),
                        "seed": int(os.getenv("AICO_RECOMMENDER_SEED", "0")),
                    }

                expr0 = str(analysis_config.get("objective_expression") or "").strip()
                if not expr0 and str(analysis_config.get("primary_objective") or "").lower() == "minimize_nv":
                    ui_lower = str(user_input or "").lower()
                    if ("cpu" in ui_lower) and (problem_type == "ex_poisson_adapt_indicator"):
                        analysis_config["objective_expression"] = "result.get('nv', float('inf')) + 0.1*result.get('cpu', 0.0)"

                feasible_count = None
                try:
                    import re
                    pconf0 = optimizer.problem_config.get_problem_config(problem_type) or {}
                    simulator_type0 = pconf0.get('simulator_type', 'freefem')
                    if simulator_type0 == 'corca_state':
                        ui0 = user_input
                        ui0_lower = ui0.lower()
                        if ('可行解' in ui0) or ('feasible' in ui0_lower):
                            m = re.search(r'(\d+)\s*(个|组)', ui0)
                            if m:
                                feasible_count = int(m.group(1))
                            if feasible_count is None:
                                raw = input("请输入需要的可行解数量(默认10): ").strip()
                                feasible_count = int(raw) if raw else 10
                            analysis_config['feasible_count'] = int(feasible_count)
                            analysis_config['objective_type'] = 'feasible'

                            m2 = re.search(r'(?:采样|抽样|尝试)\s*(\d+)\s*(?:次)?', ui0)
                            if m2:
                                try:
                                    analysis_config['feasible_max_attempts'] = int(m2.group(1))
                                except Exception:
                                    pass
                except Exception:
                    feasible_count = None

                # 显示分析结果
                objective_type = analysis_config.get('objective_type', 'simple')
                primary_objective = analysis_config.get('primary_objective', 'Unknown')
                opt_params = analysis_config.get('opt_params', [])
                
                # 优先使用用户指定的算法
                user_specified = analysis_config.get('user_specified_algorithm')
                if user_specified and '-' in user_specified:
                    optimizer_name, method = user_specified.split('-', 1)
                    print(f"🎯 用户指定算法: {user_specified}")
                else:
                    optimizer_name = analysis_config.get('optimizer', 'SciPy')
                    method = analysis_config.get('method', 'Nelder-Mead')
                    rec_cfg = analysis_config.get("algo_recommender") or {}
                    if bool(rec_cfg.get("enabled", False)):
                        model_path = str(rec_cfg.get("model_path") or "").strip()
                        if not model_path:
                            model_path = "/home/ycl/AICO-Intellig/results/model_algo_ranker_pool9_warm10.joblib"
                        print("✓ 算法选择: 推荐器(warm-start) 将在执行前短试跑后决定")
                        print(
                            f"   warmup: iters={rec_cfg.get('warmup_iterations', 10)} time_s={rec_cfg.get('warmup_time_s', 30)} k={rec_cfg.get('warmup_k', 10)} seed={rec_cfg.get('seed', 0)}"
                        )
                        print(f"   model: {model_path}")
                        analysis_config["brief_logs"] = True
                    else:
                        print(f"✓ 智能选择: {optimizer_name} - {method}")
                        if analysis_config.get('algorithm_reason'):
                            print(f"💡 选择理由: {analysis_config.get('algorithm_reason')}")
                
                objective_expression = analysis_config.get('objective_expression', '')
                constraints = analysis_config.get('constraints', [])
                target_values = analysis_config.get('target_values', {})

                if feasible_count is not None or str(objective_type).lower() == 'feasible':
                    display_primary_objective = 'feasible'
                else:
                    display_primary_objective = primary_objective

                print(f"✓ 目标类型: {objective_type}")
                print(f"✓ 主要目标: {display_primary_objective}")
                print(f"✓ 优化参数: {', '.join(opt_params)}")

                if feasible_count is not None:
                    print(f"🎯 任务: 搜索 {feasible_count} 个可行解 (CORCA-sim)")
                
                if objective_expression:
                    print(f"🎯 目标函数: {objective_expression}")
                
                if constraints:
                    print(f"📋 约束条件:")
                    for constraint in constraints:
                        if isinstance(constraint, dict):
                            metric = constraint.get('metric', '')
                            constraint_type = constraint.get('type', '')
                            value = constraint.get('value', '')
                            print(f"   - {metric} {constraint_type} {value}")
                        else:
                            print(f"   - {constraint}")
                
                if target_values:
                    print(f"🎯 目标值: {target_values}")

                confirm = input("\n确认开始执行？(y/n，默认y): ").strip().lower()
                if confirm and confirm not in ['y', 'yes', '是']:
                    print("已取消优化。")
                    continue

                if feasible_count is not None:
                    results = optimizer.find_feasible_solutions(
                        analysis_config=analysis_config,
                        opt_params=opt_params,
                        requested_count=feasible_count,
                    )

                    print("\n" + "="*50)
                    print("📊 可行解搜索结果")
                    print("="*50)
                    if results.get('success'):
                        print(f"✅ 已找到 {results.get('feasible_found', 0)}/{results.get('feasible_requested', 0)} 个可行解")
                    else:
                        print(f"⚠️ {results.get('message', '')}")
                        print(f"✅ 已找到 {results.get('feasible_found', 0)}/{results.get('feasible_requested', 0)} 个可行解")

                    sols = results.get('feasible_solutions', []) or []
                    for idx, s in enumerate(sols[:min(10, len(sols))], start=1):
                        params_i = s.get('params', {})
                        res_i = s.get('result', {})
                        keff = res_i.get('keff')
                        fq = res_i.get('FQ')
                        print(f"\n[{idx}] params={params_i}")
                        if keff is not None:
                            print(f"   keff={keff}")
                        if fq is not None:
                            print(f"   FQ={fq}")
                    continue

                # 询问是否使用智能调度
                use_scheduler = input("🤖 是否使用智能算法调度器？(y/n，默认n): ").strip().lower()
                
                if use_scheduler in ['y', 'yes', '是']:
                    # 使用调度器优化
                    results = optimizer.optimize_with_scheduler(
                        analysis_config=analysis_config,
                        opt_params=opt_params,
                        initial_method=method,
                        initial_optimizer=optimizer_name
                    )
                else:
                    # 传统优化
                    results = optimizer.optimize(
                        analysis_config=analysis_config,
                        opt_params=opt_params,
                        method=method,
                        optimizer=optimizer_name
                    )

                print("\n" + "="*50)
                print("📊 优化结果")
                print("="*50)
                
                if results.get('success'):
                    print(f"✅ 优化成功完成！")
                    print(f"🎯 最优目标值: {results.get('best_value', 'N/A'):.6f}")
                    print(f"🔧 最优参数:")
                    for param, value in results.get('best_params', {}).items():
                        print(f"   {param}: {value:.6f}")
                    print(f"⏱️  执行时间: {results.get('execution_time', 0):.2f}秒")
                    print(f"🔄 迭代次数: {results.get('iterations', 'N/A')}")
                    
                    # 显示最优解的仿真结果
                    if 'iteration_history' in results and results['iteration_history']:
                        best_iteration = min(results['iteration_history'], key=lambda x: x['objective'])
                        if 'result' in best_iteration:
                            print(f"\n📈 最优解的仿真结果:")
                            sim_result = best_iteration['result']
                            for key, value in sim_result.items():
                                if key != 'success' and isinstance(value, (int, float)):
                                    print(f"   {key}: {value:.6f}")
                    
                    show_eval = str(os.getenv("AICO_SHOW_EVAL", "")).strip().lower() in {"1", "true", "yes", "y"}
                    if show_eval:
                        analysis = optimizer.analyze_results(results, primary_objective, target_values)
                        print(f"\n📈 结果分析:")
                        print(analysis)
                else:
                    print(f"❌ 优化失败: {results.get('message', '未知错误')}")

            except Exception as e:
                print(f"❌ 优化过程出错: {e}")
                import traceback
                traceback.print_exc()
                continue
    except Exception as e:
        print(f"❌ 程序启动出错: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
