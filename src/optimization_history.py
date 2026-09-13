"""
优化历史数据库与文件存储管理器
记录和分析历史优化数据，为LLM决策提供参考，同时支持文件系统落盘
"""
import json
import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid
import shutil


class OptimizationHistory:
    """优化历史数据库与文件管理器"""
    
    def __init__(self, db_path: str = "results/optimization_history.db", runs_dir: str = "results/runs"):
        self.db_path = db_path
        self.runs_dir = runs_dir
        
        # 确保目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(runs_dir, exist_ok=True)
        
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建优化记录表 (增加 run_id 和 trajectory_path)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE,
                timestamp TEXT NOT NULL,
                problem_type TEXT NOT NULL,
                user_input TEXT NOT NULL,
                objective_type TEXT,
                opt_params TEXT,  -- JSON格式存储参数列表
                target_values TEXT,  -- JSON格式存储目标值
                optimizer TEXT NOT NULL,
                method TEXT NOT NULL,
                best_value REAL,
                iterations INTEGER,
                execution_time REAL,
                convergence_quality TEXT,
                target_achieved BOOLEAN,
                best_params TEXT,  -- JSON格式存储最优参数
                success BOOLEAN NOT NULL,
                scheduler_used BOOLEAN DEFAULT FALSE,
                algorithm_switches INTEGER DEFAULT 0,
                notes TEXT,
                trajectory_path TEXT
            )
        ''')
        
        # 创建算法性能统计表 (保持不变)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS algorithm_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_type TEXT NOT NULL,
                optimizer TEXT NOT NULL,
                method TEXT NOT NULL,
                avg_performance REAL,
                success_rate REAL,
                avg_iterations REAL,
                avg_execution_time REAL,
                usage_count INTEGER,
                last_updated TEXT,
                UNIQUE(problem_type, optimizer, method)
            )
        ''')
        
        # 创建问题模式表 (保持不变)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS problem_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_type TEXT NOT NULL,
                param_pattern TEXT,  -- 参数组合模式
                objective_pattern TEXT,  -- 目标函数模式
                successful_algorithms TEXT,  -- JSON格式存储成功的算法
                avg_difficulty REAL,  -- 平均难度评估
                pattern_count INTEGER,
                last_seen TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # 检查是否需要迁移旧表 (增加列)
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(optimization_records)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'run_id' not in columns:
                cursor.execute("ALTER TABLE optimization_records ADD COLUMN run_id TEXT")
                cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_run_id ON optimization_records(run_id)")
            if 'trajectory_path' not in columns:
                cursor.execute("ALTER TABLE optimization_records ADD COLUMN trajectory_path TEXT")
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"数据库迁移警告: {e}")

    def create_run(self, problem_type: str, optimizer: str, method: str) -> str:
        """创建一个新的运行记录，返回 run_id"""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        run_dir = os.path.join(self.runs_dir, problem_type, f"{optimizer}_{method}", run_id)
        os.makedirs(run_dir, exist_ok=True)
        return run_id, run_dir

    def save_run_data(self, run_id: str, run_dir: str, data: Dict[str, Any]):
        """保存运行数据到文件系统"""
        # 1. 保存元数据
        with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
            meta = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "problem_type": data.get("analysis_config", {}).get("problem_type"),
                "optimizer": data.get("optimizer"),
                "method": data.get("method"),
                "analysis_config": data.get("analysis_config"),
                "scheduler_summary": data.get("scheduler_summary")
            }
            json.dump(meta, f, indent=2)
            
        # 2. 保存轨迹 (如果有)
        trajectory = data.get("iteration_history", [])
        trajectory_path = os.path.join(run_dir, "trajectory.json")
        with open(trajectory_path, "w") as f:
            json.dump(trajectory, f, indent=2)
            
        # 3. 保存最终结果
        with open(os.path.join(run_dir, "result.json"), "w") as f:
            result = {k: v for k, v in data.items() if k not in ["iteration_history", "analysis_config"]}
            # 处理 numpy 类型
            import numpy as np
            def default(o):
                if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                                  np.int16, np.int32, np.int64, np.uint8,
                                  np.uint16, np.uint32, np.uint64)):
                    return int(o)
                elif isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
                    return float(o)
                elif isinstance(o, (np.ndarray,)):
                    return o.tolist()
                return str(o)
            json.dump(result, f, indent=2, default=default)
            
        return trajectory_path

    def add_optimization_record(self, result_data: Dict[str, Any]):
        """添加优化记录 (同时写入 DB 和 文件)"""
        
        # 提取关键信息
        analysis_config = result_data.get('analysis_config', {})
        problem_type = analysis_config.get('problem_type', 'unknown')
        optimizer = result_data.get('optimizer', 'unknown')
        method = result_data.get('method', 'unknown')
        
        # 1. 创建文件存储
        run_id, run_dir = self.create_run(problem_type, optimizer, method)
        trajectory_path = self.save_run_data(run_id, run_dir, result_data)
        
        # 2. 写入数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        scheduler_summary = result_data.get('scheduler_summary', {})
        
        record = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'problem_type': problem_type,
            'user_input': analysis_config.get('user_input', ''),
            'objective_type': analysis_config.get('objective_type', 'simple'),
            'opt_params': json.dumps(analysis_config.get('opt_params', [])),
            'target_values': json.dumps(analysis_config.get('target_values', {})),
            'optimizer': optimizer,
            'method': method,
            'best_value': result_data.get('best_value'),
            'iterations': result_data.get('iterations', 0),
            'execution_time': result_data.get('execution_time', 0),
            'convergence_quality': result_data.get('convergence_quality', 'unknown'),
            'target_achieved': result_data.get('target_achieved', False),
            'best_params': json.dumps(result_data.get('best_params', {})),
            'success': result_data.get('success', False),
            'scheduler_used': 'scheduler_summary' in result_data,
            'algorithm_switches': scheduler_summary.get('switch_decisions', 0),
            'notes': result_data.get('message', ''),
            'trajectory_path': trajectory_path
        }
        
        # 处理 numpy 类型用于 DB 存储
        for k, v in record.items():
            if hasattr(v, 'item'):  # numpy scalar
                record[k] = v.item()
        
        cursor.execute('''
            INSERT INTO optimization_records 
            (run_id, timestamp, problem_type, user_input, objective_type, opt_params, 
             target_values, optimizer, method, best_value, iterations, 
             execution_time, convergence_quality, target_achieved, best_params, 
             success, scheduler_used, algorithm_switches, notes, trajectory_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', tuple(record.values()))
        
        conn.commit()
        conn.close()
        
        # 更新算法性能统计
        self._update_algorithm_performance(record)
        self._update_problem_patterns(record)
        
        return run_id
    
    def _update_algorithm_performance(self, record: Dict):
        """更新算法性能统计"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 查询现有统计
        cursor.execute('''
            SELECT avg_performance, success_rate, avg_iterations, avg_execution_time, usage_count
            FROM algorithm_performance 
            WHERE problem_type=? AND optimizer=? AND method=?
        ''', (record['problem_type'], record['optimizer'], record['method']))
        
        existing = cursor.fetchone()
        
        if existing:
            # 更新现有统计
            old_avg_perf, old_success_rate, old_avg_iter, old_avg_time, old_count = existing
            new_count = old_count + 1
            
            # 计算新的平均值
            # 忽略 Inf 或极其巨大的值，避免拉低平均性能
            current_value = record['best_value']
            if current_value is not None and current_value < 1e9: # 设定一个合理的阈值
                if old_avg_perf > 1e9: # 如果旧值是坏值，直接用新值覆盖
                    new_avg_perf = current_value
                else:
                    new_avg_perf = (old_avg_perf * old_count + current_value) / new_count
            else:
                new_avg_perf = old_avg_perf
            
            new_success_rate = (old_success_rate * old_count + (1 if record['success'] else 0)) / new_count
            new_avg_iter = (old_avg_iter * old_count + record['iterations']) / new_count
            new_avg_time = (old_avg_time * old_count + record['execution_time']) / new_count
            
            cursor.execute('''
                UPDATE algorithm_performance 
                SET avg_performance=?, success_rate=?, avg_iterations=?, 
                    avg_execution_time=?, usage_count=?, last_updated=?
                WHERE problem_type=? AND optimizer=? AND method=?
            ''', (new_avg_perf, new_success_rate, new_avg_iter, new_avg_time, 
                  new_count, datetime.now().isoformat(),
                  record['problem_type'], record['optimizer'], record['method']))
        else:
            # 插入新统计
            cursor.execute('''
                INSERT INTO algorithm_performance 
                (problem_type, optimizer, method, avg_performance, success_rate, 
                 avg_iterations, avg_execution_time, usage_count, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (record['problem_type'], record['optimizer'], record['method'],
                  record['best_value'] or 0, 1 if record['success'] else 0,
                  record['iterations'], record['execution_time'], 1,
                  datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def _update_problem_patterns(self, record: Dict):
        """更新问题模式统计"""
        # 生成参数模式和目标模式的简化表示
        try:
            opt_params = json.loads(record['opt_params'])
            param_pattern = f"{len(opt_params)}params_" + "_".join(sorted(opt_params)[:3])  # 取前3个参数
        except:
            param_pattern = "unknown"
            
        objective_pattern = f"{record['objective_type']}_{record['problem_type']}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 查询现有模式
        cursor.execute('''
            SELECT successful_algorithms, avg_difficulty, pattern_count
            FROM problem_patterns 
            WHERE problem_type=? AND param_pattern=? AND objective_pattern=?
        ''', (record['problem_type'], param_pattern, objective_pattern))
        
        existing = cursor.fetchone()
        
        if existing:
            # 更新现有模式
            successful_algs = json.loads(existing[0])
            old_difficulty = existing[1]
            old_count = existing[2]
            
            # 如果优化成功，添加到成功算法列表
            if record['success']:
                alg_key = f"{record['optimizer']}-{record['method']}"
                if alg_key not in successful_algs:
                    successful_algs[alg_key] = 0
                successful_algs[alg_key] += 1
            
            # 更新难度评估（基于迭代次数和执行时间）
            difficulty = (record['iterations'] / 100 + record['execution_time'] / 60) / 2
            new_difficulty = (old_difficulty * old_count + difficulty) / (old_count + 1)
            
            cursor.execute('''
                UPDATE problem_patterns 
                SET successful_algorithms=?, avg_difficulty=?, pattern_count=?, last_seen=?
                WHERE problem_type=? AND param_pattern=? AND objective_pattern=?
            ''', (json.dumps(successful_algs), new_difficulty, old_count + 1,
                  datetime.now().isoformat(), record['problem_type'], 
                  param_pattern, objective_pattern))
        else:
            # 插入新模式
            successful_algs = {}
            if record['success']:
                alg_key = f"{record['optimizer']}-{record['method']}"
                successful_algs[alg_key] = 1
            
            difficulty = (record['iterations'] / 100 + record['execution_time'] / 60) / 2
            
            cursor.execute('''
                INSERT INTO problem_patterns 
                (problem_type, param_pattern, objective_pattern, successful_algorithms, 
                 avg_difficulty, pattern_count, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (record['problem_type'], param_pattern, objective_pattern,
                  json.dumps(successful_algs), difficulty, 1,
                  datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_historical_insights(self, problem_type: str, opt_params: List[str], 
                              objective_type: str = 'simple') -> Dict[str, Any]:
        """获取历史洞察，供LLM参考"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        insights = {
            'total_records': 0,
            'similar_problems': [],
            'algorithm_recommendations': [],
            'performance_stats': {},
            'success_patterns': [],
            'difficulty_assessment': 'medium'
        }
        
        # 1. 获取总记录数
        cursor.execute('SELECT COUNT(*) FROM optimization_records')
        insights['total_records'] = cursor.fetchone()[0]
        
        # 2. 查找相似问题
        cursor.execute('''
            SELECT user_input, optimizer, method, best_value, convergence_quality, success
            FROM optimization_records 
            WHERE problem_type=? AND objective_type=?
            ORDER BY timestamp DESC LIMIT 10
        ''', (problem_type, objective_type))
        
        similar_problems = cursor.fetchall()
        insights['similar_problems'] = [
            {
                'user_input': row[0],
                'algorithm': f"{row[1]}-{row[2]}",
                'performance': row[3],
                'quality': row[4],
                'success': row[5]
            }
            for row in similar_problems
        ]
        
        # 3. 获取算法推荐
        # 修改排序逻辑：优先按平均性能（目标值/误差）升序排序（越小越好），其次按成功率降序，最后按使用次数降序
        cursor.execute('''
            SELECT optimizer, method, success_rate, avg_performance, usage_count, avg_execution_time, avg_iterations
            FROM algorithm_performance 
            WHERE problem_type=?
            ORDER BY avg_performance ASC, success_rate DESC, usage_count DESC LIMIT 5
        ''', (problem_type,))
        
        recommendations = cursor.fetchall()
        insights['algorithm_recommendations'] = [
            {
                'algorithm': f"{row[0]}-{row[1]}",
                'success_rate': row[2],
                'avg_performance': row[3],
                'usage_count': row[4],
                'avg_execution_time': row[5] if row[5] is not None else 0.0,
                'avg_iterations': row[6] if row[6] is not None else 0.0
            }
            for row in recommendations
        ]
        
        # 4. 获取问题模式
        param_pattern = f"{len(opt_params)}params_" + "_".join(sorted(opt_params)[:3])
        objective_pattern = f"{objective_type}_{problem_type}"
        
        cursor.execute('''
            SELECT successful_algorithms, avg_difficulty, pattern_count
            FROM problem_patterns 
            WHERE problem_type=? AND param_pattern=? AND objective_pattern=?
        ''', (problem_type, param_pattern, objective_pattern))
        
        pattern = cursor.fetchone()
        if pattern:
            successful_algs = json.loads(pattern[0])
            insights['success_patterns'] = [
                {'algorithm': alg, 'success_count': count}
                for alg, count in sorted(successful_algs.items(), 
                                       key=lambda x: x[1], reverse=True)
            ]
            
            # 难度评估
            difficulty = pattern[1]
            if difficulty < 0.3:
                insights['difficulty_assessment'] = 'easy'
            elif difficulty < 0.7:
                insights['difficulty_assessment'] = 'medium'
            else:
                insights['difficulty_assessment'] = 'hard'
        
        conn.close()
        return insights
    
    def generate_llm_context(self, problem_type: str, opt_params: List[str], 
                           objective_type: str = 'simple') -> str:
        """生成供LLM参考的历史上下文"""
        insights = self.get_historical_insights(problem_type, opt_params, objective_type)
        
        context = f"""
历史优化数据分析 (基于{insights['total_records']}条历史记录):

问题类型: {problem_type}
参数数量: {len(opt_params)}
目标类型: {objective_type}
预估难度: {insights['difficulty_assessment']}

算法推荐 (基于历史成功率):
"""
        
        for i, rec in enumerate(insights['algorithm_recommendations'][:3], 1):
            avg_perf = rec['avg_performance'] if rec['avg_performance'] is not None else float('inf')
            avg_time = rec['avg_execution_time'] if rec['avg_execution_time'] is not None else 0.0
            avg_iter = rec['avg_iterations'] if rec['avg_iterations'] is not None else 0.0
            
            context += f"{i}. {rec['algorithm']} - 成功率: {rec['success_rate']:.1%}, 平均耗时: {avg_time:.2f}s, 平均迭代: {avg_iter:.1f}, 平均误差: {avg_perf:.2e}, 使用次数: {rec['usage_count']}\n"
        
        if insights['success_patterns']:
            context += f"\n相似问题成功模式:\n"
            for pattern in insights['success_patterns'][:3]:
                context += f"• {pattern['algorithm']} (成功{pattern['success_count']}次)\n"
        
        if insights['similar_problems']:
            context += f"\n最近相似问题:\n"
            for prob in insights['similar_problems'][:3]:
                status = "成功" if prob['success'] else "失败"
                context += f"• \"{prob['user_input'][:50]}...\" - {prob['algorithm']} ({status})\n"
        
        return context.strip()

    def delete_record(self, record_id: int) -> bool:
        """删除指定的优化记录"""
        return self.delete_records([record_id]) > 0

    def delete_records(self, record_ids: List[int]) -> int:
        """批量删除优化记录"""
        try:
            if not record_ids:
                return 0
                
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 使用 executemany 或者 IN 子句
            placeholders = ','.join(['?'] * len(record_ids))
            query = f'DELETE FROM optimization_records WHERE id IN ({placeholders})'
            
            cursor.execute(query, record_ids)
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            return deleted_count
        except Exception as e:
            print(f"批量删除记录失败: {e}")
            return 0

    def update_record_note(self, record_id: int, note: str) -> bool:
        """更新记录的备注"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 检查记录是否存在
            cursor.execute('SELECT id FROM optimization_records WHERE id=?', (record_id,))
            if not cursor.fetchone():
                conn.close()
                return False
                
            cursor.execute('UPDATE optimization_records SET notes=? WHERE id=?', (note, record_id))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"更新备注失败: {e}")
            return False
            
    def update_record_status(self, record_id: int, success: bool) -> bool:
        """更新记录的成功状态"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 检查记录是否存在
            cursor.execute('SELECT id FROM optimization_records WHERE id=?', (record_id,))
            if not cursor.fetchone():
                conn.close()
                return False
                
            cursor.execute('UPDATE optimization_records SET success=? WHERE id=?', (success, record_id))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"更新状态失败: {e}")
            return False
