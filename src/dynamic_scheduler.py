"""
动态算法调度器
在优化过程中智能切换算法和调整参数
"""
import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


class DynamicScheduler:
    """动态算法调度器"""
    
    def __init__(self, 
                 client,  # 直接要求传入client
                 eval_interval: int = 20,
                 progress_protect_ratio: float = 0.5,
                 time_protect: int = 1200,  # 20分钟
                 max_iter: int = 100,
                 confidence_threshold: float = 0.7):
        """
        初始化调度器
        
        Args:
            client: OpenAI客户端实例
            eval_interval: 评估间隔（迭代次数）
            progress_protect_ratio: 进度保护阈值
            time_protect: 时间保护阈值（秒）
            max_iter: 最大迭代次数
            confidence_threshold: LLM决策置信度阈值
        """
        self.client = client
        self.eval_interval = eval_interval
        self.progress_protect_ratio = progress_protect_ratio
        self.time_protect = time_protect
        self.max_iter = max_iter
        self.confidence_threshold = confidence_threshold
        
        # 运行状态
        self.start_time = time.time()
        self.loss_history = []
        self.decision_log = []
        self.current_algorithm = None
        self.current_params = None
        self.protected = False
    
    def should_evaluate(self, iteration: int) -> bool:
        """判断是否需要进行评估"""
        return iteration > 0 and iteration % self.eval_interval == 0
    
    def is_protected(self, iteration: int) -> bool:
        """判断是否进入保护阶段"""
        if self.protected:
            return True
            
        # 进度保护
        progress = iteration / self.max_iter
        if progress >= self.progress_protect_ratio:
            self.protected = True
            self._log_decision(iteration, "进入进度保护阶段", "progress_protection")
            return True
        
        # 时间保护
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.time_protect:
            self.protected = True
            self._log_decision(iteration, "进入时间保护阶段", "time_protection")
            return True
        
        return False
    
    def record_loss(self, iteration: int, loss: float):
        """记录损失值"""
        self.loss_history.append({
            'iteration': iteration,
            'loss': loss,
            'timestamp': time.time()
        })
    
    def evaluate_and_decide(self, iteration: int, current_algorithm: str, 
                          available_algorithms: List[str]) -> Dict[str, Any]:
        """
        评估当前优化趋势并做出决策
        
        Returns:
            决策结果字典，包含action, algorithm, confidence等
        """
        if self.is_protected(iteration):
            return {
                'action': 'continue',
                'algorithm': current_algorithm,
                'reason': '已进入保护阶段',
                'confidence': 1.0
            }
        
        # 分析最近的优化趋势
        trend_analysis = self._analyze_trend()
        
        # 调用LLM进行决策
        try:
            llm_decision = self._llm_decide(
                iteration, current_algorithm, available_algorithms, trend_analysis
            )
            
            # 检查置信度
            if llm_decision.get('confidence', 0) < self.confidence_threshold:
                decision = {
                    'action': 'continue',
                    'algorithm': current_algorithm,
                    'reason': f'LLM置信度过低({llm_decision.get("confidence", 0):.2f})',
                    'confidence': llm_decision.get('confidence', 0)
                }
            else:
                decision = llm_decision
            
        except Exception as e:
            # LLM调用失败，使用默认策略
            decision = {
                'action': 'continue',
                'algorithm': current_algorithm,
                'reason': f'LLM调用失败: {str(e)}',
                'confidence': 0.0
            }
        
        # 记录决策
        self._log_decision(iteration, decision['reason'], decision['action'], decision)
        
        return decision
    
    def _analyze_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """分析最近的优化趋势"""
        if len(self.loss_history) < 3:
            return {
                'trend': 'insufficient_data', 
                'improvement_rate': 0,
                'variance': 0,
                'recent_losses': [],
                'total_evaluations': len(self.loss_history)
            }
        
        # 获取最近的损失值
        recent_losses = [item['loss'] for item in self.loss_history[-window_size:]]
        
        # 计算趋势指标
        if len(recent_losses) >= 2:
            # 改进率
            improvement_rate = (recent_losses[0] - recent_losses[-1]) / max(abs(recent_losses[0]), 1e-10)
            
            # 方差（稳定性）
            variance = np.var(recent_losses)
            
            # 判断趋势
            if improvement_rate > 0.01:  # 显著改进
                trend = 'improving'
            elif improvement_rate > -0.001:  # 轻微改进或停滞
                if variance < 1e-6:
                    trend = 'stagnant'
                else:
                    trend = 'slow_progress'
            else:  # 恶化
                trend = 'deteriorating'
        else:
            improvement_rate = 0
            variance = 0
            trend = 'insufficient_data'
        
        return {
            'trend': trend,
            'improvement_rate': improvement_rate,
            'variance': variance,
            'recent_losses': recent_losses[-5:],  # 最近5次损失
            'total_evaluations': len(self.loss_history)
        }
    
    def _llm_decide(self, iteration: int, current_algorithm: str, 
                   available_algorithms: List[str], trend_analysis: Dict) -> Dict[str, Any]:
        """使用LLM进行决策"""
        
        # 构建prompt
        prompt = f"""你是一个优化算法调度专家。请根据以下信息决定是否需要切换优化算法：

当前状态：
- 当前算法：{current_algorithm}
- 当前迭代：{iteration}
- 总迭代数：{self.max_iter}
- 运行时间：{time.time() - self.start_time:.1f}秒

优化趋势分析：
- 趋势类型：{trend_analysis['trend']}
- 改进率：{trend_analysis['improvement_rate']:.6f}
- 方差（稳定性）：{trend_analysis['variance']:.6f}
- 最近5次损失值：{trend_analysis['recent_losses']}
- 总评估次数：{trend_analysis['total_evaluations']}

可选算法：{', '.join(available_algorithms)}

请根据以下规则做出决策：
1. 如果趋势为'improving'且改进率>0.01，继续当前算法
2. 如果趋势为'stagnant'或'slow_progress'，考虑切换算法
3. 如果趋势为'deteriorating'，强烈建议切换算法

返回JSON格式：
{{
    "action": "continue" 或 "switch",
    "algorithm": "推荐的算法名称",
    "reason": "决策理由",
    "confidence": 0.0-1.0之间的置信度
}}
"""

        try:
            print(f"\n🤖 LLM正在分析优化趋势...")
            print(f"   📊 当前趋势: {trend_analysis['trend']}")
            print(f"   📈 改进率: {trend_analysis['improvement_rate']:.4f}")
            print(f"   📉 最近损失: {trend_analysis['recent_losses'][-3:] if len(trend_analysis['recent_losses']) >= 3 else trend_analysis['recent_losses']}")
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是优化算法调度专家，根据优化趋势智能决策算法切换。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300,
                timeout=10  # 10秒超时
            )
            
            # 显示LLM的原始回复
            llm_response = response.choices[0].message.content.strip()
            print(f"💭 LLM分析意见:")
            
            # 尝试解析JSON并提取关键信息
            try:
                result = json.loads(llm_response)
                action = result.get('action', 'continue')
                algorithm = result.get('algorithm', current_algorithm)
                reason = result.get('reason', '无理由')
                confidence = result.get('confidence', 0.0)
                
                # 显示LLM的决策分析
                print(f"   🎯 决策: {action}")
                print(f"   🔧 推荐算法: {algorithm}")
                print(f"   💡 理由: {reason}")
                print(f"   📊 置信度: {confidence:.2f}")
                
            except json.JSONDecodeError:
                # 如果不是JSON格式，直接显示原始回复
                print(f"   💬 {llm_response}")
                # 尝试从文本中提取信息
                result = {
                    'action': 'continue',
                    'algorithm': current_algorithm,
                    'reason': '解析失败，继续当前算法',
                    'confidence': 0.3
                }
            
            # 验证结果格式
            required_keys = ['action', 'algorithm', 'reason', 'confidence']
            if not all(key in result for key in required_keys):
                raise ValueError("LLM返回格式不完整")
            
            # 验证算法名称
            if result['algorithm'] not in available_algorithms:
                print(f"   ⚠️ 推荐算法 {result['algorithm']} 不可用，继续当前算法")
                result['algorithm'] = current_algorithm
                result['action'] = 'continue'
                result['reason'] += ' (推荐算法不可用)'
            
            return result
            
        except Exception as e:
            print(f"   ❌ LLM分析失败: {str(e)}")
            raise Exception(f"LLM决策失败: {str(e)}")
    
    def _log_decision(self, iteration: int, reason: str, action: str, 
                     decision_data: Dict = None):
        """记录决策日志"""
        log_entry = {
            'iteration': iteration,
            'timestamp': time.time(),
            'elapsed_time': time.time() - self.start_time,
            'reason': reason,
            'action': action,
            'decision_data': decision_data or {}
        }
        self.decision_log.append(log_entry)
        
        # 简化的最终决策输出
        if decision_data and decision_data.get('action') == 'switch':
            print(f"🔄 最终决策: 切换到 {decision_data.get('algorithm', 'Unknown')}")
        elif decision_data and decision_data.get('confidence', 0) > 0:
            print(f"✅ 最终决策: 继续当前算法")
        else:
            print(f"📋 决策: {reason}")
        if decision_data and decision_data.get('confidence'):
            print(f"   置信度: {decision_data['confidence']:.2f}")
    
    def get_decision_summary(self) -> Dict[str, Any]:
        """获取决策摘要"""
        total_decisions = len(self.decision_log)
        switch_decisions = len([d for d in self.decision_log if d['action'] == 'switch'])
        
        return {
            'total_runtime': time.time() - self.start_time,
            'total_evaluations': len(self.loss_history),
            'total_decisions': total_decisions,
            'switch_decisions': switch_decisions,
            'protection_triggered': self.protected,
            'decision_log': self.decision_log,
            'final_loss_trend': self._analyze_trend()
        }
    
    def save_log(self, filepath: str):
        """保存完整日志"""
        log_data = {
            'scheduler_config': {
                'eval_interval': self.eval_interval,
                'progress_protect_ratio': self.progress_protect_ratio,
                'time_protect': self.time_protect,
                'max_iter': self.max_iter,
                'confidence_threshold': self.confidence_threshold
            },
            'runtime_summary': self.get_decision_summary(),
            'loss_history': self.loss_history,
            'decision_log': self.decision_log
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
