"""
工具函数模块
包含各种辅助函数
"""
import json
from typing import Union

try:
    import pandas as pd
except Exception:
    pd = None


def save_result_tool(data, filename: str) -> str:
    """保存结果到文件"""
    try:
        if filename.endswith('.json'):
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif filename.endswith('.csv'):
            if pd is None:
                raise RuntimeError("缺少依赖 pandas，无法保存 CSV")
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = pd.DataFrame(data)
            df.to_csv(filename, index=False, encoding='utf-8')
        else:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(str(data))
        return f"结果已保存到 {filename}"
    except Exception as e:
        return f"保存失败: {str(e)}"
