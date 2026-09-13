import argparse
import json
import math
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    # 1. 读输入
    with open(args.input, "r") as f:
        data = json.load(f)
    
    params = data.get("params", {})
    x = params.get("x", 0.0)
    y = params.get("y", 0.0)

    # 2. 执行计算（这里是 Sphere 函数示例）
    print(f"Running simulation with x={x}, y={y}")
    
    # 模拟计算耗时
    import time
    time.sleep(0.1)
    
    val = x**2 + y**2
    
    # 3. 写输出
    result = {
        "success": True,
        "metrics": {
            "sphere_val": val,
            "constraint_violation": max(0, x + y - 10)  # 示例约束 x+y <= 10
        }
    }
    
    with open(args.output, "w") as f:
        json.dump(result, f)
        
    print(f"Done. Result: {val}")

if __name__ == "__main__":
    main()
