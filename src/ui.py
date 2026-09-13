import json
import tkinter as tk
import queue
from tkinter import ttk, messagebox

# 兼容从项目根或src目录启动
try:
    from src.optimizer import DeepFreeFEMOptimizer
    from src.problem_config import get_optimizer_config
except ImportError:
    import os, sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from optimizer import DeepFreeFEMOptimizer
    from problem_config import get_optimizer_config

import re
import io
import contextlib
import sys, queue
from tkinter import font as tkfont


class AICOApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AICO 优化系统")
        self.root.geometry("980x680")

        # 初始化优化器与配置
        try:
            self.optimizer = DeepFreeFEMOptimizer()
            self.optimizer_cfg = get_optimizer_config()
        except Exception as e:
            messagebox.showerror("初始化失败", str(e))
            raise

        # 主分栏
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 左侧：仅保留自然语言输入与三个按钮
        ttk.Label(left, text="用自然语言描述需求（目标/变量/约束）").pack(anchor=tk.W)
        self.input_text = tk.Text(left, height=10)
        self.input_text.insert(
            tk.END,
            "最小化SNS压降，优化uMax和Mu，约束Re<2300，入口速度在[8,12]范围。"
        )
        self.input_text.pack(fill=tk.X, pady=4)

        # 初始状态（默认设置存放到 state）
        types = self.optimizer.problem_config.get_available_problem_types()
        default_type = types[0] if types else "sns"
        pconf = self.optimizer.problem_config.get_problem_config(default_type) or {}
        params = list((pconf.get("parameters") or {}).keys())
        families = list(self.optimizer_cfg.keys())
        default_family = families[0] if families else "SciPy"
        default_methods = self.optimizer_cfg.get(default_family, {}).get("methods", [])
        default_method = default_methods[0] if default_methods else "Nelder-Mead"

        self.state = {
            "problem_type": default_type,
            "opt_params": params[:2] if params else [],
            "family": default_family,
            "method": default_method,
            "use_scheduler": True
        }
        # 新增：用户覆盖标记（默认不覆盖）
        self.overrides = {
            "problem_type": False,
            "opt_params": False,
            "family": False,
            "method": False
        }

        # 按钮区
        # 统一样式（避免白底白字，并美化按钮）
        style = ttk.Style(root)
        try:
            style.theme_use('clam')  # macOS默认aqua不支持背景自定义，切换到clam更稳
        except Exception:
            pass

        # 开始优化（绿色）
        style.configure(
            'Start.TButton',
            font=('Arial', 12, 'bold'),
            foreground='#FFFFFF',
            background='#10B981',   # emerald-500
            padding=8
        )
        style.map(
            'Start.TButton',
            background=[('active', '#059669'), ('disabled', '#6EE7B7')],
            foreground=[('disabled', '#F3F4F6')]
        )

        # 中断优化（红色）
        style.configure(
            'Stop.TButton',
            font=('Arial', 12, 'bold'),
            foreground='#FFFFFF',
            background='#EF4444',   # red-500
            padding=8
        )
        style.map(
            'Stop.TButton',
            background=[('active', '#DC2626'), ('disabled', '#FCA5A5')],
            foreground=[('disabled', '#F3F4F6')]
        )

        # 打开设置（蓝色）
        style.configure(
            'Settings.TButton',
            font=('Arial', 12, 'bold'),
            foreground='#FFFFFF',
            background='#3B82F6',   # blue-500
            padding=8
        )
        style.map(
            'Settings.TButton',
            background=[('active', '#2563EB'), ('disabled', '#93C5FD')],
            foreground=[('disabled', '#F3F4F6')]
        )
        # 字体定义（失败回退至Tk默认字体）
        try:
            self.ui_font_title = tkfont.Font(family='Arial', size=13, weight='bold')
            self.ui_font_body = tkfont.Font(family='Arial', size=11)
            self.ui_font_mono = tkfont.Font(family='Menlo' if sys.platform == 'darwin' else 'Courier New', size=11)
            # 左侧输入区域更大字号
            self.ui_font_left = tkfont.Font(family='Arial', size=14)
        except Exception:
            default = tkfont.nametofont('TkDefaultFont')
            self.ui_font_title = tkfont.Font(font=default)
            self.ui_font_title.configure(weight='bold', size=13)
            self.ui_font_body = tkfont.Font(font=default)
            self.ui_font_left = tkfont.Font(font=default)
            self.ui_font_left.configure(size=14)
            self.ui_font_mono = tkfont.nametofont('TkFixedFont')
        self.input_text.configure(
            bg='#FFFFFF',
            fg='#1F2937',
            insertbackground='#111827',
            font=self.ui_font_left,  # 放大输入框字体
            relief='solid',
            bd=1,
            highlightthickness=0
        )
        self.input_text.pack(fill=tk.X, pady=4)

        # 按钮区
        btn_frame = ttk.Frame(left, style='Card.TFrame')
        btn_frame.pack(fill=tk.X, pady=10, padx=8)
        self.btn_start = ttk.Button(btn_frame, text="开始优化", style='Start.TButton', command=self._run_optimize)
        self.btn_start.grid(row=0, column=0, sticky=tk.EW, padx=(0, 6))
        self.btn_stop = ttk.Button(btn_frame, text="中断优化", style='Stop.TButton', command=self._interrupt_optimize)
        self.btn_stop.grid(row=0, column=1, sticky=tk.EW, padx=(6, 0))
        # 设置按钮独占下一行
        self.btn_settings = ttk.Button(btn_frame, text="打开设置", style='Settings.TButton', command=self._open_settings)
        self.btn_settings.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(8, 0))
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.columnconfigure(2, weight=1)

        # 右侧：结果输出文本
        ttk.Label(right, text="分析配置与结果").pack(anchor=tk.W)
        self.result_text = tk.Text(right, height=26)
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=4)

        # 后台运行控制
        self.worker_thread = None
        self.interrupt_requested = False
        # 队列与日志刷新（移除进度条正则）
        self.log_queue = queue.Queue()
        self.root.after(80, self._drain_logs)

    def _open_settings(self):
        """弹出设置窗口，手动调整默认要求（参数、优化器等）"""
        win = tk.Toplevel(self.root)
        win.title("默认设置")
        win.geometry("600x520")

        # 问题类型
        ttk.Label(win, text="问题类型").pack(anchor=tk.W, pady=(8, 0))
        types = self.optimizer.problem_config.get_available_problem_types()
        pt_var = tk.StringVar(value=self.state["problem_type"])
        pt_cb = ttk.Combobox(win, textvariable=pt_var, values=types, state="readonly")
        pt_cb.pack(fill=tk.X, padx=6)

        # 参数多选
        ttk.Label(win, text="优化参数（多选）").pack(anchor=tk.W, pady=(8, 0))
        pconf = self.optimizer.problem_config.get_problem_config(pt_var.get()) or {}
        params = list((pconf.get("parameters") or {}).keys())
        param_list = tk.Listbox(win, selectmode=tk.MULTIPLE, height=10)
        param_list.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        # 填充和默认勾选
        def refresh_params():
            pconf2 = self.optimizer.problem_config.get_problem_config(pt_var.get()) or {}
            ps = list((pconf2.get("parameters") or {}).keys())
            param_list.delete(0, tk.END)
            for p in ps:
                param_list.insert(tk.END, p)
            # 默认选中当前设置
            for i, p in enumerate(ps):
                if p in self.state["opt_params"]:
                    param_list.select_set(i)
        refresh_params()
        pt_cb.bind("<<ComboboxSelected>>", lambda _e: refresh_params())

        # 优化库与方法
        ttk.Label(win, text="优化库（家族）").pack(anchor=tk.W, pady=(8, 0))
        families = list(self.optimizer_cfg.keys())
        fam_var = tk.StringVar(value=self.state["family"])
        fam_cb = ttk.Combobox(win, textvariable=fam_var, values=families, state="readonly")
        fam_cb.pack(fill=tk.X, padx=6)

        ttk.Label(win, text="优化方法").pack(anchor=tk.W, pady=(8, 0))
        methods = self.optimizer_cfg.get(fam_var.get(), {}).get("methods", [])
        meth_var = tk.StringVar(value=self.state["method"])
        meth_cb = ttk.Combobox(win, textvariable=meth_var, values=methods, state="readonly")
        meth_cb.pack(fill=tk.X, padx=6)

        def refresh_methods():
            ms = self.optimizer_cfg.get(fam_var.get(), {}).get("methods", [])
            meth_cb.configure(values=ms)
            if ms and meth_var.get() not in ms:
                meth_var.set(ms[0])
        fam_cb.bind("<<ComboboxSelected>>", lambda _e: refresh_methods())

        # 动态调度
        use_scheduler_var = tk.BooleanVar(value=self.state["use_scheduler"])
        ttk.Checkbutton(win, text="使用动态调度（智能切换算法）", variable=use_scheduler_var).pack(anchor=tk.W, padx=6, pady=(8, 0))

        # 保存/关闭
        action_frame = ttk.Frame(win)
        action_frame.pack(fill=tk.X, pady=10)
        def save_settings():
            # 读取新值
            new_problem_type = pt_var.get()
            new_opt_params = [param_list.get(i) for i in param_list.curselection()]
            new_family = fam_var.get()
            new_method = meth_var.get()

            # 动态调度直接保存，不影响冲突合并
            self.state["use_scheduler"] = use_scheduler_var.get()

            messagebox.showinfo("提示", "设置已保存")
        ttk.Button(action_frame, text="保存", command=save_settings).grid(row=0, column=0, sticky=tk.EW, padx=(0, 6))
        ttk.Button(action_frame, text="关闭", command=win.destroy).grid(row=0, column=1, sticky=tk.EW, padx=(6, 0))
        action_frame.columnconfigure(0, weight=1)
        action_frame.columnconfigure(1, weight=1)

    def _run_optimize(self):
        """后台线程运行优化，避免阻塞界面"""
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("提示", "已有优化任务在运行")
            return
        self.interrupt_requested = False
        self.result_text.delete("1.0", tk.END)
        # 开始优化仅输出简洁提示
        self.result_text.insert(tk.END, "优化开始，请稍候...\n")

        import threading
        self.worker_thread = threading.Thread(target=self._do_optimize, daemon=True)
        self.worker_thread.start()

    def _interrupt_optimize(self):
        """标记中断（无法强制杀死线程，标记后忽略结果）"""
        self.interrupt_requested = True
        self.result_text.insert(tk.END, "\n[提示] 已请求中断，当前计算完成后将忽略结果。\n")

    # 新增：统一合并函数，保证自然语言优先，其次LLM，最后默认；仅在用户覆盖标记为True时才用用户设置
    def _resolve_config(self, user_input: str) -> dict:
        analysis_config = {}
        detected = None
        try:
            detected = self.optimizer.problem_detector.detect_problem_type(
                user_input, self.optimizer.problem_config
            )
            analysis_json = self.optimizer.problem_detector.llm_analyse(
                user_input, self.optimizer.problem_config, self.state["problem_type"]
            )
            analysis_config = json.loads(analysis_json) if analysis_json else {}
        except Exception:
            analysis_config = {}

        # 选择最终问题类型
        if self.overrides["problem_type"]:
            final_problem_type = self.state["problem_type"]
        else:
            final_problem_type = (
                analysis_config.get("problem_type") or
                detected or
                self.state["problem_type"]
            )

        # 选择最终参数
        if self.overrides["opt_params"]:
            final_opt_params = self.state["opt_params"]
        else:
            final_opt_params = (
                analysis_config.get("opt_params") or
                self.state["opt_params"] or
                []
            )

        # 选择最终算法（家族/方法）
        user_alg = analysis_config.get("user_specified_algorithm")
        parsed_family, parsed_method = None, None
        if user_alg and "-" in user_alg:
            parsed_family, parsed_method = user_alg.split("-", 1)

        if self.overrides["family"] or self.overrides["method"]:
            final_family = self.state["family"]
            final_method = self.state["method"]
        else:
            final_family = (
                analysis_config.get("optimizer") or
                parsed_family or
                self.state["family"]
            )
            final_method = (
                analysis_config.get("method") or
                parsed_method or
                self.state["method"]
            )

        return {
            "problem_type": final_problem_type,
            "opt_params": final_opt_params,
            "family": final_family,
            "method": final_method,
            "use_scheduler": self.state["use_scheduler"],
            "user_input": user_input,
            "analysis": analysis_config
        }

    def _latest_result_file(self) -> str:
        """获取最新的结果文件路径"""
        import glob, os
        # 优先从当前工作目录的 results 取
        candidates = glob.glob("results/opt_result_*.json")
        if not candidates:
            # 兼容从 src 目录启动
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidates = glob.glob(os.path.join(base, "results", "opt_result_*.json"))
        if not candidates:
            return ""
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    def _sanitize_logs(self, s: str) -> str:
        """去除日志中的表情符号，保留中文与常规字符"""
        try:
            emoji_pattern = re.compile(
                "[\U0001F300-\U0001F5FF\U0001F600-\U0001F64F"
                "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
                "\U00002700-\U000027BF\U0001F900-\U0001F9FF"
                "\U0001FA70-\U0001FAFF]+",
                flags=re.UNICODE
            )
            return emoji_pattern.sub("", s)
        except Exception:
            return s

    def _do_optimize(self):
        """实际执行优化（后台线程），仅输出日志 + 最优结果、对应参数与保存路径"""
        try:
            user_input = self.input_text.get("1.0", tk.END).strip()

            # 使用合并后的最终配置（避免冲突）
            final_cfg = self._resolve_config(user_input)

            # 构建传入优化器的 analysis_config
            analysis_config = final_cfg.get("analysis") or {}
            analysis_config["problem_type"] = final_cfg["problem_type"]
            analysis_config["user_input"] = final_cfg["user_input"]
            analysis_config["opt_params"] = final_cfg["opt_params"]
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

            # 捕获优化过程的打印日志
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                if final_cfg["use_scheduler"]:
                    result = self.optimizer.optimize_with_scheduler(
                        analysis_config=analysis_config,
                        opt_params=final_cfg["opt_params"],
                        initial_method=final_cfg["method"],
                        initial_optimizer=final_cfg["family"]
                    )
                else:
                    result = self.optimizer.optimize(
                        analysis_config=analysis_config,
                        opt_params=final_cfg["opt_params"],
                        method=final_cfg["method"],
                        optimizer=final_cfg["family"]
                    )

            # 中断标记：忽略输出
            if self.interrupt_requested:
                self.result_text.insert(tk.END, "\n[已中断] 忽略本次结果。\n")
                return

            # 输出清洗后的运行日志（去掉表情符号）
            logs = buf.getvalue()
            clean_logs = self._sanitize_logs(logs)
            if clean_logs.strip():
                self.result_text.insert(tk.END, "[运行日志]\n")
                self.result_text.insert(tk.END, clean_logs.rstrip() + "\n\n")

            # 仅输出最优结果、对应参数与保存路径
            if not result or not result.get("success", False):
                self.result_text.insert(tk.END, f"优化失败: {result.get('message', '未知错误')}\n")
                return

            best_value = result.get("best_value", None)
            best_params = result.get("best_params", {}) or {}
            save_path = self._latest_result_file()

            self.result_text.insert(tk.END, "=== 优化完成 ===\n")
            self.result_text.insert(tk.END, f"最优结果: {best_value}\n")
            self.result_text.insert(tk.END, "对应参数:\n")
            for param, value in best_params.items():
                self.result_text.insert(tk.END, f"  - {param}: {value}\n")
            self.result_text.insert(tk.END, f"保存路径: {save_path if save_path else 'results/opt_result_*.json'}\n")

        except Exception as e:
            self.result_text.insert(tk.END, f"\n[异常] {e}\n")


    def _drain_logs(self):
        """从队列取日志并更新到文本区（仅追加输出，不做同行刷新）"""
        try:
            while not self.log_queue.empty():
                kind, payload = self.log_queue.get()
                # 统一按追加输出处理
                if payload:
                    self._append_log(payload)
        finally:
            self.root.after(80, self._drain_logs)

    def _append_log(self, text: str):
        """保持原有纯追加逻辑"""
        self.result_text.insert(tk.END, text)
        self.result_text.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = AICOApp(root)
    root.mainloop()
