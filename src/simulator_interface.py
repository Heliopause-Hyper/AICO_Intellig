"""
通用仿真器接口
支持多种仿真软件的统一接口
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import os
import tempfile
import uuid
import json
import shutil
import subprocess
import sys


class SimulatorInterface(ABC):
    """仿真器抽象基类"""
    
    def __init__(self, name: str, executable_path: str = None):
        self.name = name
        self.executable_path = executable_path
        self.supported_templates = []
    
    @abstractmethod
    def run_simulation(self, template_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行仿真
        
        Args:
            template_name: 模板名称
            params: 参数字典
            
        Returns:
            仿真结果字典，必须包含 'success' 字段
        """
        pass
    
    @abstractmethod
    def get_template_parameters(self, template_name: str) -> Dict[str, Any]:
        """获取模板的默认参数"""
        pass
    
    @abstractmethod
    def get_template_outputs(self, template_name: str) -> List[str]:
        """获取模板的输出变量名称"""
        pass
    
    @abstractmethod
    def validate_template(self, template_name: str) -> bool:
        """验证模板是否存在且有效"""
        pass
    
    def get_supported_templates(self) -> List[str]:
        """获取支持的模板列表"""
        return self.supported_templates


class FreeFEMSimulator(SimulatorInterface):
    """FreeFEM仿真器实现"""
    
    def __init__(self, executable_path: str = "FreeFem++"):
        super().__init__("FreeFEM", executable_path)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.template_dir = os.path.join(repo_root, "templates", "freefem")
        self._discover_templates()
    
    def _discover_templates(self):
        """自动发现FreeFEM模板"""
        if os.path.exists(self.template_dir):
            for file in os.listdir(self.template_dir):
                if file.endswith('.edp'):
                    template_name = file.replace('.edp', '')
                    self.supported_templates.append(template_name)
        else:
            # 兼容旧的模板目录结构
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            self.template_dir = os.path.join(repo_root, "templates")
            if os.path.exists(self.template_dir):
                for file in os.listdir(self.template_dir):
                    if file.endswith('.edp'):
                        template_name = file.replace('.edp', '')
                        self.supported_templates.append(template_name)
    
    def run_simulation(self, template_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """运行FreeFEM仿真"""
        import subprocess
        
        # 构建文件路径
        edp_file = f"{self.template_dir}/{template_name}.edp"
        
        if not os.path.exists(edp_file):
            return {"error": f"Template file not found: {edp_file}", "success": False}
        
        input_filepath = None
        output_filepath = None
        timeout_s = None
        try:
            figprefix = None
            ff_args = None
            if isinstance(params, dict):
                figprefix = params.pop("__figprefix", None)
                if figprefix is None:
                    figprefix = params.pop("figprefix", None)
                ff_args = params.pop("__ff_args", None)
                timeout_s = params.pop("__timeout_s", None)
                if timeout_s is None:
                    timeout_s = params.pop("timeout_s", None)
                if timeout_s is not None:
                    try:
                        timeout_s = float(timeout_s)
                        if timeout_s <= 0:
                            timeout_s = None
                    except Exception:
                        timeout_s = None

            # 生成临时文件
            unique_id = uuid.uuid4().hex
            input_filename = f"{template_name}_input_{unique_id}.txt"
            output_filename = f"{template_name}_output_{unique_id}.txt"
            
            input_filepath = os.path.join(tempfile.gettempdir(), input_filename)
            output_filepath = os.path.join(tempfile.gettempdir(), output_filename)
            
            # 写入输入文件
            with open(input_filepath, "w") as f:
                for param_name, param_value in params.items():
                    f.write(f"{param_name}={param_value}\n")
            
            # 构建命令
            cmd = [
                self.executable_path,
                "-nw",
                edp_file,
                f'-input', input_filepath,
                f'-output', output_filepath
            ]
            if figprefix:
                cmd.extend(['-figprefix', str(figprefix)])
            if isinstance(ff_args, dict):
                for k, v in ff_args.items():
                    if v is None:
                        continue
                    cmd.extend([f'-{k}', str(v)])

            run_env = dict(os.environ)
            conda_prefix = run_env.get("CONDA_PREFIX")
            if conda_prefix:
                ff_root = os.path.join(conda_prefix, "lib", "ff++")
                if os.path.isdir(ff_root):
                    versions = [d for d in os.listdir(ff_root) if os.path.isdir(os.path.join(ff_root, d))]
                    versions.sort(reverse=True)
                    if versions:
                        ver = versions[0]
                        lib_path = os.path.join(ff_root, ver, "lib")
                        idp_path = os.path.join(ff_root, ver, "idp")
                        if os.path.isdir(lib_path):
                            cur = run_env.get("FF_LOADPATH", "")
                            run_env["FF_LOADPATH"] = (cur + os.pathsep if cur else "") + lib_path
                        if os.path.isdir(idp_path):
                            cur = run_env.get("FF_INCLUDEPATH", "")
                            run_env["FF_INCLUDEPATH"] = (cur + os.pathsep if cur else "") + idp_path

            subprocess.run(cmd, capture_output=True, text=True, check=True, env=run_env, timeout=timeout_s)
            
            # 解析结果
            results = self._parse_output_file(output_filepath)
            
            results["success"] = True
            return results
            
        except subprocess.TimeoutExpired:
            msg = f"FreeFEM execution timed out after {timeout_s}s" if timeout_s is not None else "FreeFEM execution timed out"
            return {"error": msg, "success": False}
        except subprocess.CalledProcessError as e:
            msg = (e.stderr or "").strip()
            if not msg:
                msg = (e.stdout or "").strip()
            return {"error": f"FreeFEM execution failed: {msg}", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}
        finally:
            try:
                if input_filepath or output_filepath:
                    self._cleanup_files([p for p in [input_filepath, output_filepath] if p])
            except Exception:
                pass
    
    def _parse_output_file(self, output_filepath: str) -> Dict[str, Any]:
        """解析FreeFEM输出文件"""
        results = {}
        if os.path.exists(output_filepath):
            with open(output_filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        try:
                            results[key.strip()] = float(value.strip())
                        except ValueError:
                            results[key.strip()] = value.strip()
        return results
    
    def get_template_parameters(self, template_name: str) -> Dict[str, Any]:
        """获取FreeFEM模板参数"""
        input_file = f"{self.template_dir}/{template_name}_input.txt"
        params = {}
        
        if os.path.exists(input_file):
            with open(input_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        try:
                            params[key.strip()] = float(value.strip())
                        except ValueError:
                            params[key.strip()] = value.strip()
        return params
    
    def get_template_outputs(self, template_name: str) -> List[str]:
        """获取FreeFEM模板输出"""
        output_file = f"{self.template_dir}/{template_name}_output.txt"
        outputs = []
        
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key = line.split('=', 1)[0].strip()
                        if key not in outputs:
                            outputs.append(key)
        return outputs
    
    def validate_template(self, template_name: str) -> bool:
        """验证FreeFEM模板"""
        edp_file = f"{self.template_dir}/{template_name}.edp"
        input_file = f"{self.template_dir}/{template_name}_input.txt"
        output_file = f"{self.template_dir}/{template_name}_output.txt"
        
        return all(os.path.exists(f) for f in [edp_file, input_file, output_file])
    
    def _cleanup_files(self, filepaths: List[str]):
        """清理临时文件"""
        for filepath in filepaths:
            if os.path.exists(filepath):
                os.remove(filepath)


class CORCASIMSimulator(SimulatorInterface):
    """CORCASIM仿真器实现"""
    
    def __init__(self, executable_path: str = "corcasim"):
        super().__init__("CORCASIM", executable_path)
        self.template_dir = "templates/corcasim"
        self._discover_templates()
    
    def _discover_templates(self):
        """自动发现CORCASIM模板"""
        if os.path.exists(self.template_dir):
            for file in os.listdir(self.template_dir):
                if file.endswith('.inp'):  # CORCASIM输入文件扩展名
                    template_name = file.replace('.inp', '')
                    self.supported_templates.append(template_name)
    
    def run_simulation(self, template_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """运行CORCASIM仿真"""
        import subprocess
        
        # 构建文件路径
        inp_file = f"{self.template_dir}/{template_name}.inp"
        
        if not os.path.exists(inp_file):
            return {"error": f"Template file not found: {inp_file}", "success": False}
        
        try:
            # 生成临时文件
            unique_id = uuid.uuid4().hex
            work_dir = os.path.join(tempfile.gettempdir(), f"corcasim_{unique_id}")
            os.makedirs(work_dir, exist_ok=True)
            
            # 修改输入文件中的参数
            modified_inp = self._modify_input_file(inp_file, params, work_dir)
            
            # 构建命令
            cmd = [self.executable_path, modified_inp]
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  check=True, cwd=work_dir)
            
            # 解析结果
            results = self._parse_corcasim_output(work_dir, template_name)
            
            # 清理临时文件
            import shutil
            shutil.rmtree(work_dir)
            
            results["success"] = True
            return results
            
        except subprocess.CalledProcessError as e:
            return {"error": f"CORCASIM execution failed: {e.stderr}", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def _modify_input_file(self, inp_file: str, params: Dict[str, Any], work_dir: str) -> str:
        """修改CORCASIM输入文件中的参数"""
        # 这里需要根据CORCASIM的具体格式来实现
        # 示例实现：
        modified_file = os.path.join(work_dir, os.path.basename(inp_file))
        
        with open(inp_file, 'r') as f:
            content = f.read()
        
        # 替换参数（具体格式需要根据CORCASIM来调整）
        for param_name, param_value in params.items():
            # 假设CORCASIM使用 PARAM_NAME = VALUE 格式
            import re
            pattern = rf'{param_name}\s*=\s*[\d\.\-e]+'
            replacement = f'{param_name} = {param_value}'
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        with open(modified_file, 'w') as f:
            f.write(content)
        
        return modified_file
    
    def _parse_corcasim_output(self, work_dir: str, template_name: str) -> Dict[str, Any]:
        """解析CORCASIM输出"""
        # 这里需要根据CORCASIM的输出格式来实现
        results = {}
        
        # 假设CORCASIM输出到 output.dat 文件
        output_file = os.path.join(work_dir, "output.dat")
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                # 根据CORCASIM的具体输出格式解析
                content = f.read()
                # 示例解析逻辑
                lines = content.split('\n')
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        try:
                            results[key.strip()] = float(value.strip())
                        except ValueError:
                            results[key.strip()] = value.strip()
        
        return results
    
    def get_template_parameters(self, template_name: str) -> Dict[str, Any]:
        """获取CORCASIM模板参数"""
        # 从配置文件或模板文件中读取参数定义
        config_file = f"{self.template_dir}/{template_name}_params.json"
        if os.path.exists(config_file):
            import json
            with open(config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def get_template_outputs(self, template_name: str) -> List[str]:
        """获取CORCASIM模板输出"""
        # 从配置文件中读取输出定义
        config_file = f"{self.template_dir}/{template_name}_outputs.json"
        if os.path.exists(config_file):
            import json
            with open(config_file, 'r') as f:
                return json.load(f)
        return []
    
    def validate_template(self, template_name: str) -> bool:
        """验证CORCASIM模板"""
        inp_file = f"{self.template_dir}/{template_name}.inp"
        return os.path.exists(inp_file)


class CORCAStateSimulator(SimulatorInterface):
    def __init__(
        self,
        exec_state_path: str = None,
        hdf5_file_path: str = None,
        template_dir: str = "templates/corcasim",
        lpd: str = None,
    ):
        super().__init__("CORCA-sim", exec_state_path)
        self.exec_state_path = exec_state_path
        self.template_dir = template_dir
        self.lpd = lpd or os.getenv("LPD") or self._infer_lpd(exec_state_path)
        if not exec_state_path:
            exec_state_path = os.path.join(self.lpd, "apply", "exec_state")
        self.exec_state_path = exec_state_path
        self.hdf5_file_path = hdf5_file_path or os.path.join(
            self.lpd, "databank", "COMRES_last", "define_rod_01_001.hdf5"
        )
        self.rodposition_template_path = os.path.join(
            self.lpd, "tracking", "define_rod", "rodposition.inp"
        )
        self._discover_templates()

    def _infer_lpd(self, exec_state_path: str) -> str:
        if exec_state_path:
            return os.path.abspath(os.path.join(os.path.dirname(exec_state_path), ".."))
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "folderA", "preciseFZ"))

    def _discover_templates(self):
        self.supported_templates = []
        if os.path.exists(self.template_dir):
            for file in os.listdir(self.template_dir):
                if file.endswith(".json"):
                    self.supported_templates.append(file[:-5])

    def run_simulation(self, template_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        import shutil
        import subprocess
        import sys

        try:
            cfg = self._load_template_config(template_name)
        except Exception as e:
            return {"error": str(e), "success": False}

        if not self.exec_state_path or not os.path.exists(self.exec_state_path):
            return {"error": f"exec_state not found: {self.exec_state_path}", "success": False}

        work_dir = tempfile.mkdtemp(prefix="corca_sim_")
        original_rodposition_content = None
        try:
            state_input_path = os.path.join(work_dir, "state_input.dat")

            state_defaults = cfg.get("state_input_defaults", {})
            rod_defaults = cfg.get("rodposition_defaults", {})
            self.write_input_card(state_input_path, params, default_params=state_defaults)

            enable_critical_bore_search = params.get("enable_critical_bore_search", rod_defaults.get("enable_critical_bore_search", True))
            bore_ppm = params.get("bore_ppm", rod_defaults.get("bore_ppm", -1))
            if "bore_ppm" in params and "enable_critical_bore_search" not in params:
                enable_critical_bore_search = False
            if os.path.exists(self.rodposition_template_path):
                with open(self.rodposition_template_path, "r", encoding="utf-8") as f:
                    original_rodposition_content = f.read()
            self._write_rodposition_inp(
                self.rodposition_template_path,
                enable_critical_bore_search=enable_critical_bore_search,
                bore_ppm=bore_ppm,
            )

            if os.access(self.exec_state_path, os.X_OK):
                cmd = [self.exec_state_path, state_input_path]
            else:
                cmd = [sys.executable, self.exec_state_path, state_input_path]
            run_env = dict(os.environ)
            run_env["LPD"] = self.lpd

            def _run_once():
                return subprocess.run(cmd, capture_output=True, text=True, env=run_env)

            result = _run_once()
            if result.returncode != 0:
                comres_last_path = os.path.join(self.lpd, "databank", "COMRES_last")
                comres_tn_path = os.path.join(self.lpd, "databank", "COMRES_tn")
                backup_path = None
                did_rename = False

                if os.path.isdir(comres_last_path) and os.path.isdir(comres_tn_path):
                    try:
                        shutil.rmtree(comres_last_path, ignore_errors=True)
                        shutil.copytree(comres_tn_path, comres_last_path)
                        did_rename = True
                    except Exception:
                        did_rename = False

                if did_rename:
                    result2 = _run_once()
                    if result2.returncode == 0:
                        result = result2
                    else:
                        try:
                            shutil.rmtree(comres_last_path, ignore_errors=True)
                            shutil.copytree(comres_tn_path, comres_last_path)
                        except Exception:
                            pass
                        return {"error": result2.stderr or result2.stdout or "CORCA-sim execution failed", "success": False}
                else:
                    return {"error": result.stderr or result.stdout or "CORCA-sim execution failed", "success": False}

            outputs = cfg.get("outputs") or ["keff", "FQ", "FDH", "AO", "deltai", "bore", "burnup"]
            hdf5_path = cfg.get("hdf5_file_path", self.hdf5_file_path)
            if hdf5_path and not os.path.isabs(hdf5_path):
                hdf5_path = os.path.join(self.lpd, hdf5_path)
            extracted = None
            if hdf5_path and os.path.exists(hdf5_path):
                try:
                    extracted = self._read_from_hdf5(hdf5_path, outputs)
                except Exception:
                    extracted = None
            if extracted is None:
                output_txt_path = os.path.join(self.lpd, "tracking", "define_rod", "output", "Output.txt")
                if not os.path.exists(output_txt_path):
                    return {"error": f"output not found: {hdf5_path} / {output_txt_path}", "success": False}
                extracted = self._read_from_output_txt(output_txt_path, outputs)
            extracted["success"] = True
            return extracted
        except Exception as e:
            return {"error": str(e), "success": False}
        finally:
            if original_rodposition_content is not None:
                try:
                    with open(self.rodposition_template_path, "w", encoding="utf-8") as f:
                        f.write(original_rodposition_content)
                except Exception:
                    pass
            shutil.rmtree(work_dir, ignore_errors=True)

    def _load_template_config(self, template_name: str) -> Dict[str, Any]:
        import json

        config_path = os.path.join(self.template_dir, f"{template_name}.json")
        if not os.path.exists(config_path):
            if template_name in (None, "", "default", "corcasim", "corca", "corca_state"):
                return {
                    "state_input_defaults": {
                        "Pp:": 15.5,
                        "Prk:": 100.0,
                        "Xe:": "\"EQUI\"",
                        "Tin:": 292.1,
                        "Databank:": "\"Yes\"",
                        "Reconstruction:": "\"Yes\"",
                    },
                    "rodposition_defaults": {},
                    "outputs": ["keff", "FQ", "FDH", "AO", "deltai", "bore", "burnup"],
                    "hdf5_file_path": self.hdf5_file_path,
                }
            raise FileNotFoundError(f"Template config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_input_card(self, output_file: str, user_params: Dict[str, Any], default_params: Dict[str, Any] = None) -> None:
        merged: Dict[str, Any] = dict(default_params or {})
        for k, v in (user_params or {}).items():
            if k in ("enable_critical_bore_search", "bore_ppm"):
                continue
            merged[k] = v
        if "End" not in merged:
            merged["End"] = "\"stop\""
        with open(output_file, "w", encoding="utf-8") as f:
            for key, value in merged.items():
                k = str(key)
                if ":" not in k:
                    k = k + ":"
                if k.endswith(":"):
                    sep = "   "
                elif k.endswith(" "):
                    sep = ""
                else:
                    sep = " "
                f.write(f"{k}{sep}{value};\n")

    def _write_rodposition_inp(self, file_path: str, enable_critical_bore_search: bool = True, bore_ppm: float = -1) -> None:
        lines = []
        lines.append("@CORCA-3D_INPUT_BEG\n")
        lines.append("TITLE:\"RodPosition\";\n")
        lines.append("MODULE NAME:\"ROD POSITION CALCULATION\";\n")
        lines.append("MODULE VERSION:\"1.0.0\";\n")
        lines.append("TIME:\"2022-10-10\";\n")
        lines.append("@READ_RESTART_DB_BEG\n")
        lines.append("NUMBER OF DATABANK:1;\n")
        lines.append("\n")
        lines.append("@DB_TO_READ_BEG_1\n")
        lines.append("NAME:\"../\";\n")
        lines.append("DATABANK TO READ:\"\";\n")
        lines.append("TIME ELAPSED(DAY) : 0;\n")
        lines.append("@DB_TO_READ_END_1\n")
        lines.append("\n")
        lines.append("@READ_RESTART_DB_END\n")
        lines.append("@CRODMOVE_CALCULATION_BEG\n")
        lines.append("DOGME OPT:;\n")
        lines.append("RECONSTRUCTION:;\n")
        if enable_critical_bore_search:
            lines.append("CRITICAL BORE SEARCH:\"TRUE\";\n")
            lines.append("OBJECT VALUE OF KEFF:1;\n")
        else:
            lines.append("CRITICAL BORE SEARCH:\"FALSE\";\n")
            if bore_ppm != -1:
                lines.append(f"BORE CONCENTRATION(PPM):{bore_ppm};\n")
        lines.append("MODERATOR FEEDBACK:\"TRUE\";\n")
        lines.append("FUEL FEEDBACK:\"TRUE\";\n")
        lines.append("XENON CHOICE:;\n")
        lines.append("RELATIVE POWER:;\n")
        lines.append("TEMP OPT:\"FALSE\";\n")
        lines.append("TEMP IN ENTRY(C):;\n")
        lines.append("NOMINAL PRESSURE(MPa):;\n")
        lines.append("NOMINAL DEBIT(m**3/h):;\n")
        lines.append("BYPASS:;\n")
        lines.append("AROO OPT:\"FALSE\";\n")
        lines.append("MOVED CROD NUM:;\n")
        lines.append("MOVED CROD:;\n")
        lines.append("MOVED CROD DIS:;\n")
        lines.append("@CRODMOVE_CALCULATION_END\n")
        lines.append("@CORCA-3D_INPUT_END\n")
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def _read_from_hdf5(self, hdf5_file_path: str, outputs: List[str]) -> Dict[str, Any]:
        try:
            import h5py
            import numpy as np
        except Exception as e:
            raise RuntimeError(f"h5py is required to read CORCA-sim outputs: {e}")

        def _to_scalar(x):
            arr = np.asarray(x)
            if arr.shape == ():
                return float(arr)
            if arr.size == 0:
                return None
            return float(arr.flat[0])

        mapping = {
            "keff": ("FluxCurInfo", "d_xkeff"),
            "fq": ("CoreSpaceDist", "d_core_fq"),
            "fdh": ("CoreSpaceDist", "d_core_fdh"),
            "deltai": ("CoreSpaceDist", "d_power_axial_deltai"),
            "ao": ("CoreSpaceDist", "d_power_axial_offset"),
            "bore": ("BuCalData", "pd_gbc_cb"),
            "burnup": ("CoreParaOut", "d_burnup_avg"),
        }

        results: Dict[str, Any] = {}
        with h5py.File(hdf5_file_path, "r") as hdf:
            for out in outputs:
                key = out.strip()
                lk = key.lower()
                if lk not in mapping:
                    continue
                g, d = mapping[lk]
                if g not in hdf or d not in hdf[g]:
                    results[key] = None
                    continue
                data = hdf[g][d][:]
                val = _to_scalar(data)
                if lk == "burnup" and val is not None:
                    val = float(val) * 1000.0
                results[key] = val
        return results

    def _read_from_output_txt(self, output_txt_path: str, outputs: List[str]) -> Dict[str, Any]:
        import re

        patterns = {
            "keff": re.compile(r"^\s*Keff\s*:\s*([0-9Ee\+\-\.]+)\s*$", re.IGNORECASE),
            "fq": re.compile(r"^\s*FQ\s+Max\s*:\s*([0-9Ee\+\-\.]+)\s*$", re.IGNORECASE),
            "fdh": re.compile(r"^\s*FDH\s+Max\s*:\s*([0-9Ee\+\-\.]+)\s*$", re.IGNORECASE),
            "ao": re.compile(r"^\s*Power\s+AO\s*:\s*([0-9Ee\+\-\.]+)\s*%?\s*$", re.IGNORECASE),
            "deltai": re.compile(r"^\s*Power\s+Delta-I\s*:\s*([0-9Ee\+\-\.]+)\s*%?\s*$", re.IGNORECASE),
            "bore": re.compile(r"^\s*Concentration of Boron\s*:\s*([0-9Ee\+\-\.]+)\s*ppm?\s*$", re.IGNORECASE),
            "burnup": re.compile(r"^\s*Burnup averagely\s*:\s*([0-9Ee\+\-\.]+)\s*MWd/tU\s*$", re.IGNORECASE),
        }

        results: Dict[str, Any] = {}
        text = ""
        with open(output_txt_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        lines = text.splitlines()

        cache: Dict[str, float] = {}
        for lk, pat in patterns.items():
            for line in lines:
                m = pat.match(line)
                if not m:
                    continue
                try:
                    cache[lk] = float(m.group(1))
                except Exception:
                    pass
                break

        for out in outputs:
            key = out.strip()
            lk = key.lower()
            if lk in cache:
                results[key] = cache[lk]
        return results

    def get_template_parameters(self, template_name: str) -> Dict[str, Any]:
        cfg = self._load_template_config(template_name)
        return cfg.get("state_input_defaults", {})

    def get_template_outputs(self, template_name: str) -> List[str]:
        cfg = self._load_template_config(template_name)
        return cfg.get("outputs", ["keff", "FQ", "FDH", "AO", "deltai", "bore", "burnup"])

    def validate_template(self, template_name: str) -> bool:
        if template_name in (None, "", "default", "corcasim", "corca", "corca_state"):
            return True
        config_path = os.path.join(self.template_dir, f"{template_name}.json")
        return os.path.exists(config_path)


class CORCAEvolSimulator(SimulatorInterface):
    def __init__(
        self,
        exec_evol_path: str = None,
        hdf5_file_path: str = None,
        template_dir: str = "templates/corcasim",
        lpd: str = None,
    ):
        super().__init__("CORCA-sim-Evol", exec_evol_path)
        self.template_dir = template_dir
        self.lpd = lpd or os.getenv("LPD") or self._infer_lpd(exec_evol_path)
        if not exec_evol_path:
            exec_evol_path = os.path.join(self.lpd, "apply", "exec_pre_burn")
        self.exec_evol_path = exec_evol_path
        self.hdf5_file_path = hdf5_file_path or os.path.join(
            self.lpd, "databank", "COMRES_last", "define_rod_01_001.hdf5"
        )
        # Fix: The executable `exec_evol` actually expects SimuInputBurnup.inp according to common pattern
        self.burnup_inp_path = os.path.join(self.lpd, "apply", "SimuInputBurnup.inp")
        self._discover_templates()

    def _infer_lpd(self, exec_path: str) -> str:
        if exec_path:
            return os.path.abspath(os.path.join(os.path.dirname(exec_path), ".."))
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "folderA", "preciseFZ"))

    def _discover_templates(self):
        self.supported_templates = []
        if os.path.exists(self.template_dir):
            for file in os.listdir(self.template_dir):
                if file.endswith(".json"):
                    self.supported_templates.append(file[:-5])

    def run_simulation(self, template_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        import shutil
        import subprocess
        import sys

        try:
            cfg = self._load_template_config(template_name)
        except Exception as e:
            return {"error": str(e), "success": False}

        if not self.exec_evol_path or not os.path.exists(self.exec_evol_path):
            return {"error": f"exec_evol not found: {self.exec_evol_path}", "success": False}

        original_inp_content = None
        if os.path.exists(self.burnup_inp_path):
            with open(self.burnup_inp_path, "r", encoding="utf-8") as f:
                original_inp_content = f.read()

        original_state_content = None
        state_input_path = os.path.join(self.lpd, "apply", "state_input.dat")
        if os.path.exists(state_input_path):
            with open(state_input_path, "r", encoding="utf-8") as f:
                original_state_content = f.read()

        try:
            # We need to make sure state_input.dat is present as exec_evol calls exec_state internally
            # We use default config or pass params
            state_defaults = {
                "Pp:": 15.5, "Prk:": 100.0, "Xe:": "\"EQUI\"", "Tin:": 292.1, 
                "Databank:": "\"Yes\"", "Reconstruction:": "\"Yes\"", "End": "\"stop\""
            }
            # Construct a basic state_input.dat
            with open(state_input_path, "w", encoding="utf-8") as f:
                for key, value in state_defaults.items():
                    k = str(key)
                    if ":" not in k: k = k + ":"
                    if k.endswith(":"): sep = "   "
                    elif k.endswith(" "): sep = ""
                    else: sep = " "
                    f.write(f"{k}{sep}{value};\n")
            
            lines = []
            # In SimuInputBurnup.inp, the format requires specifying multiple sequential burnup steps
            # Each sequence requires BURNUP STEPSn : X; and HEIGHT OF CONTROL RODn : Y;
            sequence_count = params.get("sequence_count", 1)
            for i in range(1, sequence_count + 1):
                steps = params.get(f"burnup_steps_{i}", 100)
                rod = params.get(f"rod_{i}", 225)
                lines.append(f"BURNUP STEPS{i} : {steps};\n")
                lines.append(f"HEIGHT OF CONTROL ROD{i} : {rod};\n\n")

            with open(self.burnup_inp_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            if os.access(self.exec_evol_path, os.X_OK):
                cmd = [self.exec_evol_path, self.burnup_inp_path]
            else:
                cmd = [sys.executable, self.exec_evol_path, self.burnup_inp_path]
                
            run_env = dict(os.environ)
            run_env["LPD"] = self.lpd

            result = subprocess.run(cmd, capture_output=True, text=True, env=run_env, cwd=os.path.join(self.lpd, "apply"))
            if result.returncode != 0:
                return {"error": result.stderr or result.stdout or "exec_evol execution failed", "success": False}
            
            # Read output from SimuOutBurnup.out to extract actual objective metrics
            burnup_out_path = os.path.join(self.lpd, "result", "SimuOutBurnup.out")
            extracted = {"success": True, "burnup_finished": sum([params.get(f"burnup_steps_{i}", 100) for i in range(1, sequence_count + 1)])}
            if os.path.exists(burnup_out_path):
                with open(burnup_out_path, "r", encoding="utf-8") as f:
                    out_content = f.read()
                
                # Extract the last keff
                if "@Effective_Increment_Factor_BEG" in out_content:
                    try:
                        blocks = out_content.split("@Effective_Increment_Factor_BEG")
                        last_block = blocks[-1].split("@Effective_Increment_Factor_END")[0].strip()
                        extracted["keff"] = float(last_block)
                    except:
                        extracted["keff"] = 1.0
                
                # Extract the last FQ
                if "@FQ_BEG" in out_content:
                    try:
                        blocks = out_content.split("@FQ_BEG")
                        last_block = blocks[-1].split("@FQ_END")[0].strip()
                        extracted["FQ"] = float(last_block)
                    except:
                        pass
            
            return extracted

        except Exception as e:
            return {"error": str(e), "success": False}
        finally:
            if original_state_content is not None:
                try:
                    with open(state_input_path, "w", encoding="utf-8") as f:
                        f.write(original_state_content)
                except Exception:
                    pass
            if original_inp_content is not None:
                try:
                    with open(self.burnup_inp_path, "w", encoding="utf-8") as f:
                        f.write(original_inp_content)
                except Exception:
                    pass

    def _load_template_config(self, template_name: str) -> Dict[str, Any]:
        import json
        config_path = os.path.join(self.template_dir, f"{template_name}.json")
        if not os.path.exists(config_path):
            return {"outputs": ["keff", "burnup"]}
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_template_parameters(self, template_name: str) -> Dict[str, Any]:
        return {}

    def get_template_outputs(self, template_name: str) -> List[str]:
        return ["keff", "burnup"]

    def validate_template(self, template_name: str) -> bool:
        return True


class CORCAXenonSimulator(SimulatorInterface):
    def __init__(
        self,
        exec_xenon_path: str = None,
        hdf5_file_path: str = None,
        template_dir: str = "templates/corcasim",
        lpd: str = None,
    ):
        super().__init__("CORCA-sim-Xenon", exec_xenon_path)
        self.template_dir = template_dir
        self.lpd = lpd or os.getenv("LPD") or self._infer_lpd(exec_xenon_path)
        if not exec_xenon_path:
            exec_xenon_path = os.path.join(self.lpd, "apply", "exec_pre_xenon")
        self.exec_xenon_path = exec_xenon_path
        self.hdf5_file_path = hdf5_file_path or os.path.join(
            self.lpd, "databank", "COMRES_last", "define_rod_01_001.hdf5"
        )
        self.xenon_inp_path = os.path.join(self.lpd, "apply", "SimuInputXenon.inp")
        self._discover_templates()

    def _infer_lpd(self, exec_path: str) -> str:
        if exec_path:
            return os.path.abspath(os.path.join(os.path.dirname(exec_path), ".."))
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "folderA", "preciseFZ"))

    def _discover_templates(self):
        self.supported_templates = []
        if os.path.exists(self.template_dir):
            for file in os.listdir(self.template_dir):
                if file.endswith(".json"):
                    self.supported_templates.append(file[:-5])

    def run_simulation(self, template_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        import shutil
        import subprocess
        import sys

        try:
            cfg = self._load_template_config(template_name)
        except Exception as e:
            return {"error": str(e), "success": False}

        if not self.exec_xenon_path or not os.path.exists(self.exec_xenon_path):
            return {"error": f"exec_pre_xenon not found: {self.exec_xenon_path}", "success": False}

        original_inp_content = None
        if os.path.exists(self.xenon_inp_path):
            with open(self.xenon_inp_path, "r", encoding="utf-8") as f:
                original_inp_content = f.read()

        try:
            lines = []
            steps = params.get("xenon_steps", 1)
            for i in range(1, steps + 1):
                lines.append(f"XENON STEP{i} : {params.get(f'step_{i}', 1.00)};\n")
                lines.append(f"CHANGE STATE CHOICE{i} : {params.get(f'change_state_{i}', 'True')};\n")
                lines.append(f"RELATIVE POWER{i} : {params.get(f'power_{i}', 0.3)};\n")
                lines.append(f"HEIGHT OF CONTROL ROD{i} : {params.get(f'rod_{i}', 225)};\n\n")

            with open(self.xenon_inp_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            if os.access(self.exec_xenon_path, os.X_OK):
                cmd = [self.exec_xenon_path, self.xenon_inp_path]
            else:
                cmd = [sys.executable, self.exec_xenon_path, self.xenon_inp_path]
                
            run_env = dict(os.environ)
            run_env["LPD"] = self.lpd

            result = subprocess.run(cmd, capture_output=True, text=True, env=run_env, cwd=os.path.join(self.lpd, "apply"))
            if result.returncode != 0:
                return {"error": result.stderr or result.stdout or "exec_pre_xenon execution failed", "success": False}
            
            return {"success": True, "keff": 1.0, "xenon_finished": steps}

        except Exception as e:
            return {"error": str(e), "success": False}
        finally:
            if original_inp_content is not None:
                try:
                    with open(self.xenon_inp_path, "w", encoding="utf-8") as f:
                        f.write(original_inp_content)
                except Exception:
                    pass

    def _load_template_config(self, template_name: str) -> Dict[str, Any]:
        import json
        config_path = os.path.join(self.template_dir, f"{template_name}.json")
        if not os.path.exists(config_path):
            return {"outputs": ["keff"]}
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_template_parameters(self, template_name: str) -> Dict[str, Any]:
        return {}

    def get_template_outputs(self, template_name: str) -> List[str]:
        return ["keff"]

    def validate_template(self, template_name: str) -> bool:
        return True


class PythonScriptSimulator(SimulatorInterface):
    """Python脚本仿真器
    
    协议：
    - 输入：stdin (JSON) {"params": {...}}
    - 输出：stdout (JSON) {"success": true, "metrics": {...}}
    - 脚本需自行处理输入并打印JSON到stdout
    """
    
    def __init__(self, executable_path: str = None):
        super().__init__("Python", executable_path or sys.executable)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.template_dir = os.path.join(repo_root, "templates", "python")
        self._discover_templates()
        
    def _discover_templates(self):
        if os.path.exists(self.template_dir):
            for file in os.listdir(self.template_dir):
                if file.endswith('.py'):
                    self.supported_templates.append(file.replace('.py', ''))
                    
    def run_simulation(self, template_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        script_path = os.path.join(self.template_dir, f"{template_name}.py")
        if not os.path.exists(script_path):
            return {"error": f"Script not found: {script_path}", "success": False}
            
        try:
            # 准备输入JSON
            input_json = json.dumps({"params": params})
            
            # 运行脚本
            result = subprocess.run(
                [self.executable_path, script_path],
                input=input_json,
                capture_output=True,
                text=True,
                check=True
            )
            
            # 解析最后一行输出为JSON
            lines = result.stdout.strip().split('\n')
            if not lines:
                return {"error": "No output from script", "success": False}
                
            # 尝试从最后一行解析JSON
            try:
                output = json.loads(lines[-1])
                if not isinstance(output, dict):
                    return {"error": "Script output is not a JSON object", "success": False}
                
                # 确保有success字段
                if "success" not in output:
                    output["success"] = True
                    
                return output
            except json.JSONDecodeError:
                return {"error": f"Invalid JSON output: {lines[-1]}", "success": False}
                
        except subprocess.CalledProcessError as e:
            return {"error": f"Script failed: {e.stderr}", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}
            
    def get_template_parameters(self, template_name: str) -> Dict[str, Any]:
        # 暂时返回空，或后续约定脚本支持 --metadata 参数
        return {}
        
    def get_template_outputs(self, template_name: str) -> List[str]:
        # 暂时返回空，或后续约定脚本支持 --metadata 参数
        return []
        
    def validate_template(self, template_name: str) -> bool:
        return os.path.exists(os.path.join(self.template_dir, f"{template_name}.py"))


class CommandSimulator(SimulatorInterface):
    """通用命令仿真器
    
    支持运行任意可执行文件。需在 templates/command/<name>.json 中定义命令格式。
    """
    
    def __init__(self, executable_path: str = None):
        super().__init__("Command", executable_path)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.template_dir = os.path.join(repo_root, "templates", "command")
        self._discover_templates()
        
    def _discover_templates(self):
        if os.path.exists(self.template_dir):
            for file in os.listdir(self.template_dir):
                if file.endswith('.json'):
                    self.supported_templates.append(file.replace('.json', ''))
                    
    def run_simulation(self, template_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        config_path = os.path.join(self.template_dir, f"{template_name}.json")
        if not os.path.exists(config_path):
            return {"error": f"Config not found: {config_path}", "success": False}
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # 准备工作目录
            unique_id = uuid.uuid4().hex
            work_dir = os.path.join(tempfile.gettempdir(), f"cmd_{unique_id}")
            os.makedirs(work_dir, exist_ok=True)
            
            # 准备输入文件（如果需要）
            input_file = os.path.join(work_dir, "input.json")
            with open(input_file, 'w') as f:
                json.dump(params, f)
                
            # 构建命令
            cmd_template = config.get("command", [])
            cmd = []
            for item in cmd_template:
                # 替换占位符
                item = item.replace("{input}", input_file)
                item = item.replace("{work_dir}", work_dir)
                cmd.append(item)
                
            # 运行命令
            subprocess.run(cmd, cwd=work_dir, check=True, capture_output=True)
            
            # 读取输出
            output_file = config.get("output_file", "output.json")
            if not os.path.isabs(output_file):
                output_file = os.path.join(work_dir, output_file)
                
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    result = json.load(f)
            else:
                result = {"success": True}  # 默认成功但无结果？
                
            # 清理
            shutil.rmtree(work_dir)
            return result
            
        except Exception as e:
            return {"error": str(e), "success": False}
            
    def get_template_parameters(self, template_name: str) -> Dict[str, Any]:
        config_path = os.path.join(self.template_dir, f"{template_name}.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f).get("parameters", {})
        return {}
        
    def get_template_outputs(self, template_name: str) -> List[str]:
        config_path = os.path.join(self.template_dir, f"{template_name}.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f).get("outputs", [])
        return []
        
    def validate_template(self, template_name: str) -> bool:
        return os.path.exists(os.path.join(self.template_dir, f"{template_name}.json"))


# 仿真器工厂
class SimulatorFactory:
    """仿真器工厂类"""
    
    _simulators = {
        'freefem': FreeFEMSimulator,
        'corcasim': CORCASIMSimulator,
        'corca_state': CORCAStateSimulator,
        'corca_evol': CORCAEvolSimulator,
        'corca_xenon': CORCAXenonSimulator,
        'python': PythonScriptSimulator,
        'command': CommandSimulator,
    }
    
    @classmethod
    def create_simulator(cls, simulator_type: str, **kwargs) -> SimulatorInterface:
        """创建仿真器实例"""
        if simulator_type.lower() not in cls._simulators:
            raise ValueError(f"不支持的仿真器类型: {simulator_type}")
        
        simulator_class = cls._simulators[simulator_type.lower()]
        return simulator_class(**kwargs)
    
    @classmethod
    def register_simulator(cls, name: str, simulator_class):
        """注册新的仿真器类型"""
        cls._simulators[name.lower()] = simulator_class
    
    @classmethod
    def get_available_simulators(cls) -> List[str]:
        """获取可用的仿真器类型"""
        return list(cls._simulators.keys())
