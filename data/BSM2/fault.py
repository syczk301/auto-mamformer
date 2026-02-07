# -*- coding: utf-8 -*-
"""
BSM2 故障数据集生成器
===================

基于 bsm2-python 包生成 12 种故障类型的仿真数据集。

故障类型对照表：
| No. | 故障类型描述                                    | 仿真细节                                                    |
|-----|------------------------------------------------|-------------------------------------------------------------|
|   1 | 自养菌最大比增长速率(μₐ)降低                    | 第3天从0.5→0.45 d⁻¹；第3-5天线性降至0.4 d⁻¹                 |
|   2 | 异养菌最大比增长速率(μₕ)降低                    | 第3天从4→3 d⁻¹；第3-5天线性降至1.2 d⁻¹                      |
|   3 | 二沉池沉降速度(νₛ)降低                         | 第3天从250→150 m·d⁻¹；第3-5天线性降至80 m·d⁻¹               |
|   4 | 反应器5溶解氧(DO)设定点变化                     | 第3天从2→2.2 g·m⁻³                                          |
|   5 | 内回流执行器漂移                                | 第3天增至基准+8000；第3-14天线性增至基准+12000 m³·d⁻¹       |
|   6 | 降雨干扰                                        | 使用 raininfluent.csv 进水数据                              |
|   7 | 暴风雨干扰                                      | 使用加强的降雨进水数据（流量×1.5）                           |
|   8 | 反应器5 DO传感器漂移                            | 第3-14天从0线性漂移至-1.3 g·m⁻³                             |
|   9 | 反应器5 DO传感器偏置                            | 第3天偏置突变为-0.3 g·m⁻³                                   |
|  10 | 反应器5 DO传感器故障                            | 第3天起传感器读数固定为1.8 g·m⁻³                            |
|  11 | 剩余活性污泥(WAS)流量执行器漂移                 | 第3-14天线性漂移至标称值的1.8倍                             |
|  12 | 浓缩机性能下降(TSS去除效率降低)                 | 第3-14天TSS去除效率线性降至标称值的60%               |

安装依赖:
    pip install bsm2-python numpy pandas

使用方法:
    python fault.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

from bsm2_python.bsm2_cl import BSM2CL

# ============================================================================
# ASM1 参数索引定义（bsm2-python 未导出这些常量，根据标准 ASM1 模型定义）
# ============================================================================
MU_H = 0   # 异养菌最大比增长速率 μH_max 在 asm1par 数组中的索引，标称值 = 4.0 d⁻¹
MU_A = 3   # 自养菌最大比增长速率 μA_max 在 asm1par 数组中的索引，标称值 = 0.5 d⁻¹

# ASM1 状态变量索引
SO_INDEX = 7  # 溶解氧 SO 在状态向量中的索引


# ============================================================================
# 辅助函数
# ============================================================================
def linear_ramp(t: float, t0: float, t1: float, y0: float, y1: float) -> float:
    """
    分段线性变化函数。
    
    Args:
        t: 当前时间
        t0: 起始时间
        t1: 结束时间
        y0: 起始值
        y1: 结束值
    
    Returns:
        在 [t0, t1] 区间内线性插值的结果
    """
    if t <= t0:
        return y0
    if t >= t1:
        return y1
    return y0 + (y1 - y0) * (t - t0) / (t1 - t0)


def get_package_data_path(filename: str) -> Path:
    """获取 bsm2-python 包内置数据文件的路径。"""
    import bsm2_python
    return Path(bsm2_python.__file__).parent / "data" / filename


def load_influent_csv(filepath: Path) -> np.ndarray:
    """
    加载进水 CSV 文件，自动修复格式错误。
    
    bsm2-python 包的 raininfluent.csv 存在数据格式问题（如 '30.044.50'），
    此函数会尝试修复这些问题。
    
    Args:
        filepath: CSV 文件路径
    
    Returns:
        修复后的 numpy 数组
    """
    import csv
    
    data_rows = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row_idx, row in enumerate(reader):
            fixed_row = []
            for col_idx, val in enumerate(row):
                try:
                    fixed_row.append(float(val))
                except ValueError:
                    # 尝试修复格式错误（如 '30.044.50' -> 取第一个有效数字）
                    # 这种错误通常是两个数字意外拼接
                    parts = val.split('.')
                    if len(parts) >= 3:
                        # 假设格式是 "XX.XX" + "X.XX" 拼接，取前两部分
                        # bsm2-python 包的 raininfluent.csv 有已知格式错误，静默修复
                        fixed_val = float(f"{parts[0]}.{parts[1]}")
                        fixed_row.append(fixed_val)
                    else:
                        raise ValueError(f"无法解析: 行{row_idx} 列{col_idx} 值'{val}'")
            data_rows.append(fixed_row)
    
    return np.array(data_rows)


def ensure_data_length(
    data: np.ndarray, 
    end_day: float
) -> np.ndarray:
    """
    确保进水数据覆盖所需的仿真时长。
    如果数据长度不足，将通过循环拼接的方式延长数据。
    
    Args:
        data: 原始进水数据 (time, vars)
        end_day: 仿真结束时间
    
    Returns:
        延长后的数据
    """
    if data.shape[0] < 2:
        return data
        
    t_start = data[0, 0]
    t_end = data[-1, 0]
    duration = t_end - t_start
    
    if duration <= 0:
        return data
        
    last_time = t_end
    current_data = data
    
    # 如果数据覆盖时间不足 end_day，则进行循环拼接
    while last_time < end_day + 1.0:  # 多留一点余量
        # 复制除了第一行以外的所有数据（避免时间点重复）
        next_chunk = data[1:].copy()
        
        # 平移时间轴
        next_chunk[:, 0] = next_chunk[:, 0] - t_start + last_time
        
        # 拼接
        current_data = np.vstack([current_data, next_chunk])
        
        # 更新状态
        last_time = current_data[-1, 0]
    
    return current_data


# ============================================================================
# 故障类型枚举
# ============================================================================
class FaultID(str, Enum):
    """12种故障类型标识符。"""
    F01_MU_A_REDUCED = "F01_mu_a_reduced"
    F02_MU_H_REDUCED = "F02_mu_h_reduced"
    F03_SETTLING_V_REDUCED = "F03_settling_velocity_reduced"
    F04_DO_SETPOINT_STEP = "F04_do_setpoint_step"
    F05_INTERNAL_RECYCLE_DRIFT = "F05_internal_recycle_drift"
    F06_RAINY_INFLUENT = "F06_rainy_influent"
    F07_STORMY_INFLUENT = "F07_stormy_influent"
    F08_DO_SENSOR_DRIFT = "F08_do_sensor_drift"           # 正向漂移→曝气减少
    F09_DO_SENSOR_OFFSET = "F09_do_sensor_offset"         # 负向偏置→曝气增加
    F10_DO_SENSOR_FAILURE = "F10_do_sensor_failure"       # 卡死低值→过度曝气
    F11_WAS_FLOW_DRIFT = "F11_was_flow_drift"
    F12_THICKENER_TSS_REMOVAL_DROP = "F12_thickener_drop"


# ============================================================================
# 故障配置数据类
# ============================================================================
@dataclass
class FaultConfig:
    """
    故障注入配置参数。
    
    所有参数值均基于文献中的标准 BSM1/BSM2 故障诊断协议。
    """
    fault_id: FaultID
    
    # 故障注入时间窗口（天）
    start_day: float = 3.0      # 故障开始时间
    ramp_end_day: float = 5.0   # 渐变故障的结束时间（用于 F01-F03）
    end_day: float = 609.0       # 仿真结束时间（进水数据范围需匹配或循环）
    
    # F01: 自养菌增长速率 μA 故障参数（轻微降低：-20%）
    mu_a_base: float = 0.5      # 基准值 0.5 d⁻¹
    mu_a_day3: float = 0.45     # 第3天突降至 0.45 d⁻¹ (-10%)
    mu_a_day5: float = 0.4      # 第5天降至 0.4 d⁻¹ (-20%)
    
    # F02: 异养菌增长速率 μH 故障参数（中等降低：-70%）
    mu_h_base: float = 4.0      # 基准值 4.0 d⁻¹
    mu_h_day3: float = 3.0      # 第3天突降至 3.0 d⁻¹ (-25%)
    mu_h_day5: float = 1.2      # 第5天降至 1.2 d⁻¹ (-70%)
    
    # F03: 沉降速度 v0_max 故障参数（中等降低：-68%）
    v0max_base: float = 250.0   # 基准值 250 m·d⁻¹
    v0max_day3: float = 150.0   # 第3天降至 150 m·d⁻¹ (-40%)
    v0max_day5: float = 80.0    # 第5天降至 80 m·d⁻¹ (-68%)
    
    # F04: DO 设定点故障参数（轻微升高：+10%）
    so4_base: float = 2.0       # 基准 DO 设定点 2.0 g·m⁻³
    so4_step: float = 2.2       # 第3天升至 2.2 g·m⁻³ (+10%)
    
    # F05: 内回流漂移参数（绝对值增加）
    qintr_drift_day3: float = 8000.0    # 第3天增至基准+8000 m³·d⁻¹
    qintr_drift_day14: float = 12000.0  # 第14天增至基准+12000 m³·d⁻¹
    
    # F08-F10: DO 传感器故障参数
    do_drift_end: float = -1.3   # F08: 漂移至 -1.3 g·m⁻³
    do_offset: float = -0.3      # F09: 偏置 -0.3 g·m⁻³
    do_fail_value: float = 1.8   # F10: 卡死在 1.8 g·m⁻³
    
    # F11: WAS 流量漂移参数（中等增加：+80%）
    was_scale_day3: float = 1.0  # 第3天：1.0倍（正常）
    was_scale_day14: float = 1.8 # 第14天：1.8倍 (+80%)
    
    # F12: 浓缩机 TSS 去除效率下降参数（中等降低：-40%）
    thick_tss_scale_day3: float = 1.0   # 第3天：100%效率
    thick_tss_scale_day14: float = 0.6  # 第14天：60%效率 (-40%)


# ============================================================================
# 带故障注入的 BSM2 闭环控制类
# ============================================================================
class FaultyBSM2CL(BSM2CL):
    """
    继承 BSM2CL，通过重写 step() 方法实现故障注入。
    
    支持的故障注入机制：
    - 动力学参数修改（F01-F02: μA, μH）
    - 沉淀池参数修改（F03: v0_max, F11: q_w）
    - 控制系统干扰（F04: DO设定点, F05: 内回流）
    - 传感器故障（F08-F10: DO传感器漂移/偏置/失效）
    - 浓缩机故障（F12: TSS去除效率）
    """

    def __init__(self, *args, fault: Optional[FaultConfig] = None, **kwargs):
        """
        初始化带故障注入的 BSM2 模型。
        
        Args:
            fault: 故障配置，None 表示正常运行
            *args, **kwargs: 传递给 BSM2CL 的参数
        """
        super().__init__(*args, **kwargs)
        self.fault = fault
        
        # 缓存基准值，用于故障注入时的参考
        self._baseline: Dict[str, Any] = {}
        self._cache_baseline_values()
        
        # 收集所有反应器引用（reactor1-5）
        self._reactors: List[Any] = []
        for name in ["reactor1", "reactor2", "reactor3", "reactor4", "reactor5"]:
            reactor = getattr(self, name, None)
            if reactor is not None and hasattr(reactor, "asm1par"):
                self._reactors.append(reactor)
        
        # 用于记录时序数据的列表
        self._kla4_history: List[float] = []
        self._do_sensor_history: List[float] = []

    def _cache_baseline_values(self):
        """缓存所有可能被故障修改的参数的基准值。"""
        # 沉淀池参数
        if hasattr(self, "settler") and self.settler is not None:
            self._baseline["v0max"] = float(self.settler.sedpar[0])
            self._baseline["q_w"] = float(self.settler.q_w)
        
        # 内回流流量
        if hasattr(self, "qintr"):
            self._baseline["qintr"] = float(self.qintr)
        
        # 浓缩机参数
        if hasattr(self, "thickener") and self.thickener is not None:
            self._baseline["thick_tss"] = float(self.thickener.t_par[1])

    def _in_fault_window(self, t_day: float) -> bool:
        """检查当前时间是否在故障注入窗口内。"""
        if self.fault is None:
            return False
        return t_day >= self.fault.start_day

    def _apply_kinetic_faults(self, t_day: float):
        """
        应用动力学参数故障（F01: μA降低, F02: μH降低）。
        
        修改所有反应器的 asm1par 数组中对应的参数值。
        """
        if self.fault is None:
            return
        
        t0 = self.fault.start_day
        t1 = self.fault.ramp_end_day
        
        # F01: 自养菌增长速率降低
        if self.fault.fault_id == FaultID.F01_MU_A_REDUCED:
            if t_day < t0:
                mu_a = self.fault.mu_a_base
            else:
                # 第3天突降至 day3 值，然后线性降至 day5 值
                mu_a = linear_ramp(t_day, t0, t1, 
                                   self.fault.mu_a_day3, 
                                   self.fault.mu_a_day5)
            for reactor in self._reactors:
                reactor.asm1par[MU_A] = mu_a
        
        # F02: 异养菌增长速率降低
        if self.fault.fault_id == FaultID.F02_MU_H_REDUCED:
            if t_day < t0:
                mu_h = self.fault.mu_h_base
            else:
                mu_h = linear_ramp(t_day, t0, t1,
                                   self.fault.mu_h_day3,
                                   self.fault.mu_h_day5)
            for reactor in self._reactors:
                reactor.asm1par[MU_H] = mu_h

    def _apply_settler_faults(self, t_day: float):
        """
        应用沉淀池相关故障（F03: 沉降速度降低, F11: WAS流量漂移）。
        """
        if self.fault is None or self.settler is None:
            return
        
        t0 = self.fault.start_day
        t1 = self.fault.ramp_end_day
        
        # F03: 沉降速度降低
        if self.fault.fault_id == FaultID.F03_SETTLING_V_REDUCED:
            if t_day < t0:
                v0max = self.fault.v0max_base
            else:
                v0max = linear_ramp(t_day, t0, t1,
                                    self.fault.v0max_day3,
                                    self.fault.v0max_day5)
            self.settler.sedpar[0] = v0max
        
        # F11: WAS 流量执行器漂移
        if self.fault.fault_id == FaultID.F11_WAS_FLOW_DRIFT:
            qw_base = self._baseline.get("q_w", 300.0)
            if t_day < t0:
                scale = 1.0
            else:
                scale = linear_ramp(t_day, t0, self.fault.end_day,
                                    self.fault.was_scale_day3,
                                    self.fault.was_scale_day14)
            self.settler.q_w = qw_base * scale

    def _apply_recycle_faults(self, t_day: float):
        """
        应用内回流故障（F05: 内回流执行器漂移）。
        
        优化后使用比例因子：第3天增至1.5倍，第14天增至2.0倍。
        """
        if self.fault is None:
            return
        if self.fault.fault_id != FaultID.F05_INTERNAL_RECYCLE_DRIFT:
            return

        t0 = self.fault.start_day
        qintr_base = self._baseline.get("qintr", 61944.0)

        if t_day < t0:
            drift = 0.0
        else:
            drift = linear_ramp(t_day, t0, self.fault.end_day,
                                self.fault.qintr_drift_day3,
                                self.fault.qintr_drift_day14)

        self.qintr = qintr_base + drift

    def _apply_thickener_faults(self, t_day: float):
        """
        应用浓缩机故障（F12: TSS去除效率下降）。
        """
        if self.fault is None or self.thickener is None:
            return
        if self.fault.fault_id != FaultID.F12_THICKENER_TSS_REMOVAL_DROP:
            return
        
        t0 = self.fault.start_day
        tss_base = self._baseline.get("thick_tss", 98.0)
        
        if t_day < t0:
            scale = 1.0
        else:
            scale = linear_ramp(t_day, t0, self.fault.end_day,
                                self.fault.thick_tss_scale_day3,
                                self.fault.thick_tss_scale_day14)
        
        self.thickener.t_par[1] = tss_base * scale

    def _get_faulty_so4ref(self, t_day: float) -> Optional[float]:
        """
        获取可能被故障修改的 DO 设定点（F04）。
        
        Returns:
            修改后的设定点值，或 None 表示使用默认值
        """
        if self.fault is None:
            return None
        if self.fault.fault_id != FaultID.F04_DO_SETPOINT_STEP:
            return None
        
        if t_day >= self.fault.start_day:
            return self.fault.so4_step
        return self.fault.so4_base

    def _apply_sensor_fault(self, raw_signal: float, t_day: float) -> float:
        """
        应用 DO 传感器故障（F08: 正向漂移, F09: 负向偏置, F10: 卡死低值）。
        
        优化后三种故障产生不同的系统响应：
        - F08: DO偏高 → 控制器认为氧气足够 → 减少曝气 → 实际DO下降 → 氨氮可能升高
        - F09: DO偏低 → 控制器认为氧气不足 → 增加曝气 → 能耗升高
        - F10: DO严重偏低 → 控制器持续增加曝气 → 过度曝气

        Args:
            raw_signal: 传感器原始测量值
            t_day: 当前仿真时间（天）

        Returns:
            故障后的传感器输出值
        """
        if self.fault is None:
            return raw_signal

        t0 = self.fault.start_day

        # F08: DO 传感器正向漂移（0 → +1.5）
        if self.fault.fault_id == FaultID.F08_DO_SENSOR_DRIFT:
            if t_day < t0:
                drift = 0.0
            else:
                drift = linear_ramp(t_day, t0, self.fault.end_day,
                                    0.0, self.fault.do_drift_end)
            return raw_signal + drift

        # F09: DO 传感器负向偏置（-0.5）
        if self.fault.fault_id == FaultID.F09_DO_SENSOR_OFFSET:
            if t_day >= t0:
                return raw_signal + self.fault.do_offset
            return raw_signal

        # F10: DO 传感器卡死在低值（0.5）
        if self.fault.fault_id == FaultID.F10_DO_SENSOR_FAILURE:
            if t_day >= t0:
                return self.fault.do_fail_value
            return raw_signal

        return raw_signal

    def step(self, i: int, so4ref: float | None = None):
        """
        执行一个仿真步骤，包含故障注入逻辑。
        
        重写 BSM2CL.step() 以实现：
        1. 参数故障注入（动力学、沉淀池、浓缩机）
        2. 控制系统干扰（DO设定点、内回流）
        3. 传感器故障注入
        
        Args:
            i: 当前时间步索引
            so4ref: 外部指定的 DO 设定点（可选）
        """
        t_day = float(self.simtime[i])
        
        # ===== 1) 应用参数故障（在控制计算之前）=====
        if self._in_fault_window(t_day):
            self._apply_kinetic_faults(t_day)
            self._apply_settler_faults(t_day)
            self._apply_recycle_faults(t_day)
            self._apply_thickener_faults(t_day)
        
        # ===== 2) 处理 DO 设定点故障 =====
        if so4ref is None:
            so4ref = self._get_faulty_so4ref(t_day)
        
        # ===== 3) 控制回路（带传感器故障注入）=====
        if so4ref is not None:
            self.pid4.setpoint = so4ref
        
        stepsize = self.timesteps[i]
        
        # 获取噪声索引
        idx_noise = int(np.where(self.noise_timestep - 1e-7 <= t_day)[0][-1])
        
        # 传感器输出（y_out4[SO_INDEX] 是反应器4的溶解氧）
        raw_sensor = self.so4_sensor.output(
            self.y_out4[SO_INDEX], 
            stepsize, 
            self.noise_so4[idx_noise]
        )
        
        # 应用传感器故障
        sensor_signal = self._apply_sensor_fault(raw_sensor, t_day)
        self._do_sensor_history.append(sensor_signal)
        
        # PID 控制器输出
        control_signal = self.pid4.output(sensor_signal, stepsize)
        
        # 执行器输出
        actuator_signal = self.kla4_actuator.output(control_signal, stepsize)
        
        # 更新曝气参数
        self.kla4_a = actuator_signal
        self._kla4_history.append(actuator_signal)
        
        # 保持反应器3和5与反应器4的比例关系
        if self.kla4_a != 0:
            ratio3 = self._baseline.get("kla3_ratio", 1.0)
            ratio5 = self._baseline.get("kla5_ratio", 1.0)
            self.kla3_a = actuator_signal * ratio3
            self.kla5_a = actuator_signal * ratio5
        
        self.klas = np.array([0, 0, self.kla3_a, self.kla4_a, self.kla5_a])
        
        # ===== 4) 执行基类的仿真步骤 =====
        # 调用 BSM2Base.step() 而不是 BSM2CL.step() 以避免重复控制逻辑
        super(BSM2CL, self).step(i)


# ============================================================================
# 数据集导出功能
# ============================================================================
# ASM1 状态变量名称（21个）
ASM1_STATE_NAMES = [
    "SI", "SS", "XI", "XS", "XBH", "XBA", "XP", "SO", 
    "SNO", "SNH", "SND", "XND", "SALK", "TSS", "Q", 
    "TEMP", "SD1", "SD2", "SD3", "XD4", "XD5"
]


def export_timeseries(
    model: FaultyBSM2CL, 
    out_csv: Path, 
    fault_id: str,
    label: int
) -> pd.DataFrame:
    """
    将仿真结果导出为 CSV 文件。
    
    导出内容：
    - time_day: 仿真时间（天）
    - fault_id: 故障类型标识
    - label: 故障类别标签（0=正常, 1-12=对应故障）
    - in_XXX: 进水状态变量（21个）
    - eff_XXX: 出水状态变量（21个）
    - reactor5_XXX: 反应器5出口状态（21个）
    - IQI, EQI, OCI: 性能指标
    - KLA4: 曝气系数
    
    Args:
        model: 完成仿真的模型实例
        out_csv: 输出 CSV 文件路径
        fault_id: 故障类型字符串标识
        label: 故障类别标签
    
    Returns:
        导出的 DataFrame
    """
    n_steps = len(model.y_eff_all)
    t = model.simtime[:n_steps]
    
    df = pd.DataFrame({
        "time_day": t,
        "fault_id": fault_id,
        "label": label
    })
    
    # 进水状态变量
    for k, name in enumerate(ASM1_STATE_NAMES):
        if k < model.y_in_all.shape[1]:
            df[f"in_{name}"] = model.y_in_all[:n_steps, k]
    
    # 出水状态变量
    for k, name in enumerate(ASM1_STATE_NAMES):
        if k < model.y_eff_all.shape[1]:
            df[f"eff_{name}"] = model.y_eff_all[:n_steps, k]
    
    # 反应器5出口状态（用于 DO 监测）
    if hasattr(model, "y_out5_all") and model.y_out5_all is not None:
        for k, name in enumerate(ASM1_STATE_NAMES):
            if k < model.y_out5_all.shape[1]:
                df[f"r5_{name}"] = model.y_out5_all[:n_steps, k]
    
    # 性能指标
    if hasattr(model, "iqi_all") and model.iqi_all is not None:
        df["IQI"] = model.iqi_all[:n_steps]
    if hasattr(model, "eqi_all") and model.eqi_all is not None:
        df["EQI"] = model.eqi_all[:n_steps]
    if hasattr(model, "oci_all") and model.oci_all is not None:
        df["OCI"] = model.oci_all[:n_steps]
    
    # 曝气系数时序
    if model._kla4_history:
        kla4_arr = np.array(model._kla4_history)
        if len(kla4_arr) >= n_steps:
            df["KLA4"] = kla4_arr[:n_steps]
        else:
            # 填充到相同长度
            df["KLA4"] = np.pad(kla4_arr, (0, n_steps - len(kla4_arr)), 
                               mode='edge')
    
    # 传感器读数时序
    if model._do_sensor_history:
        sensor_arr = np.array(model._do_sensor_history)
        if len(sensor_arr) >= n_steps:
            df["DO_sensor"] = sensor_arr[:n_steps]
        else:
            df["DO_sensor"] = np.pad(sensor_arr, (0, n_steps - len(sensor_arr)),
                                     mode='edge')
    
    df.to_csv(out_csv, index=False)
    return df


# ============================================================================
# 仿真运行器
# ============================================================================
def create_stabilized_model(
    data_in: np.ndarray | str,
    end_day: float = 609.0,
    timestep_day: float = 1.0 / 1440.0,
    use_noise: int = 1,
) -> FaultyBSM2CL:
    """
    创建并稳态化一个基准模型（无故障）。
    
    Args:
        data_in: 进水数据（numpy数组或CSV文件路径）
        end_day: 仿真结束时间（天）
        timestep_day: 时间步长（天）
        use_noise: 是否添加传感器噪声
    
    Returns:
        稳态化后的模型实例
    """
    model = FaultyBSM2CL(
        data_in=data_in,
        endtime=end_day,
        timestep=timestep_day,
        use_noise=use_noise,
        fault=None  # 稳态时不注入故障
    )
    model.stabilize()
    return model


def save_steady_state(model: FaultyBSM2CL) -> dict:
    """
    保存模型的稳态状态变量。
    
    Args:
        model: 已稳态化的模型
    
    Returns:
        包含关键状态变量的字典
    """
    import copy
    state = {
        # 出水状态
        'y_eff': model.y_eff.copy(),
        # 反应器状态
        'y_out1': model.y_out1.copy(),
        'y_out2': model.y_out2.copy(),
        'y_out3': model.y_out3.copy(),
        'y_out4': model.y_out4.copy(),
        'y_out5': model.y_out5.copy(),
        # 沉淀池状态
        'ys_of': model.ys_of.copy(),
        'ys_r': model.ys_r.copy() if hasattr(model, 'ys_r') else None,
        # 控制器状态
        'kla3_a': model.kla3_a,
        'kla4_a': model.kla4_a,
        'kla5_a': model.kla5_a,
        'klas': model.klas.copy(),
        # PID 状态
        'pid4_integral': model.pid4.integral,
        'pid4_prev_error': model.pid4.prev_error,
        # 传感器状态
        'so4_sensor_state': model.so4_sensor.state.copy() if hasattr(model.so4_sensor, 'state') else None,
    }
    return state


def restore_steady_state(model: FaultyBSM2CL, state: dict):
    """
    恢复模型的稳态状态变量。
    
    Args:
        model: 要恢复状态的模型
        state: 之前保存的状态字典
    """
    model.y_eff = state['y_eff'].copy()
    model.y_out1 = state['y_out1'].copy()
    model.y_out2 = state['y_out2'].copy()
    model.y_out3 = state['y_out3'].copy()
    model.y_out4 = state['y_out4'].copy()
    model.y_out5 = state['y_out5'].copy()
    model.ys_of = state['ys_of'].copy()
    if state['ys_r'] is not None:
        model.ys_r = state['ys_r'].copy()
    model.kla3_a = state['kla3_a']
    model.kla4_a = state['kla4_a']
    model.kla5_a = state['kla5_a']
    model.klas = state['klas'].copy()
    model.pid4.integral = state['pid4_integral']
    model.pid4.prev_error = state['pid4_prev_error']
    if state['so4_sensor_state'] is not None:
        model.so4_sensor.state = state['so4_sensor_state'].copy()


def run_simulation_from_steady_state(
    data_in: np.ndarray | str,
    steady_state: dict,
    fault: Optional[FaultConfig],
    end_day: float = 609.0,
    timestep_day: float = 1.0 / 1440.0,
    use_noise: int = 1,
) -> FaultyBSM2CL:
    """
    从共享稳态开始，运行带故障的动态仿真。
    
    创建新模型实例并恢复稳态，确保所有故障场景从相同的初始状态开始。
    
    Args:
        data_in: 进水数据
        steady_state: 之前保存的稳态状态
        fault: 故障配置，None 表示正常运行
        end_day: 仿真结束时间（天）
        timestep_day: 时间步长
        use_noise: 是否添加噪声
    
    Returns:
        完成仿真的模型实例
    """
    # 创建新模型实例
    model = FaultyBSM2CL(
        data_in=data_in,
        endtime=end_day,
        timestep=timestep_day,
        use_noise=use_noise,
        fault=fault
    )
    
    # 恢复稳态
    restore_steady_state(model, steady_state)
    model.stabilized = True
    
    # 运行动态仿真
    for i in range(len(model.simtime)):
        if model.simtime[i] > end_day:
            break
        model.step(i)
    
    return model


def run_simulation(
    data_in: np.ndarray | str,
    fault: Optional[FaultConfig],
    end_day: float = 365.0,
    timestep_day: float = 1.0 / 1440.0,  # 1分钟步长
    use_noise: int = 1,
    stabilize_days: float = 100.0,
) -> FaultyBSM2CL:
    """
    运行单次故障仿真（独立稳态版本，用于天气故障等特殊场景）。
    
    Args:
        data_in: 进水数据（numpy数组或CSV文件路径）
        fault: 故障配置，None 表示正常运行
        end_day: 仿真结束时间（天）
        timestep_day: 时间步长（天）
        use_noise: 是否添加传感器噪声（0=无噪声, 1=有噪声）
        stabilize_days: 稳态稳定期（天）
    
    Returns:
        完成仿真的模型实例
    """
    model = FaultyBSM2CL(
        data_in=data_in,
        endtime=end_day,
        timestep=timestep_day,
        use_noise=use_noise,
        fault=fault
    )
    
    # 先进行稳态稳定
    print(f"    稳态稳定中...")
    model.stabilize()
    
    # 运行动态仿真
    print(f"    动态仿真中...")
    for i in range(len(model.simtime)):
        if model.simtime[i] > end_day:
            break
        model.step(i)
    
    return model


def generate_fault_dataset(
    out_dir: Path,
    end_day: float = 609.0,
    timestep_day: float = 1.0 / 1440.0,
    use_noise: int = 1,
):
    """
    生成完整的 12 种故障数据集。
    
    使用 bsm2-python 内置的进水数据：
    - dryinfluent.csv: 干燥天气（用于正常及大部分故障）
    - raininfluent.csv: 降雨天气（F06）
    - 暴风雨通过放大降雨进水模拟（F07）
    
    Args:
        out_dir: 输出目录
        end_day: 仿真结束时间
        timestep_day: 时间步长
        use_noise: 是否添加噪声
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取内置进水数据路径
    # 使用 609 天的长周期数据作为基准
    dry_influent = get_package_data_path("dyninfluent_bsm2.csv")
    rain_influent = get_package_data_path("raininfluent.csv")
    
    print(f"使用进水数据:")
    print(f"  基准数据(609天): {dry_influent}")
    print(f"  降雨天气: {rain_influent}")
    print(f"输出目录: {out_dir}")
    print()
    
    all_dfs = []
    
    # ===== 创建共享稳态基准模型 =====
    print("创建共享稳态基准模型...")
    print("    稳态稳定中...")
    base_model = create_stabilized_model(
        str(dry_influent), 
        end_day=end_day, 
        timestep_day=timestep_day,
        use_noise=use_noise
    )
    # 保存稳态状态
    steady_state = save_steady_state(base_model)
    print("    稳态完成！所有基准天气故障将从此状态开始。")
    print()
    
    # ===== 0) 正常运行（无故障）=====
    print("[0/12] 正常运行...")
    print("    动态仿真中...")
    model = run_simulation_from_steady_state(
        str(dry_influent), steady_state, fault=None, 
        end_day=end_day, timestep_day=timestep_day, use_noise=use_noise
    )
    df = export_timeseries(model, out_dir / "F00_normal.csv", "NORMAL", label=0)
    all_dfs.append(df)
    
    # ===== 1-5) 基准天气下的过程故障（共享稳态）=====
    process_faults = [
        (FaultID.F01_MU_A_REDUCED, 1, "自养菌增长速率降低"),
        (FaultID.F02_MU_H_REDUCED, 2, "异养菌增长速率降低"),
        (FaultID.F03_SETTLING_V_REDUCED, 3, "沉降速度降低"),
        (FaultID.F04_DO_SETPOINT_STEP, 4, "DO设定点突变"),
        (FaultID.F05_INTERNAL_RECYCLE_DRIFT, 5, "内回流漂移"),
    ]
    
    for fault_id, label, desc in process_faults:
        print(f"[{label}/12] {desc}...")
        print("    动态仿真中...")
        fault = FaultConfig(fault_id=fault_id)
        model = run_simulation_from_steady_state(
            str(dry_influent), steady_state, fault=fault,
            end_day=end_day, timestep_day=timestep_day, use_noise=use_noise
        )
        df = export_timeseries(model, out_dir / f"{fault_id.value}.csv",
                              fault_id.value, label=label)
        all_dfs.append(df)
    
    # ===== 6) 降雨干扰 =====
    print("[6/12] 降雨干扰...")
    # 加载降雨数据（修复格式错误）并确保长度足够
    rain_data = load_influent_csv(rain_influent)
    rain_data = ensure_data_length(rain_data, end_day)
    fault = FaultConfig(fault_id=FaultID.F06_RAINY_INFLUENT)
    model = run_simulation(rain_data, fault=fault,
                          end_day=end_day, timestep_day=timestep_day,
                          use_noise=use_noise)
    df = export_timeseries(model, out_dir / f"{FaultID.F06_RAINY_INFLUENT.value}.csv",
                          FaultID.F06_RAINY_INFLUENT.value, label=6)
    all_dfs.append(df)
    
    # ===== 7) 暴风雨干扰 =====
    print("[7/12] 暴风雨干扰...")
    # 使用已加载的降雨数据，放大流量
    stormy_data = rain_data.copy()
    # 列索引：14=Q(流量)
    if stormy_data.shape[1] > 14:
        stormy_data[:, 14] *= 1.5   # Q 流量放大1.5倍
    fault = FaultConfig(fault_id=FaultID.F07_STORMY_INFLUENT)
    model = run_simulation(stormy_data, fault=fault,
                          end_day=end_day, timestep_day=timestep_day,
                          use_noise=use_noise)
    df = export_timeseries(model, out_dir / f"{FaultID.F07_STORMY_INFLUENT.value}.csv",
                          FaultID.F07_STORMY_INFLUENT.value, label=7)
    all_dfs.append(df)
    
    # ===== 8-10) DO 传感器故障（共享稳态）=====
    sensor_faults = [
        (FaultID.F08_DO_SENSOR_DRIFT, 8, "DO传感器漂移"),
        (FaultID.F09_DO_SENSOR_OFFSET, 9, "DO传感器偏置"),
        (FaultID.F10_DO_SENSOR_FAILURE, 10, "DO传感器失效"),
    ]
    
    for fault_id, label, desc in sensor_faults:
        print(f"[{label}/12] {desc}...")
        print("    动态仿真中...")
        fault = FaultConfig(fault_id=fault_id)
        model = run_simulation_from_steady_state(
            str(dry_influent), steady_state, fault=fault,
            end_day=end_day, timestep_day=timestep_day, use_noise=use_noise
        )
        df = export_timeseries(model, out_dir / f"{fault_id.value}.csv",
                              fault_id.value, label=label)
        all_dfs.append(df)
    
    # ===== 11-12) 污泥相关故障（共享稳态）=====
    sludge_faults = [
        (FaultID.F11_WAS_FLOW_DRIFT, 11, "WAS流量漂移"),
        (FaultID.F12_THICKENER_TSS_REMOVAL_DROP, 12, "浓缩机效率下降"),
    ]
    
    for fault_id, label, desc in sludge_faults:
        print(f"[{label}/12] {desc}...")
        print("    动态仿真中...")
        fault = FaultConfig(fault_id=fault_id)
        model = run_simulation_from_steady_state(
            str(dry_influent), steady_state, fault=fault,
            end_day=end_day, timestep_day=timestep_day, use_noise=use_noise
        )
        df = export_timeseries(model, out_dir / f"{fault_id.value}.csv",
                              fault_id.value, label=label)
        all_dfs.append(df)
    
    # ===== 合并所有数据 =====
    print("\n合并所有数据...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(out_dir / "all_faults_combined.csv", index=False)
    
    print(f"\n完成！数据已保存至: {out_dir.resolve()}")
    print(f"  - 单独文件: F00_normal.csv, F01_*.csv, ..., F12_*.csv")
    print(f"  - 合并文件: all_faults_combined.csv")
    print(f"  - 总样本数: {len(combined_df)}")
    
    return combined_df


def run_single_fault(
    fault_choice: int,
    out_dir: Path,
    end_day: float = 609.0,
    timestep_day: float = 1.0 / 1440.0,
    use_noise: int = 1,
):
    """
    运行单个故障仿真。
    
    Args:
        fault_choice: 故障编号 (0-12)
        out_dir: 输出目录
        end_day: 仿真结束时间
        timestep_day: 时间步长
        use_noise: 是否添加噪声
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 故障定义
    FAULT_DEFINITIONS = [
        (None, 0, "正常运行", "F00_normal"),
        (FaultID.F01_MU_A_REDUCED, 1, "自养菌增长速率降低", None),
        (FaultID.F02_MU_H_REDUCED, 2, "异养菌增长速率降低", None),
        (FaultID.F03_SETTLING_V_REDUCED, 3, "沉降速度降低", None),
        (FaultID.F04_DO_SETPOINT_STEP, 4, "DO设定点突变", None),
        (FaultID.F05_INTERNAL_RECYCLE_DRIFT, 5, "内回流漂移", None),
        (FaultID.F06_RAINY_INFLUENT, 6, "降雨干扰", None),
        (FaultID.F07_STORMY_INFLUENT, 7, "暴风雨干扰", None),
        (FaultID.F08_DO_SENSOR_DRIFT, 8, "DO传感器漂移", None),
        (FaultID.F09_DO_SENSOR_OFFSET, 9, "DO传感器偏置", None),
        (FaultID.F10_DO_SENSOR_FAILURE, 10, "DO传感器失效", None),
        (FaultID.F11_WAS_FLOW_DRIFT, 11, "WAS流量漂移", None),
        (FaultID.F12_THICKENER_TSS_REMOVAL_DROP, 12, "浓缩机效率下降", None),
    ]
    
    if fault_choice < 0 or fault_choice > 12:
        print(f"错误：故障编号 {fault_choice} 无效，请选择 0-12")
        return
    
    fault_id, label, desc, filename = FAULT_DEFINITIONS[fault_choice]
    
    # 获取进水数据路径
    dry_influent = get_package_data_path("dyninfluent_bsm2.csv")
    rain_influent = get_package_data_path("raininfluent.csv")
    
    print(f"\n{'='*60}")
    print(f"仿真故障 {label}: {desc}")
    print(f"{'='*60}")
    
    # 处理天气相关故障
    if fault_choice == 6:  # 降雨
        print("加载降雨进水数据...")
        rain_data = load_influent_csv(rain_influent)
        rain_data = ensure_data_length(rain_data, end_day)
        fault = FaultConfig(fault_id=fault_id)
        model = run_simulation(rain_data, fault=fault,
                              end_day=end_day, timestep_day=timestep_day,
                              use_noise=use_noise)
        out_file = out_dir / f"{fault_id.value}.csv"
        df = export_timeseries(model, out_file, fault_id.value, label=label)
        
    elif fault_choice == 7:  # 暴风雨
        print("加载并放大降雨进水数据...")
        rain_data = load_influent_csv(rain_influent)
        rain_data = ensure_data_length(rain_data, end_day)
        stormy_data = rain_data.copy()
        if stormy_data.shape[1] > 14:
            stormy_data[:, 14] *= 1.5   # 流量放大1.5倍
        fault = FaultConfig(fault_id=fault_id)
        model = run_simulation(stormy_data, fault=fault,
                              end_day=end_day, timestep_day=timestep_day,
                              use_noise=use_noise)
        out_file = out_dir / f"{fault_id.value}.csv"
        df = export_timeseries(model, out_file, fault_id.value, label=label)
        
    else:  # 其他故障（使用共享稳态）
        print("创建稳态基准模型...")
        print("    稳态稳定中...")
        base_model = create_stabilized_model(
            str(dry_influent), end_day=end_day, 
            timestep_day=timestep_day, use_noise=use_noise
        )
        steady_state = save_steady_state(base_model)
        print("    稳态完成！")
        
        print("    动态仿真中...")
        if fault_id is None:
            fault = None
            out_file = out_dir / "F00_normal.csv"
            fault_id_str = "NORMAL"
        else:
            fault = FaultConfig(fault_id=fault_id)
            out_file = out_dir / f"{fault_id.value}.csv"
            fault_id_str = fault_id.value
        
        model = run_simulation_from_steady_state(
            str(dry_influent), steady_state, fault=fault,
            end_day=end_day, timestep_day=timestep_day, use_noise=use_noise
        )
        df = export_timeseries(model, out_file, fault_id_str, label=label)
    
    print(f"\n完成！数据已保存至: {out_file}")
    print(f"样本数: {len(df)}")
    return df


def show_menu():
    """显示交互菜单。"""
    print("\n" + "="*60)
    print("BSM2 故障数据集生成器")
    print("="*60)
    print("\n可选故障类型:")
    print("  [0]  正常运行（无故障）")
    print("  [1]  自养菌增长速率降低 (μA -80%)")
    print("  [2]  异养菌增长速率降低 (μH -90%)")
    print("  [3]  沉降速度降低 (v0 -90%)")
    print("  [4]  DO设定点突变 (-60%)")
    print("  [5]  内回流漂移 (+200%)")
    print("  [6]  降雨干扰")
    print("  [7]  暴风雨干扰")
    print("  [8]  DO传感器正向漂移 (+2.5)")
    print("  [9]  DO传感器负向偏置 (-1.0)")
    print("  [10] DO传感器卡死低值 (0.2)")
    print("  [11] WAS流量漂移 (+150%)")
    print("  [12] 浓缩机效率下降 (-70%)")
    print("  [99] 生成全部故障数据集")
    print("  [q]  退出")
    print()


# ============================================================================
# 主程序入口
# ============================================================================
if __name__ == "__main__":
    # 配置参数
    OUTPUT_DIR = Path(__file__).parent / "bsm2_fault_dataset"
    END_DAY = 609.0           # 仿真 609 天
    TIMESTEP = 1.0 / 1440.0  # 1 分钟步长 (1/1440 天)，bsm2-python 要求 ≤1分钟
    USE_NOISE = 1            # 添加传感器噪声
    
    def parse_fault_selection(choice_str: str) -> list:
        """
        解析故障选择输入，支持多种格式：
        - 单个数字: "1" -> [1]
        - 范围: "7-9" -> [7, 8, 9]
        - 逗号分隔: "1,3,5" -> [1, 3, 5]
        - 混合: "1,3-5,7" -> [1, 3, 4, 5, 7]
        """
        result = []
        parts = choice_str.replace(' ', '').split(',')
        for part in parts:
            if '-' in part:
                # 范围格式 "7-9"
                try:
                    start, end = part.split('-')
                    start, end = int(start), int(end)
                    result.extend(range(start, end + 1))
                except:
                    pass
            else:
                # 单个数字
                try:
                    result.append(int(part))
                except:
                    pass
        return sorted(set(result))  # 去重并排序
    
    while True:
        show_menu()
        choice = input("请输入选择 (0-12, 7-9, 1,3,5, 99全部, q退出): ").strip().lower()
        
        if choice == 'q':
            print("退出程序。")
            break
        
        if choice == '99':
            # 生成全部
            print("\n开始生成全部故障数据集...")
            generate_fault_dataset(
                out_dir=OUTPUT_DIR,
                end_day=END_DAY,
                timestep_day=TIMESTEP,
                use_noise=USE_NOISE,
            )
        else:
            # 解析选择
            fault_list = parse_fault_selection(choice)
            
            if not fault_list:
                print("无效输入，请输入有效的数字或范围。")
                continue
            
            # 验证范围
            invalid = [f for f in fault_list if f < 0 or f > 12]
            if invalid:
                print(f"无效的故障编号: {invalid}，有效范围是 0-12")
                continue
            
            print(f"\n将生成以下故障: {fault_list}")
            
            # 逐个生成
            for fault_num in fault_list:
                run_single_fault(
                    fault_choice=fault_num,
                    out_dir=OUTPUT_DIR,
                    end_day=END_DAY,
                    timestep_day=TIMESTEP,
                    use_noise=USE_NOISE,
                )
        
        input("\n按 Enter 继续...")
