明白了！这是完全专注于**数据集本身**（列的含义、数据格式、物理意义）的说明文档，没有任何脚本运行或代码相关的内容。

***

# 🛳️ Ship Main Engine Sensor Dataset
# 船舶主机传感器数据集

## 📖 Overview / 概述

This dataset contains **normalized** and **desensitized** sensor data collected from the main engines of two large vessels: **YuanFuYang (远福洋)** and **YuanBeiHai (远北海)**.
The data is organized by individual voyages. All physical values have been scaled to a range of `[0, 1]`, and sensitive time information has been converted into relative time steps.

本数据集包含从 **远福洋** 和 **远北海** 两艘大型船舶主机采集的**归一化**及**脱敏**传感器数据。
数据按独立航次组织。所有物理数值均已缩放至 `[0, 1]` 区间，敏感的时间信息已转换为相对时间步长。

---

## ⚙️ Data Specifications / 数据规格

*   **File Format / 文件格式**: `.csv`
*   **Value Range / 数值范围**: `0.0` to `1.0` (Min-Max Normalized / 归一化)
*   **Missing Values / 缺失值**: Filled with `0` (None / 无缺失)
*   **Time Format / 时间格式**: Relative Steps (No absolute timestamps / 无绝对时间戳)

---

## 📊 Feature Description / 特征描述

Each CSV file contains **13 columns**. The columns correspond to the following physical sensors and operational parameters:
每个 CSV 文件包含 **13 列**数据。各列对应的物理传感器及运行参数如下：

| Variable Name (变量名) | Chinese Name (中文名) | Description (English) | Role (角色) |
| :--- | :--- | :--- | :--- |
| **step** | **时间步长** | **Relative Time Index** (Sequence Order) | **Index / 索引** |
| **Me1Thrust_bearing_pad_temp** | **推力轴承瓦温** | **Main Engine Thrust Bearing Pad Temperature** | **Target / 预测目标** |
| `Me1Nms` | 主机转速 | Main Engine RPM (Revolutions Per Minute) | Input Feature |
| `Me1Load` | 主机负荷 | Main Engine Load Indicator | Input Feature |
| `Me1Axial_vib` | 轴向振动 | Main Engine Axial Vibration | Input Feature |
| `Me1Main_brg_lo_in_press` | 主轴承滑油进口压力 | Main Bearing Lube Oil Inlet Pressure | Input Feature |
| `Me1Main_brg_lo_in_temp` | 主轴承滑油进口温度 | Main Bearing Lube Oil Inlet Temperature | Input Feature |
| `Me1Sw_com_temp` | 海水总管温度 | Seawater Common/Inlet Temperature | Input Feature |
| `Df` | 燃油流量 | Fuel Flow Meter Reading | Input Feature |
| `Da` | 吃水 | Draft (Depth of the ship's keel) | Input Feature |
| `Rudder` | 舵角 | Rudder Angle | Input Feature |
| `TrueVwr` | 相对风速 | Relative Wind Speed | Input Feature |
| `EeIndex8` | 能效索引 | Energy Efficiency Index | Input Feature |

---

## 📝 Notes / 备注

1.  **Uniformity / 一致性**:
    *   The column `Me1Sw_com_temp` represents Seawater Temperature for both ships (mapped from `Me1Sw_in_temp` for YuanBeiHai).
    *   `Me1Sw_com_temp` 列代表海水温度，两艘船已统一列名（远北海原为 `Me1Sw_in_temp`，已映射）。

2.  **Privacy / 隐私**:
    *   GPS coordinates (Latitude/Longitude) and MMSI (Ship ID) have been removed.
    *   GPS 坐标（经纬度）和 MMSI（船舶识别码）已被移除。

3.  **Data Quality / 数据质量**:
    *   Columns with sensor failures (all-NaNs in raw data) have been zero-filled to ensure format consistency.
    *   因传感器故障导致的原数据全空列已被填充为 0，以保证数据格式一致性。
