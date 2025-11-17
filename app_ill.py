# app_ill.py — 含零输入 & 上限逻辑
import numpy as np
import streamlit as st
from joblib import load
from pathlib import Path

def inv_logit(z):
    return 1.0 / (1.0 + np.exp(-z))

# ---------- 页面设置 ----------
st.set_page_config(page_title="山核桃黑籽病预警系统", layout="centered")

# ---------- 加载模型 ----------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "disease_model_poly.pkl"
model = load(MODEL_PATH)

scaler = model["scaler"]
poly = model["poly"]
ridge = model["ridge"]
SPORE_FACTOR = model["spore_factor"]
Y_MAX = model["y_max"]

# ---------- 模型边界 ----------
TEMP_MAX = 1900.0
SPORE_MAX = (60 + 9 + 16 + 45 + 5 + 30) * SPORE_FACTOR  # 训练集最大孢子估算

# ---------- 页面标题 ----------
st.markdown("""
<h2 style="text-align:center;">山核桃黑籽病预警系统</h2>
<p style="text-align:center;color:#aaa;">
输入 5 月 15 日至 8 月 15 日高温时长、5/7 月周孢子峰值及经营水平，系统将评估黑籽病风险等级。
</p>
<hr>
""", unsafe_allow_html=True)

# ---------- 输入 ----------
st.subheader("一、环境条件（温度）")
hours = st.number_input("5 月 15 日至 8 月 15 日期间 >28℃ 的累计小时数", 0.0, 2160.0, 200.0, 10.0)

st.subheader("二、孢子流量（周峰值，单位：孢子数）")
c1, c2 = st.columns(2)
with c1:
    may_peak_spores = st.number_input("5 月周孢子峰值（孢子数）", 0.0, 1_000_000.0, 200_000.0, 1000.0)
with c2:
    july_peak_spores = st.number_input("7 月周孢子峰值（孢子数）", 0.0, 1_000_000.0, 200_000.0, 1000.0)

st.subheader("三、经营条件")
level = st.selectbox("经营水平", ["良好", "中等", "一般"])
level_code = {"良好": 0, "中等": 1, "一般": 2}[level]

# ---------- 预测函数 ----------
def predict(heat_hours, may_spores, july_spores, level_code):
    spore_sum = may_spores + july_spores

    # === 全为零输入 ===
    if heat_hours == 0 and may_spores == 0 and july_spores == 0:
        return 0.0, "zero"

    # === 上限触发 ===
    if heat_hours >= TEMP_MAX or spore_sum >= SPORE_MAX:
        return Y_MAX, "max"

    # === 正常预测 ===
    x_raw = np.array([[heat_hours, may_spores, july_spores, level_code]])
    xs = scaler.transform(x_raw)
    xp = poly.transform(xs)
    z = ridge.predict(xp)
    y_pred = Y_MAX * inv_logit(z)
    y_pred = float(np.clip(y_pred, 0.0, Y_MAX))

    # === 按经营调整 ===
    if level_code == 0:
        y_pred *= 0.9
    elif level_code == 2:
        y_pred *= 1.1

    return float(np.clip(y_pred, 0.0, Y_MAX)), "normal"

# ---------- 预测按钮 ----------
if st.button("开始预测"):
    pred, status = predict(hours, may_peak_spores, july_peak_spores, level_code)

    # ===== 风险分类 =====
    if status == "zero":
        color, label, text_color = "#4CD964", "发病风险：极低", "black"
    elif status == "max":
        color, label, text_color = "#FF4C4C", "发病风险：极高", "white"
    elif pred > 30:
        color, label, text_color = "#FF4C4C", "发病风险：极高", "white"
    elif pred > 20:
        color, label, text_color = "#FFD93D", "发病风险：较高", "black"
    elif pred > 10:
        color, label, text_color = "#4DA6FF", "发病风险：中等", "white"
    else:
        color, label, text_color = "#4CD964", "发病风险：较低", "black"

    st.markdown(f"""
        <div style="
            padding:30px;border-radius:14px;
            background:{color};
            text-align:center;
            font-size:26px;font-weight:700;
            color:{text_color};
            box-shadow:0 4px 10px rgba(0,0,0,0.15);
        ">{label}</div>
    """, unsafe_allow_html=True)

    # === 指标说明 ===
    st.markdown("### 指标说明")
    st.write(
        f"- 高温时长：**{hours:.1f} 小时**\n"
        f"- 5 月周孢子峰值：**{may_peak_spores:.0f} 孢子**\n"
        f"- 7 月周孢子峰值：**{july_peak_spores:.0f} 孢子**\n"
        f"- 经营水平：**{level}**"
    )
    st.markdown("""
    **颜色与发病严重程度对应：**  
    - 🔴 **红色**：发病风险极高  
    - 🟡 **黄色**：较高  
    - 🔵 **蓝色**：中等  
    - 🟢 **绿色**：较低  
    - ⚪ **白色/绿底**：极低  
    """)
else:
    st.info("请填写参数并点击“开始预测”")
