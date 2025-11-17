# app_ill.py
import numpy as np
import streamlit as st
from joblib import load
from pathlib import Path

# ---------- 数学函数 ----------
def inv_logit(z: np.ndarray) -> np.ndarray:
    """logit 的反函数，把实数映射回 (0, 1) 区间"""
    return 1.0 / (1.0 + np.exp(-z))

# ---------- 基本设置 ----------
st.set_page_config(
    page_title="山核桃黑籽病预警系统",
    layout="centered",
)

# ---------- 加载模型 ----------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "disease_model_poly.pkl"
model = load(MODEL_PATH)

scaler = model["scaler"]            # 标准化器
poly = model["poly"]                # 多项式特征器
ridge = model["ridge"]              # 岭回归模型
feature_names = model["feature_names"]
SPORE_FACTOR = model["spore_factor"]
Y_MAX = model["y_max"]              # 最大发病率上限（如 50）

# ---------- 页面标题 ----------
st.markdown(
    """
    <h2 style="text-align:center; margin-bottom:0.2rem;">山核桃黑籽病预警系统</h2>
    <p style="text-align:center; color: #bbb;">
        输入 5 月 15 日至 8 月 15 日高温时长、5/7 月周孢子峰值及经营水平，系统将评估黑籽病风险等级
    </p>
    <hr style="margin-top:0.5rem; margin-bottom:1rem;">
    """,
    unsafe_allow_html=True,
)

# ---------- 1. 环境条件 ----------
st.subheader("一、环境条件（温度）")
hours = st.number_input(
    "5 月 15 日至 8 月 15 日期间 >28℃ 的累计小时数",
    min_value=0.0,
    max_value=2160.0,  # 90 天 * 24 小时
    value=200.0,
    step=10.0,
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------- 2. 孢子流量 ----------
st.subheader("二、孢子流量（周峰值，单位：孢子数）")
col1, col2 = st.columns(2)
with col1:
    may_peak_spores = st.number_input(
        "5 月周孢子峰值（孢子数）",
        min_value=0.0,
        max_value=1_000_000.0,
        value=200_000.0,
        step=1_000.0,
        format="%.0f",
    )
with col2:
    july_peak_spores = st.number_input(
        "7 月周孢子峰值（孢子数）",
        min_value=0.0,
        max_value=1_000_000.0,
        value=200_000.0,
        step=1_000.0,
        format="%.0f",
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------- 3. 经营条件 ----------
st.subheader("三、经营条件")
level = st.selectbox("经营水平", ["良好", "中等", "一般"])
encode_map = {"良好": 0, "中等": 1, "一般": 2}
level_code = encode_map[level]

# ---------- 4. 预测函数 ----------
def predict_from_inputs(heat_hours, may_peak_spores, july_peak_spores, level_code):
    x_raw = np.array([[heat_hours, may_peak_spores, july_peak_spores, level_code]])
    x_scaled = scaler.transform(x_raw)
    x_poly = poly.transform(x_scaled)
    z = ridge.predict(x_poly)
    y_pred = Y_MAX * inv_logit(z)
    return float(np.clip(y_pred, 0.0, Y_MAX))

# ---------- 5. 预测按钮 ----------
if st.button("开始预测"):
    pred = predict_from_inputs(
        heat_hours=hours,
        may_peak_spores=may_peak_spores,
        july_peak_spores=july_peak_spores,
        level_code=level_code,
    )

    # 风险分级
    if pred > 30:
        color, label, text_color = "#FF4C4C", "发病风险：极高", "white"
    elif pred > 25:
        color, label, text_color = "#FFD93D", "发病风险：较高", "black"
    elif pred > 20:
        color, label, text_color = "#4DA6FF", "发病风险：中等", "white"
    else:
        color, label, text_color = "#4CD964", "发病风险：较低", "black"

    # 风险卡片展示
    st.markdown(
        f"""
        <div style="
            padding: 30px;
            border-radius: 14px;
            background: {color};
            text-align: center;
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            font-size: 26px;
            font-weight: 700;
            color:{text_color};
        ">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 指标说明
    st.markdown("### 指标说明")
    st.write(
        f"- 高温时长：**{hours:.1f} 小时**\n"
        f"- 5 月周孢子峰值：**{may_peak_spores:.0f} 孢子**\n"
        f"- 7 月周孢子峰值：**{july_peak_spores:.0f} 孢子**\n"
        f"- 经营水平：**{level}**"
    )

    st.markdown(
        """
        **颜色与发病严重程度对应关系：**  
        - 🔴 **红色**：发病风险极高  
        - 🟡 **黄色**：发病风险较高  
        - 🔵 **蓝色**：发病风险中等  
        - 🟢 **绿色**：发病风险较低  
        """
    )
else:
    st.warning("请填写以上参数后，点击“开始预测”进行风险评估。")



