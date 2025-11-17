import numpy as np
import streamlit as st
from joblib import load
from pathlib import Path

# ========== 基本设置 ==========
st.set_page_config(
    page_title="山核桃黑籽病发病率预测系统",
    layout="centered",
)

# ========== 加载模型 ==========
BASE_DIR = Path(__file__).parent  # app_ill.py 所在目录
MODEL_PATH = BASE_DIR / "disease_model_poly.pkl"
model = load(MODEL_PATH)
coef = model["coef"]
scaler = model["scaler"]
feature_names = model["feature_names"]
SPORE_FACTOR = model["spore_factor"]   # 虽然加载了，但前端已经不用它了

# ========== 页面标题 ==========
st.markdown(
    """
    <h2 style="text-align:center; margin-bottom:0.2rem;">山核桃黑籽病发病率预测系统</h2>
    <p style="text-align:center; color: #555;">
        输入 5 月 15 日至 8 月 15 日高温时长、5/7 月周孢子峰值及经营水平，系统将评估黑籽病风险等级
    </p>
    <hr style="margin-top:0.5rem; margin-bottom:1rem;">
    """,
    unsafe_allow_html=True,
)

# ========== 1. 环境条件 ==========
st.subheader("一、环境条件（温度）")

hours = st.number_input(
    "5 月 15 日至 8 月 15 日期间 >28℃ 的累计小时数",
    min_value=0.0,
    max_value=3000.0,
    value=300.0,
    step=10.0,
)

st.markdown("<br>", unsafe_allow_html=True)

# ========== 2. 孢子流量 ==========
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

# ========== 3. 经营条件 ==========
st.subheader("三、经营条件")

level = st.selectbox(
    "经营水平",
    ["良好", "中等", "一般"],
)
encode_map = {"良好": 0, "中等": 1, "一般": 2}
level_code = encode_map[level]

st.markdown("<br>", unsafe_allow_html=True)

# ========== 预测函数 ==========
def predict_from_inputs(heat_hours,
                        may_peak_spores,
                        july_peak_spores,
                        level_code):

    x_raw = np.array([[heat_hours,
                       may_peak_spores,
                       july_peak_spores,
                       level_code]])

    z = scaler.transform(x_raw)           # (1, 4)
    z_design = np.c_[np.ones(len(z)), z]  # 偏置 + 特征

    pred = float(z_design @ coef)
    return max(0.0, min(pred, 100.0))

# ========== 按钮与结果输出 ==========
if st.button("开始预测"):

    pred = predict_from_inputs(
        heat_hours=hours,
        may_peak_spores=may_peak_spores,
        july_peak_spores=july_peak_spores,
        level_code=level_code,
    )

    # 风险等级颜色
    if pred > 30:
        color = "#FF4C4C"
        label = "发病风险：极高"
        text_color = "white"

    elif pred > 20:
        color = "#FFD93D"
        label = "发病风险：较高"
        text_color = "black"

    elif pred > 10:
        color = "#4DA6FF"
        label = "发病风险：中等"
        text_color = "white"

    else:
        color = "#4CD964"
        label = "发病风险：较低"
        text_color = "black"

    # ======== 风险等级卡片 ========
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

    # ======== 指标说明 + 颜色说明 ========
    st.markdown("### 指标说明")
    st.write(
        f"- 高温时长：**{hours:.1f} 小时**\n"
        f"- 5 月周孢子峰值：**{may_peak_spores:.0f} 孢子**\n"
        f"- 7 月周孢子峰值：**{july_peak_spores:.0f} 孢子**\n"
        f"- 经营水平：**{level}**"
    )

    # 这里就是你要的颜色对应关系说明
    st.markdown(
        """
        **颜色与发病严重程度对应关系：**  
        - 🔴 **红色**：发病极高（> 30%）  
        - 🟡 **黄色**：发病较高（20% ~ 30%）  
        - 🔵 **蓝色**：发病中等（10% ~ 20%）  
        - 🟢 **绿色**：发病较低（< 10%）  
        """
    )

else:
    st.warning("请填写以上参数后，点击“开始预测”进行风险评估。")

