import numpy as np
import streamlit as st
from joblib import load

# ========== 基本设置 ==========
st.set_page_config(
    page_title="山核桃黑籽病发病率预测系统",
    layout="centered",
)

# ========== 加载模型 ==========
model = load(r"C:\Users\zhbshen\PycharmProjects\PythonProject\disease_model_poly.pkl")
coef = model["coef"]
scaler = model["scaler"]
feature_names = model["feature_names"]
SPORE_FACTOR = model["spore_factor"]

# ========== 页面标题 ==========
st.markdown(
    """
    <h2 style="text-align:center; margin-bottom:0.2rem;">山核桃黑籽病发病率预测系统</h2>
    <p style="text-align:center; color: #555;">
        输入 5 月 15 日至 8 月 15 日高温时长、5/7 月三种孢子周峰值及经营水平，系统将评估黑籽病风险等级
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

st.markdown("**1）5 月孢子峰值周**")
c1, c2, c3 = st.columns(3)
with c1:
    sp1_may = st.number_input("小孢拟盘（5 月）", min_value=0, max_value=500000, value=10000, step=100)
with c2:
    sp2_may = st.number_input("葡萄座腔菌（5 月）", min_value=0, max_value=500000, value=3000, step=100)
with c3:
    sp3_may = st.number_input("假可可毛色二孢（5 月）", min_value=0, max_value=500000, value=5000, step=100)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("**2）7 月孢子峰值周**")
d1, d2, d3 = st.columns(3)
with d1:
    sp1_july = st.number_input("小孢拟盘（7 月）", min_value=0, max_value=500000, value=8000, step=100)
with d2:
    sp2_july = st.number_input("葡萄座腔菌（7 月）", min_value=0, max_value=500000, value=2000, step=100)
with d3:
    sp3_july = st.number_input("假可可毛色二孢（7 月）", min_value=0, max_value=500000, value=4000, step=100)

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
                        sp1_may, sp2_may, sp3_may,
                        sp1_july, sp2_july, sp3_july,
                        level_code):
    x_raw = np.array([[heat_hours,
                       sp1_may, sp2_may, sp3_may,
                       sp1_july, sp2_july, sp3_july,
                       level_code]])

    z = scaler.transform(x_raw)
    z_design = np.c_[np.ones(len(z)), z]
    pred = float(z_design @ coef)

    return max(0.0, min(pred, 100.0))

# ========== 按钮与结果输出 ==========
if st.button("开始预测"):

    pred = predict_from_inputs(
        heat_hours=hours,
        sp1_may=sp1_may, sp2_may=sp2_may, sp3_may=sp3_may,
        sp1_july=sp1_july, sp2_july=sp2_july, sp3_july=sp3_july,
        level_code=level_code,
    )

    # 风险等级
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

    # ========= 风险等级卡片（已去掉预测数值）=========
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
        f"- 5 月小孢拟盘：**{sp1_may}**\n"
        f"- 5 月葡萄座腔菌：**{sp2_may}**\n"
        f"- 5 月假可可毛色二孢：**{sp3_may}**\n"
        f"- 7 月小孢拟盘：**{sp1_july}**\n"
        f"- 7 月葡萄座腔菌：**{sp2_july}**\n"
        f"- 7 月假可可毛色二孢：**{sp3_july}**\n"
        f"- 经营水平：**{level}**"
    )

else:
    st.warning("请填写以上参数后，点击“开始预测”进行风险评估。")
