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
poly = model["poly"]

# ========== 页面标题 ==========
st.markdown(
    """
    <h2 style="text-align:center; margin-bottom:0.2rem;">山核桃黑籽病发病率预测系统</h2>
    <p style="text-align:center; color: #555;">
        输入近三个月高温时长、三种孢子数量及经营水平，系统将评估黑籽病风险等级
    </p>
    <hr style="margin-top:0.5rem; margin-bottom:1rem;">
    """,
    unsafe_allow_html=True,
)

# ========== 输入区 ==========
st.subheader("环境与经营指标")

col1, col2 = st.columns(2)

with col1:
    hours = st.number_input(
        "三个月内 >28℃ 的总小时数",
        min_value=0.0,
        max_value=2000.0,
        value=300.0,
        step=10.0,
    )

with col2:
    level = st.selectbox(
        "经营水平",
        ["良好", "中等", "一般"],
    )

encode_map = {"良好": 0, "中等": 1, "一般": 2}
level_code = encode_map[level]

# ========== 孢子数量输入 ==========
st.subheader("五月孢子峰值周（单位：孢子数）")

c1, c2, c3 = st.columns(3)
with c1:
    spore1 = st.number_input("小孢拟盘孢子数", min_value=0, max_value=200000, value=10000, step=100)
with c2:
    spore2 = st.number_input("葡萄座腔菌孢子数", min_value=0, max_value=200000, value=3000, step=100)
with c3:
    spore3 = st.number_input("假可可毛色二孢孢子数", min_value=0, max_value=200000, value=5000, step=100)

st.markdown("<br>", unsafe_allow_html=True)

# ========== 预测按钮 ==========
if st.button("开始预测"):

    # ========== 组装特征（直接使用用户输入的孢子数） ==========
    x = np.array([[hours, spore1, spore2, spore3, level_code]])
    x_poly = poly.transform(x)
    pred = float(x_poly @ coef)

    pred = max(0.0, min(pred, 100.0))  # 限制范围

    # ========= 风险等级颜色（红-黄-蓝-绿） =========
    if pred > 30:
        color = "#FF4C4C"  # 红
        label = "发病风险：极高"
        text_color = "white"
    elif pred > 20:
        color = "#FFD93D"  # 黄
        label = "发病风险：较高"
        text_color = "black"
    elif pred > 10:
        color = "#4DA6FF"  # 蓝
        label = "发病风险：中等"
        text_color = "white"
    else:
        color = "#4CD964"  # 绿
        label = "发病风险：较低"
        text_color = "black"

    # ========= 风险等级卡片（不显示具体发病率） ==========
    st.markdown(
        f"""
        <div style="
            padding: 30px;
            border-radius: 14px;
            background: {color};
            text-align: center;
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            font-size: 28px;
            font-weight: 700;
            color:{text_color};
        ">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========= 指标说明 ==========
    st.markdown("### 指标说明")
    st.write(
        f"- 三个月高温时长：**{hours:.1f} 小时**\n"
        f"- 小孢拟盘孢子数：**{spore1}**\n"
        f"- 葡萄座腔菌孢子数：**{spore2}**\n"
        f"- 假可可毛色二孢孢子数：**{spore3}**\n"
        f"- 经营水平：**{level}**"
    )


else:
    st.warning("请填写以上参数后，点击“开始预测”进行风险评估。")
