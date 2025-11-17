import numpy as np
import streamlit as st
from joblib import load
from pathlib import Path

# ========== 基本设置 ==========
st.set_page_config(
    page_title="山核桃黑籽病预警系统",
    layout="centered",
)

# ========== 加载模型（适配“二次多项式 + 岭回归”） ==========
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "disease_model_poly.pkl"
model = load(MODEL_PATH)

# 训练脚本已把截距和多项式系数合并：coef = [intercept, beta_1, ...]
coef_full = model["coef"]
scaler = model["scaler"]          # 作用在 4 个原始特征（高温、5月孢子、7月孢子、经营编码）
poly = model.get("poly", None)    # PolynomialFeatures（必须存在，除非你保存的是旧线性模型）
y_scale = model.get("y_scale", 50.0)  # 训练时用的上限（通常为 50）
# 说明：前端输入的就是孢子数（不是格数），所以这里不需要 spore_factor
# SPORE_FACTOR = model.get("spore_factor", 7638)

# ========== 页面标题 ==========
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

# ========== 1. 环境条件 ==========
st.subheader("一、环境条件（温度）")
hours = st.number_input(
    "5 月 15 日至 8 月 15 日期间 >28℃ 的累计小时数",
    min_value=0.0, max_value=2160.0, value=200.0, step=10.0,
)

st.markdown("<br>", unsafe_allow_html=True)

# ========== 2. 孢子流量 ==========
st.subheader("二、孢子流量（周峰值，单位：孢子数）")
col1, col2 = st.columns(2)
with col1:
    may_peak_spores = st.number_input(
        "5 月周孢子峰值（孢子数）", min_value=0.0, max_value=1_000_000.0,
        value=200_000.0, step=1_000.0, format="%.0f",
    )
with col2:
    july_peak_spores = st.number_input(
        "7 月周孢子峰值（孢子数）", min_value=0.0, max_value=1_000_000.0,
        value=200_000.0, step=1_000.0, format="%.0f",
    )

st.markdown("<br>", unsafe_allow_html=True)

# ========== 3. 经营条件 ==========
st.subheader("三、经营条件")
level = st.selectbox("经营水平", ["良好", "中等", "一般"])
encode_map = {"良好": 0, "中等": 1, "一般": 2}
level_code = encode_map[level]

# ========== 预测函数（多项式 → 岭回归；输出连续 0~y_scale） ==========
def predict_from_inputs(heat_hours, may_spores, july_spores, level_code):
    """
    heat_hours: 三个月内 >28℃ 的总小时数
    may_spores: 5 月周孢子峰值（孢子数）
    july_spores: 7 月周孢子峰值（孢子数）
    level_code: 经营编码（良好=0 / 中等=1 / 一般=2）
    """
    # 1) 原始 4 维特征
    x_base = np.array([[heat_hours, may_spores, july_spores, level_code]], dtype=float)

    # 2) 标准化（与训练保持一致）
    xz = scaler.transform(x_base)

    # 3) 多项式展开（degree=2，含交互）
    if poly is not None:
        x_feat = poly.transform(xz)
    else:
        # 兼容极少数旧模型（没有 poly），直接用标准化后的线性特征
        x_feat = xz

    # 4) 线性点乘（[1, x_feat] @ coef_full），再缩放回 0~y_scale（通常 y_scale=50）
    y_scaled = float(np.c_[np.ones((1, 1)), x_feat] @ coef_full)   # 期望在 0~1
    y_pred = float(np.clip(y_scaled, 0.0, 1.0) * y_scale)          # 连续 0~y_scale

    return y_pred

# ========== 预测按钮 ==========
if st.button("开始预测"):
    pred = predict_from_inputs(
        heat_hours=hours,
        may_spores=may_peak_spores,     # 注意：这里前端输入的是“孢子数”，无需再 × 7638
        july_spores=july_peak_spores,
        level_code=level_code,
    )

    # —— 连续预测 + 四档可视化分级（不做硬规则）——
    #   以下阈值仍按 10/20/30（单位：%）；若你的 y_scale 是 50，则含义为 0~50% 区间内的分档
    if pred > 30:
        color, label, text_color = "#FF4C4C", "发病风险：极高", "white"
    elif pred > 20:
        color, label, text_color = "#FFD93D", "发病风险：较高", "black"
    elif pred > 10:
        color, label, text_color = "#4DA6FF", "发病风险：中等", "white"
    else:
        color, label, text_color = "#4CD964", "发病风险：较低", "black"

    # 结果卡片
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
        **颜色与发病严重程度对应关系（基于模型连续预测值）：**  
        - 🔴 **红色**：发病风险极高  
        - 🟡 **黄色**：发病风险较高  
        - 🔵 **蓝色**：发病风险中等  
        - 🟢 **绿色**：发病风险较低
        """
    )

else:
    st.warning("请填写以上参数后，点击“开始预测”进行风险评估。")
