import numpy as np
import streamlit as st
from joblib import load

# ========== 基本设置 ==========
st.set_page_config(
    page_title="山核桃黑籽病预警系统",
    layout="centered",
)


# ========== 加载模型 ==========
model = load(r"disease_model_poly.pkl")
coef = model["coef"]
scaler = model["scaler"]
feature_names = model["feature_names"]
SPORE_FACTOR = float(model.get("spore_factor", 395.0))  # 默认 395

# ========== 页面标题 ==========
st.markdown(
    """
    <h2 style="text-align:center; margin-bottom:0.2rem;">山核桃黑籽病预警系统</h2>
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
    "5 月 15 日至 8 月 15 日期间 >28℃并<35℃ 的累计小时数",
    min_value=0.0,
    max_value=3000.0,
    value=300.0,
    step=10.0,
)

st.markdown("<br>", unsafe_allow_html=True)

# ========== 2. 孢子（周峰值计数） ==========
st.subheader("二、孢子（周峰值计数）")

st.markdown("**1）5 月周孢子峰值**")
c1, c2, c3 = st.columns(3)
with c1:
    sp1_may = st.number_input("小孢拟盘", min_value=0, max_value=500000, value=10000, step=100)
with c2:
    sp2_may = st.number_input("葡萄座腔菌", min_value=0, max_value=500000, value=3000, step=100)
with c3:
    sp3_may = st.number_input("假可可毛色二孢", min_value=0, max_value=500000, value=5000, step=100)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("**2）7 月周孢子峰值**")
d1, d2, d3 = st.columns(3)
with d1:
    sp1_july = st.number_input("小孢拟盘", min_value=0, max_value=500000, value=8000, step=100)
with d2:
    sp2_july = st.number_input("葡萄座腔菌", min_value=0, max_value=500000, value=2000, step=100)
with d3:
    sp3_july = st.number_input("假可可毛色二孢", min_value=0, max_value=500000, value=4000, step=100)

st.markdown("<br>", unsafe_allow_html=True)

# ========== 3. 经营条件（mgt_code 0-4） ==========
st.subheader("三、经营条件")

encode_map = {"优": 0, "良": 1, "中": 2, "一般": 3, "差": 4}
level = st.selectbox("经营水平", list(encode_map.keys()))
level_code = encode_map[level]

st.markdown("<br>", unsafe_allow_html=True)

# ========== 预测函数（严格按 feature_names 组装） ==========
def predict_from_inputs(
    heat_hours,
    sp1_may, sp2_may, sp3_may,
    sp1_july, sp2_july, sp3_july,
    level_code
):
    may_spore_N = float(sp1_may + sp2_may + sp3_may)
    jul_spore_N = float(sp1_july + sp2_july + sp3_july)

    may_total = may_spore_N
    jul_total = jul_spore_N

    feature_map = {
        "heat_hours": float(heat_hours),
        "may_spore_N": float(may_spore_N),
        "jul_spore_N": float(jul_spore_N),
        "mgt_code": float(level_code),
    }

    x_raw = np.array([[feature_map[name] for name in feature_names]], dtype=float)

    z = scaler.transform(x_raw)  # (1, p)

    # ====== ✅ 修复点：永远输出单个标量，不再 float(向量) ======
    coef_arr = np.asarray(coef)

    # coef 一维：分“含截距/不含截距”
    if coef_arr.ndim == 1 and coef_arr.shape[0] == z.shape[1] + 1:
        z_design = np.c_[np.ones((z.shape[0], 1)), z]   # (1, p+1)
        res = z_design @ coef_arr                       # (1,)
    elif coef_arr.ndim == 1 and coef_arr.shape[0] == z.shape[1]:
        res = z @ coef_arr                              # (1,)
    else:
        # 兜底：不管 res 是啥形状，先算出来再取第一个值（保证不报 TypeError）
        z_design = np.c_[np.ones((z.shape[0], 1)), z]
        res = z_design @ coef_arr

    pred = float(np.ravel(np.asarray(res))[0])
    # ============================================================

    return max(0.0, min(pred, 100.0)), may_total, jul_total, may_spore_N, jul_spore_N


# ========== 结果输出 ==========
if st.button("开始预测"):

    pred, may_total, jul_total, may_spore_N, jul_spore_N = predict_from_inputs(
        heat_hours=hours,
        sp1_may=sp1_may, sp2_may=sp2_may, sp3_may=sp3_may,
        sp1_july=sp1_july, sp2_july=sp2_july, sp3_july=sp3_july,
        level_code=level_code,
    )

    if pred > 30:
        color = "#FF4C4C"; label = "发病风险：极高"; text_color = "white"
    elif pred > 20:
        color = "#FFD93D"; label = "发病风险：较高"; text_color = "black"
    elif pred > 10:
        color = "#4DA6FF"; label = "发病风险：中等"; text_color = "white"
    else:
        color = "#4CD964"; label = "发病风险：较低"; text_color = "black"

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

    
    st.markdown("##### 颜色与发病严重程度对应：")
    st.markdown(
        """
        <div style="margin-top: 0.2rem; line-height: 2.0; font-size: 18px;">
          <div style="display:flex; align-items:center; gap:10px;">
            <span style="width:18px; height:18px; border-radius:50%; background:#FF4C4C; display:inline-block;"></span>
            <span><b>红色：</b>发病风险极高（pred &gt; 30）</span>
          </div>
          <div style="display:flex; align-items:center; gap:10px;">
            <span style="width:18px; height:18px; border-radius:50%; background:#FFD93D; display:inline-block;"></span>
            <span><b>黄色：</b>较高（20 &lt; pred ≤ 30）</span>
          </div>
          <div style="display:flex; align-items:center; gap:10px;">
            <span style="width:18px; height:18px; border-radius:50%; background:#4DA6FF; display:inline-block;"></span>
            <span><b>蓝色：</b>中等（10 &lt; pred ≤ 20）</span>
          </div>
          <div style="display:flex; align-items:center; gap:10px;">
            <span style="width:18px; height:18px; border-radius:50%; background:#4CD964; display:inline-block;"></span>
            <span><b>绿色：</b>较低（pred ≤ 10）</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    st.write(
        f"- 高温时长：**{hours:.1f} 小时**\n"
        f"- 5 月三种孢子周峰值合计：**{may_total:.0f}**\n"
        f"- 7 月三种孢子周峰值合计：**{jul_total:.0f}**\n"
        f"- 经营水平：**{level}**\n"
    )

else:
    st.warning("请填写以上参数后，点击“开始预测”进行风险评估。")




