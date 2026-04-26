import streamlit as st
import plotly.graph_objects as go
import numpy as np
from pendulum_engine import lqr_gain, simulate_lqr, simulate_mpc
 
st.set_page_config(page_title="Nonlinear Control & MPC", page_icon="\U0001F39B\uFE0F", layout="wide")
st.title("\U0001F39B\uFE0F Inverted Pendulum: LQR vs MPC Control")
st.markdown("**Nonlinear & Optimization-Based Control**")
st.divider()
 
st.sidebar.header("\U0001F4D0 Initial Conditions")
theta_0 = st.sidebar.slider("Initial pendulum angle (deg)", -30, 30, 12)
x_0 = st.sidebar.slider("Initial cart position (m)", -1.0, 1.0, 0.0, 0.1)
 
st.sidebar.header("\u2696\uFE0F LQR Weights")
q_x = st.sidebar.slider("Q (cart position)", 0.1, 100.0, 1.0, 0.1)
q_theta = st.sidebar.slider("Q (pendulum angle)", 1.0, 1000.0, 10.0, 1.0)
r_u = st.sidebar.slider("R (control effort)", 0.01, 10.0, 0.1, 0.01)
 
st.sidebar.header("\U0001F3AF MPC Settings")
horizon = st.sidebar.slider("Prediction horizon N", 5, 30, 10)
 
st.sidebar.header("\u26A1 Disturbance")
disturb_on = st.sidebar.checkbox("Inject disturbance", value=False)
disturb_t = st.sidebar.slider("Disturbance time (s)", 0.5, 4.0, 2.0, 0.1) if disturb_on else None
disturb_mag = st.sidebar.slider("Disturbance magnitude (N)", -20, 20, 10) if disturb_on else 0
 
t = np.linspace(0, 5, 200)
x0 = [x_0, 0, np.radians(theta_0), 0]
 
with st.spinner("Computing LQR..."):
    Q = np.diag([q_x, 1, q_theta, 1])
    R = np.array([[r_u]])
    K = lqr_gain(Q, R)
    sol_lqr, F_lqr = simulate_lqr(K, t, x0, disturb_t, disturb_mag)
 
with st.spinner("Computing MPC (slower)..."):
    sol_mpc, F_mpc = simulate_mpc(t, x0, N=horizon, disturbance_t=disturb_t, disturbance_mag=disturb_mag)
 
c1,c2,c3,c4 = st.columns(4)
with c1: st.metric("LQR Final \u03B8", f"{np.degrees(sol_lqr[-1,2]):.2f}\u00B0")
with c2: st.metric("MPC Final \u03B8", f"{np.degrees(sol_mpc[-1,2]):.2f}\u00B0")
with c3: st.metric("LQR Max |F|", f"{abs(F_lqr).max():.2f} N")
with c4: st.metric("MPC Max |F|", f"{abs(F_mpc).max():.2f} N")
st.divider()
 
col1, col2 = st.columns(2)
with col1:
    st.subheader("\U0001F4C8 Pendulum Angle vs Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=np.degrees(sol_lqr[:,2]), name="LQR (linear)", line=dict(color="#2563EB",width=2)))
    fig.add_trace(go.Scatter(x=t, y=np.degrees(sol_mpc[:,2]), name="MPC (optimization)", line=dict(color="#DC2626",width=2)))
    fig.add_hline(y=0, line_dash="dash", line_color="green", annotation_text="Setpoint")
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="\u03B8 (deg)", height=350, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
 
with col2:
    st.subheader("\u26A1 Control Force vs Time")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t, y=F_lqr.flatten(), name="LQR force", line=dict(color="#2563EB",width=2)))
    fig2.add_trace(go.Scatter(x=t, y=F_mpc.flatten(), name="MPC force", line=dict(color="#DC2626",width=2)))
    fig2.update_layout(xaxis_title="Time (s)", yaxis_title="Force (N)", height=350, template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)
 
st.subheader("\U0001F3AF Cart Position vs Time")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=t, y=sol_lqr[:,0], name="LQR cart x", line=dict(color="#2563EB",width=2)))
fig3.add_trace(go.Scatter(x=t, y=sol_mpc[:,0], name="MPC cart x", line=dict(color="#DC2626",width=2)))
fig3.update_layout(xaxis_title="Time (s)", yaxis_title="Cart position x (m)", height=300, template="plotly_white")
st.plotly_chart(fig3, use_container_width=True)
 
st.subheader("\U0001F4DA Theory")
st.markdown("""
**LQR (Linear-Quadratic Regulator):** Solves the Riccati equation for the linearized system. Computes a constant gain matrix K. Fast (computed once) but linear approximation only.
 
**MPC (Model Predictive Control):** At each timestep, solves a constrained optimization problem over a prediction horizon. Handles constraints explicitly. Computationally heavier but more flexible.
 
**Key Insight:** For small angles, LQR and MPC perform similarly. For large angles or constrained inputs, MPC outperforms because it accounts for system constraints in the optimization.
""")
st.caption("Nonlinear Control & MPC | Oscar Vincent Dbritto")
