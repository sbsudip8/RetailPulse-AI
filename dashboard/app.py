import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="RetailPulse",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RetailPulse — Customer Analytics & Demand Forecasting")
st.caption("AI-Powered Retail Intelligence Platform")

# ── helper to load CSVs safely ──────────────────────────────────────────────
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA = BASE_DIR / "data"

def load(filename):
    path = DATA / filename
    if path.exists():
        return pd.read_csv(path)
    return None

# Dashboard Overview
daily = load("daily_sales.csv")
rfm = load("rfm_segmented.csv")
churn = load("churn_scores.csv")

if daily is not None and rfm is not None and churn is not None:

    high_risk = (churn["ChurnProba"] >= 0.5).sum()

    st.subheader("📊 Business Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Revenue",
        f"£{daily['Revenue'].sum():,.0f}"
    )

    col2.metric(
        "Avg Daily Revenue",
        f"£{daily['Revenue'].mean():,.0f}"
    )

    col3.metric(
        "Customers",
        len(rfm)
    )

    col4.metric(
        "High Risk Customers",
        high_risk
    )

    if high_risk > 100:
        st.error(
            f"⚠️ Alert: {high_risk} customers are at high churn risk!"
        )

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Demand Forecast",
    "👥 Customer Segments",
    "⚠️ Churn Risk",
    "📦 Inventory"
])
# ── TAB 1: Demand Forecasting ────────────────────────────────────────────────
with tab1:
    st.header("Demand Forecasting")

    daily = load("daily_sales.csv")
    forecast = load("prophet_forecast.csv")

    if daily is None:
        st.warning("daily_sales.csv not found. Run notebook 02_cleaning.ipynb first.")
    else:
        daily["Date"] = pd.to_datetime(daily["Date"])

        # What-if slider
        st.subheader("What-if Analysis")
        col1, col2 = st.columns([1, 3])
        with col1:
            growth = st.slider("Expected growth %", -30, 50, 0, step=5)
            horizon = st.slider("Forecast horizon (days)", 7, 60, 30)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily["Date"], y=daily["Revenue"],
                mode="lines", name="Actual Sales",
                line=dict(color="#7F77DD", width=1.5)
            ))
            if "Rolling7" in daily.columns:
                fig.add_trace(go.Scatter(
                    x=daily["Date"], y=daily["Rolling7"],
                    mode="lines", name="7-day Rolling Avg",
                    line=dict(color="#1D9E75", width=2, dash="dash")
                ))

            # Prophet forecast overlay
            if forecast is not None:
                forecast["ds"] = pd.to_datetime(forecast["ds"])
                future = forecast[forecast["ds"] > daily["Date"].max()].head(horizon).copy()
                future["yhat_adjusted"] = future["yhat"] * (1 + growth / 100)
                fig.add_trace(go.Scatter(
                    x=future["ds"], y=future["yhat_adjusted"],
                    mode="lines", name=f"Forecast (+{growth}%)",
                    line=dict(color="#EF9F27", width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=pd.concat([future["ds"], future["ds"][::-1]]),
                    y=pd.concat([future["yhat_upper"], future["yhat_lower"][::-1]]),
                    fill="toself", fillcolor="rgba(239,159,39,0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Confidence interval"
                ))

            fig.update_layout(
                height=380, margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

        # Key metrics
        st.subheader("Key Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Revenue", f"£{daily['Revenue'].sum():,.0f}")
        m2.metric("Avg Daily Revenue", f"£{daily['Revenue'].mean():,.0f}")
        m3.metric("Peak Day", f"£{daily['Revenue'].max():,.0f}")
        m4.metric("Forecast Horizon", f"{horizon} days")

        if forecast is not None:
            csv = forecast.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Download Forecast CSV",
                csv,
                "prophet_forecast.csv",
                "text/csv"
            )

# ── TAB 2: Customer Segments ─────────────────────────────────────────────────
with tab2:
    st.header("Customer Segmentation")

    rfm = load("rfm_segmented.csv")

    if rfm is None:
        st.warning("rfm_segmented.csv not found. Run notebook 03_segmentation.ipynb first.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            # Segment distribution pie
            seg_counts = rfm["Segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Count"]
            fig_pie = px.pie(
                seg_counts, names="Segment", values="Count",
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Customer Distribution by Segment"
            )
            fig_pie.update_layout(height=340, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # RFM scatter
            fig_scatter = px.scatter(
                rfm, x="Recency", y="Monetary",
                color="Segment", size="Frequency",
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="RFM Scatter (size = Frequency)",
                labels={"Recency": "Recency (days)", "Monetary": "Monetary (£)"}
            )
            fig_scatter.update_layout(height=340, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Segment summary table
        st.subheader("Segment Summary")
        summary = rfm.groupby("Segment").agg(
            Customers=("CustomerID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean")
        ).round(1).reset_index()
        summary.columns = ["Segment", "Customers", "Avg Recency (days)", "Avg Orders", "Avg Revenue (£)"]
        st.dataframe(summary, use_container_width=True, hide_index=True)

        # Segment filter
        st.subheader("Browse Customers by Segment")
        selected_seg = st.selectbox("Select segment", rfm["Segment"].unique())
        filtered = rfm[rfm["Segment"] == selected_seg][["CustomerID","Recency","Frequency","Monetary","Segment"]]
        st.dataframe(filtered.head(50), use_container_width=True, hide_index=True)

        csv = rfm.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Segmentation CSV",
            csv,
            "rfm_segmented.csv",
            "text/csv"
        )

        # ── TAB 3: Churn Risk ────────────────────────────────────────────────────────
with tab3:
    st.header("Churn Risk Analysis")

    churn = load("churn_scores.csv")

    if churn is None:
        st.warning("churn_scores.csv not found. Run notebook 08_churn.ipynb first.")
    else:
        # Risk threshold slider
        threshold = st.slider("Churn risk threshold", 0.3, 0.9, 0.5, step=0.05)
        churn["Risk Level"] = churn["ChurnProba"].apply(
            lambda x: "🔴 High" if x >= threshold else ("🟡 Medium" if x >= threshold * 0.6 else "🟢 Low")
        )

        col1, col2, col3 = st.columns(3)
        high_risk = (churn["ChurnProba"] >= threshold).sum()
        col1.metric("High Risk Customers", high_risk,
                    delta=f"{high_risk/len(churn)*100:.1f}% of base", delta_color="inverse")
        col2.metric("Total Customers", len(churn))
        col3.metric("Avg Churn Probability", f"{churn['ChurnProba'].mean():.2%}")

        col_a, col_b = st.columns([1, 1])

        with col_a:
            # Churn probability histogram
            fig_hist = px.histogram(
                churn, x="ChurnProba", nbins=30,
                title="Churn Probability Distribution",
                color_discrete_sequence=["#D85A30"]
            )
            fig_hist.add_vline(x=threshold, line_dash="dash",
                               line_color="#534AB7", annotation_text="Threshold")
            fig_hist.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_b:
            # Risk level bar
            risk_counts = churn["Risk Level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            fig_bar = px.bar(
                risk_counts, x="Risk Level", y="Count",
                title="Customers by Risk Level",
                color="Risk Level",
                color_discrete_map={
                    "🔴 High": "#D85A30",
                    "🟡 Medium": "#EF9F27",
                    "🟢 Low": "#1D9E75"
                }
            )
            fig_bar.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0),
                                  showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # High risk customer table
        st.subheader("High Risk Customers (Action Required)")
        high_risk_df = churn[churn["ChurnProba"] >= threshold].sort_values(
            "ChurnProba", ascending=False
        )[["CustomerID","Recency","Frequency","Monetary","ChurnProba"]].head(20)
        high_risk_df["ChurnProba"] = high_risk_df["ChurnProba"].apply(lambda x: f"{x:.2%}")
        st.dataframe(high_risk_df, use_container_width=True, hide_index=True)

        # Export button
        csv = churn.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export churn scores CSV", csv,
                           "churn_scores.csv", "text/csv")
        
        # ── TAB 4: Inventory Optimization ────────────────────────────────────────────
with tab4:
    st.header("Inventory Optimization")

    inv = load("inventory_recommendations.csv")
    forecast = load("prophet_forecast.csv")

    if inv is None:
        st.warning("inventory_recommendations.csv not found. Run notebook 09_inventory.ipynb first.")
    else:
        # Show key metrics from saved file
        inv_dict = dict(zip(inv["Metric"], inv["Value"]))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Daily Demand", f"£{inv_dict.get('Avg Daily Demand', 0):,.0f}")
        col2.metric("Safety Stock",     f"£{inv_dict.get('Safety Stock', 0):,.0f}")
        col3.metric("Reorder Point",    f"£{inv_dict.get('Reorder Point', 0):,.0f}")
        col4.metric("Recommended Order",f"£{inv_dict.get('Reorder Qty', 0):,.0f}")

    # What-if inventory simulator
    st.subheader("Inventory Simulator")
    col_s1, col_s2 = st.columns([1, 2])

    with col_s1:
        sim_demand  = st.number_input("Avg daily demand (£)", value=10000, step=500)
        lead_time   = st.slider("Lead time (days)", 1, 14, 3)
        safety_mult = st.slider("Safety stock multiplier", 1.0, 3.0, 1.5, step=0.1)

    safety_stock   = sim_demand * safety_mult
    reorder_point  = (sim_demand * lead_time) + safety_stock
    order_qty      = sim_demand * 30

    with col_s2:
        fig_inv = go.Figure()
        days = list(range(0, 35))
        stock_level = [order_qty - sim_demand * d for d in days]
        fig_inv.add_trace(go.Scatter(
            x=days, y=stock_level, mode="lines+markers",
            name="Stock level", line=dict(color="#7F77DD", width=2)
        ))
        fig_inv.add_hline(y=reorder_point, line_dash="dash",
                          line_color="#D85A30",
                          annotation_text=f"Reorder point £{reorder_point:,.0f}")
        fig_inv.add_hline(y=safety_stock, line_dash="dot",
                          line_color="#EF9F27",
                          annotation_text=f"Safety stock £{safety_stock:,.0f}")
        fig_inv.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Days", yaxis_title="Stock Level (£)"
        )
        st.plotly_chart(fig_inv, use_container_width=True)

        csv = inv.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Inventory Report",
        csv,
        "inventory_recommendations.csv",
        "text/csv"
    )

        # ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/RetailPulse-v1.0-purple")
    st.markdown("### About")
    st.markdown(
        "RetailPulse is an AI-powered retail analytics platform built on the "
        "UCI Online Retail dataset (~541K transactions)."
    )
    st.markdown("### Data Status")

    for fname, label in [
        ("daily_sales.csv",            "Daily sales"),
        ("rfm_segmented.csv",          "RFM segments"),
        ("churn_scores.csv",           "Churn scores"),
        ("inventory_recommendations.csv", "Inventory"),
        ("prophet_forecast.csv",       "Forecast"),
    ]:
        exists = (DATA / fname).exists()
        st.markdown(f"{'✅' if exists else '❌'} {label}")

    st.markdown("---")
    st.caption("Built with Streamlit · MLflow · XGBoost · Prophet")

