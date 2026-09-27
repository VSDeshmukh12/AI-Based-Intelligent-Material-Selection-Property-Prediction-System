"""
AI-Based Intelligent Material Selection and Mechanical Property
Prediction System - Interactive Dashboard.

Run locally with:
    pip install -r requirements.txt
    streamlit run app.py

This ties together the preprocessing, hybrid ML/DL models, and the
multi-criteria recommendation engine into a single interactive UI, per
Objective #6 of the project (user-friendly dashboard for visualization
and engineering analysis) and Objective #7 (PDF report generation).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from recommend import recommend_materials, load_materials

st.set_page_config(page_title="AI Material Selection System", layout="wide")

DATA_PATH = "data/materials_dataset.csv"
MODELS_DIR = "models"


@st.cache_data
def get_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def get_models():
    models = {}
    for name in ["RandomForest", "GradientBoosting", "ANN_MLP"]:
        path = f"{MODELS_DIR}/{name}.joblib"
        if os.path.exists(path):
            models[name] = joblib.load(path)
    scaler = joblib.load(f"{MODELS_DIR}/scaler.joblib") if os.path.exists(f"{MODELS_DIR}/scaler.joblib") else None
    encoders = joblib.load(f"{MODELS_DIR}/encoders.joblib") if os.path.exists(f"{MODELS_DIR}/encoders.joblib") else None
    return models, scaler, encoders


df = get_data()
models, scaler, encoders = get_models()

st.title("AI-Based Intelligent Material Selection & Property Prediction System")
st.caption("Hybrid Machine Learning + Deep Learning decision-support platform for "
           "mechanical engineering material selection.")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Dataset Explorer", "Property Prediction", "Material Recommendation", "Model Performance",
    "Advanced AHP-TOPSIS", "Uncertainty & SHAP", "Feedback / Closed-Loop",
])

# ---------------- Tab 1: Dataset Explorer ----------------
with tab1:
    st.subheader("Material Dataset")
    cat_filter = st.multiselect("Filter by category", sorted(df["Material_Category"].unique()))
    view = df[df["Material_Category"].isin(cat_filter)] if cat_filter else df
    st.dataframe(view, use_container_width=True, height=350)

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.scatter(view, x="Density_g_cm3", y="Tensile_Strength_MPa",
                           color="Material_Category", hover_name="Material_Name",
                           title="Tensile Strength vs Density")
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.box(view, x="Material_Category", y="Cost_USD_per_kg",
                       title="Cost Distribution by Category")
        fig2.update_xaxes(tickangle=30)
        st.plotly_chart(fig2, use_container_width=True)

# ---------------- Tab 2: Property Prediction ----------------
with tab2:
    st.subheader("Predict Mechanical Properties from Material Attributes")
    if not models:
        st.warning("No trained models found. Run `python src/train_models.py` first.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            category = st.selectbox("Material Category", sorted(df["Material_Category"].unique()))
            density = st.slider("Density (g/cm3)", 0.5, 9.0, 2.7, 0.01)
            melting = st.slider("Melting Point (C)", 100, 3000, 650)
        with col2:
            max_temp = st.slider("Max Service Temperature (C)", 50, 2300, 200)
            cost = st.slider("Cost (USD/kg)", 0.3, 60.0, 3.0, 0.1)
            corrosion = st.selectbox("Corrosion Resistance", ["Low", "Medium", "High"])
        with col3:
            machinability = st.slider("Machinability Index (1-10)", 1.0, 10.0, 7.0, 0.1)
            sustainability = st.slider("Sustainability Score (0-100)", 0, 100, 60)
            recyclable = st.selectbox("Recyclability", ["Yes", "No", "Partial"])

        model_choice = st.selectbox("Model", list(models.keys()), index=0)

        if st.button("Predict Properties"):
            row = pd.DataFrame([{
                "Material_Category": encoders["Material_Category"].transform([category])[0],
                "Density_g_cm3": density,
                "Melting_Point_C": melting,
                "Max_Service_Temp_C": max_temp,
                "Cost_USD_per_kg": cost,
                "Corrosion_Resistance": encoders["Corrosion_Resistance"].transform([corrosion])[0],
                "Machinability_Index_1to10": machinability,
                "Sustainability_Score_0to100": sustainability,
                "Recyclability": encoders["Recyclability"].transform([recyclable])[0],
            }])
            row_scaled = scaler.transform(row)
            pred = models[model_choice].predict(row_scaled)[0]
            target_cols = ["Tensile_Strength_MPa", "Yield_Strength_MPa", "Hardness_HB",
                            "Fatigue_Strength_MPa", "Thermal_Conductivity_W_mK"]
            result = pd.DataFrame({"Property": target_cols, "Predicted Value": np.round(pred, 1)})
            st.table(result)

# ---------------- Tab 3: Material Recommendation ----------------
with tab3:
    st.subheader("Intelligent Material Recommendation")
    st.caption("Set engineering constraints; the engine ranks materials with a "
               "weighted multi-criteria match score.")
    c1, c2, c3 = st.columns(3)
    with c1:
        min_tensile = st.number_input("Min Tensile Strength (MPa)", 0, 2000, 300)
        min_yield = st.number_input("Min Yield Strength (MPa)", 0, 2000, 200)
    with c2:
        max_density = st.number_input("Max Density (g/cm3)", 0.0, 10.0, 5.0)
        max_cost = st.number_input("Max Cost (USD/kg)", 0.0, 100.0, 15.0)
    with c3:
        min_temp = st.number_input("Min Service Temp Capability (C)", 0, 2500, 0)
        top_n = st.slider("Number of results", 3, 20, 10)

    if st.button("Get Recommendations"):
        ranked, err = recommend_materials(
            min_tensile=min_tensile, min_yield=min_yield,
            min_temp_capability=min_temp, max_cost=max_cost,
            max_density=max_density, top_n=top_n, df=df,
        )
        if err:
            st.error(err)
        else:
            st.dataframe(ranked, use_container_width=True)
            fig = px.bar(ranked, x="Material_Name", y="Match_Score", color="Material_Category",
                         title="Recommendation Match Scores")
            fig.update_xaxes(tickangle=30)
            st.plotly_chart(fig, use_container_width=True)

# ---------------- Tab 4: Model Performance ----------------
with tab4:
    st.subheader("Model Performance Comparison")
    metrics_path = "outputs/model_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        rows = []
        for model_name, target_metrics in metrics.items():
            for target, vals in target_metrics.items():
                rows.append({"Model": model_name, "Property": target, **vals})
        mdf = pd.DataFrame(rows)
        fig = px.bar(mdf, x="Property", y="R2", color="Model", barmode="group",
                     title="R2 Score by Model and Property")
        fig.update_xaxes(tickangle=20)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(mdf, use_container_width=True)
    else:
        st.warning("Run `python src/train_models.py` to generate performance metrics.")

    if os.path.exists("outputs/Material_Selection_Report.pdf"):
        with open("outputs/Material_Selection_Report.pdf", "rb") as f:
            st.download_button("Download Engineering PDF Report", f,
                                file_name="Material_Selection_Report.pdf")


# =============================================================================
# RECONSTRUCTED / NEWLY WRITTEN ADDITION (everything below this line)
# Tabs 1-4 above are the original baseline dashboard and are UNCHANGED.
# Tabs 5-7 below integrate the reconstructed advanced modules (AHP-TOPSIS,
# application profiles, EMSI model selection, uncertainty quantification,
# custom Kernel-SHAP, recommendation explainability, risk-aware ranking,
# weight sensitivity, and the feedback / closed-loop prototype). None of
# this UI code was recovered from the original project; it is new,
# written per the documented spec.
# =============================================================================
from application_profiles import get_profile_weights, list_profiles, CRITERIA, BENEFIT_FLAGS
from recommend import recommend_materials_ahp_topsis
from model_selection import load_metrics as emsi_load_metrics, select_best_model_per_property
from uncertainty import predict_with_uncertainty
from shap_explainer import kernel_shap_values
from recommendation_explainer import explain_ranking, compare_top_two
from risk_ranking import risk_aware_rank
from topsis_sensitivity import sensitivity_analysis
from feedback_system import record_feedback, load_feedback
from closed_loop import propose_weight_adjustment
from preprocess import CATEGORICAL_COLS, TARGET_COLS

FEATURE_COLS_9 = [
    "Material_Category", "Density_g_cm3", "Melting_Point_C",
    "Max_Service_Temp_C", "Cost_USD_per_kg", "Corrosion_Resistance",
    "Machinability_Index_1to10", "Sustainability_Score_0to100", "Recyclability",
]


def _encode_rows(rows_df, encoders):
    enc = rows_df[FEATURE_COLS_9].copy()
    for c in CATEGORICAL_COLS:
        enc[c] = encoders[c].transform(enc[c])
    return enc


# ---------------- Tab 5: Advanced AHP-TOPSIS Recommendation ----------------
with tab5:
    st.subheader("Advanced Multi-Criteria Recommendation (AHP + TOPSIS)")
    st.caption("Select a predefined engineering application profile (AHP-derived weights) "
               "and rank candidates with a standard TOPSIS closeness coefficient, "
               "with explainability and a risk-aware variant.")

    profile_name = st.selectbox("Application Profile", list(list_profiles().keys()))
    st.caption(list_profiles()[profile_name])
    weights, ahp_result = get_profile_weights(profile_name)
    with st.expander("AHP weights & consistency check"):
        st.json({k: round(v, 3) for k, v in weights.items()})
        st.write(f"Consistency Ratio (CR): {ahp_result['consistency_ratio']:.3f}  "
                 f"({'consistent' if ahp_result['consistent'] else 'INCONSISTENT - CR >= 0.10'})")

    c1, c2, c3 = st.columns(3)
    with c1:
        at_min_tensile = st.number_input("Min Tensile Strength (MPa)", 0, 2000, 300, key="at_tensile")
        at_min_yield = st.number_input("Min Yield Strength (MPa)", 0, 2000, 200, key="at_yield")
    with c2:
        at_max_density = st.number_input("Max Density (g/cm3)", 0.0, 10.0, 5.0, key="at_density")
        at_max_cost = st.number_input("Max Cost (USD/kg)", 0.0, 100.0, 15.0, key="at_cost")
    with c3:
        at_min_temp = st.number_input("Min Service Temp Capability (C)", 0, 2500, 0, key="at_temp")
        at_top_n = st.slider("Number of results", 3, 20, 10, key="at_topn")

    if st.button("Run AHP-TOPSIS Ranking"):
        ranked, dmatrix, err = recommend_materials_ahp_topsis(
            weights, min_tensile=at_min_tensile, min_yield=at_min_yield,
            min_temp_capability=at_min_temp, max_cost=at_max_cost,
            max_density=at_max_density, top_n=at_top_n, df=df,
        )
        if err:
            st.error(err)
        else:
            st.session_state["at_ranked"] = ranked
            st.session_state["at_dmatrix"] = dmatrix
            st.session_state["at_weights"] = weights
            st.session_state["at_profile"] = profile_name

    if "at_ranked" in st.session_state:
        ranked = st.session_state["at_ranked"]
        dmatrix = st.session_state["at_dmatrix"]
        weights = st.session_state["at_weights"]
        st.dataframe(ranked, use_container_width=True)

        cand_dmatrix = dmatrix.loc[ranked.index]
        labels = ranked["Material_Name"].tolist()
        breakdown, summary = explain_ranking(cand_dmatrix, weights, BENEFIT_FLAGS, material_labels=labels)
        st.markdown(f"**Why this top pick?** {summary}")
        st.dataframe(breakdown, use_container_width=True)
        st.info(compare_top_two(ranked, cand_dmatrix, weights, BENEFIT_FLAGS))

        colA, colB = st.columns(2)
        with colA:
            if st.button("Run Risk-Aware Ranking (advisory)"):
                if not models or scaler is None or encoders is None:
                    st.warning("Models/scaler/encoders not available - cannot compute predicted "
                               "lower-bound properties for risk-aware ranking.")
                elif "RandomForest" not in models:
                    st.warning("RandomForest model not available - risk-aware ranking requires "
                               "per-tree uncertainty from RandomForest.")
                else:
                    cand_rows = df[df["Material_ID"].isin(ranked.index)].set_index("Material_ID").loc[ranked.index]
                    X_cand = _encode_rows(cand_rows, encoders)
                    X_cand_scaled = scaler.transform(X_cand)
                    unc = predict_with_uncertainty(models["RandomForest"], X_cand_scaled, TARGET_COLS)
                    lower_bounds = {
                        "Tensile_Strength_MPa": unc["Tensile_Strength_MPa"]["lower"],
                        "Yield_Strength_MPa": unc["Yield_Strength_MPa"]["lower"],
                    }
                    risk_ranked, comparison = risk_aware_rank(cand_dmatrix, weights, BENEFIT_FLAGS, lower_bounds)
                    st.dataframe(risk_ranked, use_container_width=True)
                    if comparison["top1_changed"]:
                        st.warning(f"Risk-aware top pick differs from primary: "
                                   f"{comparison['primary_top1']} -> {comparison['risk_aware_top1']}. "
                                   f"Advisory only; does not change the primary ranking above.")
                    else:
                        st.success("Risk-aware ranking agrees with the primary top-1 pick. Advisory only.")
        with colB:
            if st.button("Run Weight Sensitivity Analysis"):
                sens_df, sens_summary = sensitivity_analysis(cand_dmatrix, weights, BENEFIT_FLAGS)
                st.write(f"Top-1 stability rate under +/-20% weight perturbations: "
                         f"{sens_summary['stability_rate']*100:.0f}%")
                st.dataframe(sens_df, use_container_width=True)

# ---------------- Tab 6: Uncertainty & SHAP ----------------
with tab6:
    st.subheader("Prediction Uncertainty & SHAP Explainability")
    if not models or scaler is None or encoders is None:
        st.warning("No trained models found. Run `python src/train_models.py` first.")
    else:
        st.caption("EMSI (Engineering Model Selection Index) picks, per predicted property, "
                   "whichever trained model scores best on held-out R2/MAE/RMSE.")
        try:
            metrics = emsi_load_metrics()
            best_per_prop = select_best_model_per_property(metrics)
            st.json(best_per_prop)
        except FileNotFoundError as e:
            st.warning(str(e))
            best_per_prop = None

        st.divider()
        st.caption("Pick a material row to see a Random-Forest prediction interval "
                   "(tree-percentile band) and a Kernel-SHAP feature-attribution breakdown "
                   "for Tensile Strength.")
        material_choice = st.selectbox("Material row (by ID)", df["Material_ID"].tolist())
        if st.button("Explain Prediction"):
            row = df[df["Material_ID"] == material_choice]
            X_row = _encode_rows(row, encoders)
            X_row_scaled = scaler.transform(X_row)

            if "RandomForest" in models:
                unc = predict_with_uncertainty(models["RandomForest"], X_row_scaled, TARGET_COLS)
                ts = unc["Tensile_Strength_MPa"]
                st.write(f"Tensile Strength prediction interval: "
                         f"{ts['lower'][0]:.1f} - {ts['upper'][0]:.1f} MPa "
                         f"(mean {ts['mean'][0]:.1f} MPa). "
                         f"{'Width uses RMSE-based calibration.' if ts['calibrated'] else 'Uncalibrated (metrics unavailable).'}")

                bg = np.zeros(X_row_scaled.shape[1], dtype=float)  # background = this row's own scaled features as fallback
                shap_out = kernel_shap_values(
                    models["RandomForest"].predict, bg, X_row_scaled[0], target_index=0,
                )
                shap_df = pd.DataFrame({
                    "Feature": FEATURE_COLS_9,
                    "SHAP_Value": shap_out["phi"],
                }).sort_values("SHAP_Value", key=abs, ascending=False)
                st.dataframe(shap_df, use_container_width=True)
                st.caption(f"SHAP method: {shap_out['method']} enumeration over "
                           f"{shap_out['n_coalitions']} coalitions. "
                           f"Base value {shap_out['base_value']:.1f} + sum(SHAP) = "
                           f"{shap_out['base_value'] + shap_out['phi'].sum():.1f} "
                           f"(model output {shap_out['f_x']:.1f}).")
            else:
                st.warning("RandomForest model not available for uncertainty/SHAP.")

# ---------------- Tab 7: Feedback / Closed-Loop ----------------
with tab7:
    st.subheader("Recommendation Feedback (prototype closed loop)")
    st.caption("Rate a recommended material as accepted/rejected. This is advisory: "
               "it proposes a weight nudge for review, it never auto-applies changes.")

    if "at_ranked" not in st.session_state:
        st.info("Run the Advanced AHP-TOPSIS tab first to generate recommendations to rate.")
    else:
        ranked = st.session_state["at_ranked"]
        profile_name = st.session_state["at_profile"]
        weights = st.session_state["at_weights"]
        dmatrix = st.session_state["at_dmatrix"]

        fb_choice = st.selectbox("Material to rate", ranked.index.tolist(),
                                  format_func=lambda mid: f"{mid} - {ranked.loc[mid, 'Material_Name']}")
        rank_val = int(ranked.loc[fb_choice, "TOPSIS_Rank"])
        rating = st.radio("Rating", ["accept", "reject"], horizontal=True)
        comment = st.text_input("Optional comment")
        if st.button("Submit Feedback"):
            record_feedback(profile_name, fb_choice, rank_val, rating, comment)
            st.success("Feedback recorded to outputs/feedback_log.csv")

        st.divider()
        fb_log = load_feedback()
        if not fb_log.empty:
            st.dataframe(fb_log, use_container_width=True)
            if st.button("Propose Weight Adjustment from Feedback"):
                cand_dmatrix = dmatrix.loc[ranked.index]
                proposed, explanation = propose_weight_adjustment(weights, cand_dmatrix, BENEFIT_FLAGS)
                st.write(explanation)
                st.json({k: round(v, 3) for k, v in proposed.items()})
        else:
            st.caption("No feedback recorded yet.")
