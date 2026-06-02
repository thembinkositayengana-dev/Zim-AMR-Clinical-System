import streamlit as st
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
from model import train_inference_engine, clean_hospital_data

# --- 1. SYSTEM SETTINGS & LOGIN ---
st.set_page_config(page_title="Zim-AMR Clinical System v2.5", layout="wide")

if "auth" not in st.session_state: st.session_state["auth"] = False

if not st.session_state["auth"]:
    st.title("🔐 Secure Stewardship Portal")
    st.subheader("Clinical Decision Support System | Version 2.5")
    pwd = st.text_input("Clinician Access Code:", type="password")
    if st.button("Login"):
        if pwd == "ZimDoc2025":
            st.session_state["auth"] = True
            st.rerun()
        else:
            st.error("Invalid Credentials.")
    st.stop()

# --- 2. DATA LOADING ---
model, columns, metrics = train_inference_engine()

# RESTORED: Full Stewardship Database
AWARE_DATA = {
    "Amoxicillin": {"group": "ACCESS", "color": "#28a745", "msg": "First-line antibiotic; low resistance potential."},
    "Gentamicin": {"group": "ACCESS", "color": "#28a745", "msg": "Recommended for severe Gram-negative infections."},
    "Ceftriaxone": {"group": "WATCH", "color": "#fd7e14", "msg": "Watch group; monitor for local resistance patterns."},
    "Ciprofloxacin": {"group": "WATCH", "color": "#fd7e14", "msg": "High resistance potential; restrict use strictly."},
    "Meropenem": {"group": "RESERVE", "color": "#dc3545", "msg": "RESERVED for confirmed multi-drug resistance."}
}

# --- 3. SIDEBAR ---
st.sidebar.title("🏥 Informatics Portal")
st.sidebar.markdown(f"**Lead:** T. Tayengana")
st.sidebar.markdown(f"**Registry ID:** R224976A")
st.sidebar.markdown("---")
# RESTORED: Amount of records studied
st.sidebar.info(f"📊 Training Records: {metrics['total_n']}")
st.sidebar.success(f"✅ F1-CV Score: {metrics['cv_f1']:.1f}%")
if st.sidebar.button("Logout"):
    st.session_state["auth"] = False
    st.rerun()

# --- 4. TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🩺 Clinical Analysis", "📊 Scientific Validation", "📜 Full Audit Logs", "🔬 Lab Surveillance"])

with tab1:
    st.title("🛡️ Antibiotic Guidance Engine")
    st.caption("Integrated Framework for Zimbabwe | v2.5")
    c_in, c_out = st.columns([1, 1])
    
    with c_in:
        st.subheader("📋 Case Parameters")
        p_id = st.text_input("Patient ID / MRN", value="PGH-")
        p_name = st.text_input("Full Name")
        ca, cb = st.columns(2)
        with ca:
            p_age = st.number_input("Age", 0, 115, 30)
            p_month = st.slider("Month of Admission", 1, 12, datetime.datetime.now().month)
            admission = st.selectbox("Admission Status", ["Walk-in", "Hospitalized > 48hrs"])
        with cb:
            p_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            p_year = st.number_input("Consultation Year", 2023, 2090, 2026) 
            prev_res = st.selectbox("Previous Resistance?", ["No", "Yes", "Unknown"])
        
        p_ward = st.selectbox("Ward", ["General Medicine", "ICU", "Neonatal Ward", "Maternity", "Outpatient"])
        org = st.selectbox("Suspected Organism", ["E_coli", "Klebsiella pneumoniae", "S. aureus (MRSA)", "Pseudomonas"])
        spec = st.selectbox("Specimen Source", ["Urine", "Blood", "Wound", "Sputum", "CSF"])
        anti = st.selectbox("Proposed Antibiotic", list(AWARE_DATA.keys()))

    with c_out:
        st.subheader("💡 Decision Support Output")
        
        # RESTORED: Stewardship Badge Logic
        steward = AWARE_DATA.get(anti)
        st.markdown(f"""
            <div style="background-color:{steward['color']}; padding:15px; border-radius:10px; color:white; text-align:center; font-weight:bold;">
                WHO STEWARDSHIP: {steward['group']}
            </div>
        """, unsafe_allow_html=True)
        st.caption(f"**Clinical Guideline:** {steward['msg']}")

        if st.button("🚀 Run Prediction Engine"):
            if p_id and p_name:
                input_df = pd.DataFrame([{
                    'Organism': org.capitalize(), 'Specimen': spec.capitalize(), 'Age': p_age, 
                    'Gender': p_gender.capitalize(), 'Ward': p_ward.capitalize(), 
                    'Admission_Status': admission.capitalize(), 'Prev_Resistance': prev_res.capitalize(), 
                    'Antibiotic': anti.capitalize(), 'Month': p_month, 'Year': p_year
                }])
                input_df_encoded = pd.get_dummies(input_df).reindex(columns=columns, fill_value=0)
                prob = model.predict_proba(input_df_encoded)[0][1]
                
                if prob > 0.6: st.success(f"### RESULT: PREDICTED SUSCEPTIBLE ({prob*100:.1f}%)")
                else: st.error(f"### RESULT: HIGH RISK OF RESISTANCE ({(1-prob)*100:.1f}%)")
                
                # Ranking Engine
                st.subheader("📊 Therapy Ranking")
                rank_list = []
                for drug in AWARE_DATA.keys():
                    temp_df = pd.DataFrame([{'Organism': org.capitalize(), 'Specimen': spec.capitalize(), 'Age': p_age, 'Gender': p_gender.capitalize(), 'Ward': p_ward.capitalize(), 'Admission_Status': admission.capitalize(), 'Prev_Resistance': prev_res.capitalize(), 'Antibiotic': drug.capitalize(), 'Month': p_month, 'Year': p_year}])
                    temp_df_encoded = pd.get_dummies(temp_df).reindex(columns=columns, fill_value=0)
                    p_succ = model.predict_proba(temp_df_encoded)[0][1]
                    rank_list.append({"Antibiotic": drug, "Raw": p_succ, "Success Prob": f"{p_succ*100:.1f}%", "Group": AWARE_DATA[drug]['group']})
                st.table(pd.DataFrame(rank_list).sort_values(by="Raw", ascending=False)[["Antibiotic", "Success Prob", "Group"]])

                # Clinical Audit
                log_entry = {"Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "ID": p_id, "Name": p_name, "Drug": anti, "Result": "S" if prob > 0.6 else "R"}
                pd.DataFrame([log_entry]).to_csv("full_clinical_history.csv", mode='a', index=False, header=not os.path.exists("full_clinical_history.csv"))
            else: st.warning("Complete Patient Identification first.")

    st.markdown("---")
    st.warning("⚠️ **DISCLAIMER:** Guidance based on retrospective data. Final authority rests with the physician.")

with tab2:
    st.title("📊 Scientific Validation")
    # METRICS ROW (Including F1)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['accuracy']:.1f}%")
    m2.metric("Sensitivity", f"{metrics['sensitivity']:.1f}%")
    m3.metric("Specificity", f"{metrics['specificity']:.1f}%")
    m4.metric("F1-Score", f"{metrics['f1']:.1f}%")
    m5.metric("AUROC", f"{metrics['auroc']:.2f}")

    col_cm, col_chart = st.columns([1, 2])
    with col_cm:
        st.subheader("📋 Confusion Matrix")
        tn, fp, fn, tp = metrics['cm']
        st.table(pd.DataFrame([[tn, fp], [fn, tp]], index=['Actual Res', 'Actual Susc'], columns=['Pred Res', 'Pred Susc']))
    with col_chart:
        st.subheader("🔍 Feature Importance")
        importances = pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False).head(10)
        st.bar_chart(importances)

with tab3:
    st.title("📜 Full System Audit Logs")
    if os.path.exists("full_clinical_history.csv"):
        st.dataframe(pd.read_csv("full_clinical_history.csv"), width=1500)

with tab4:
    st.title("🔬 Lab Surveillance & Ingestion")
    uploaded_file = st.file_uploader("Upload 2023-2025 Lab Export (CSV)", type="csv")
    if uploaded_file and st.button("⚙️ Process and Train"):
        df_new = pd.read_csv(uploaded_file)
        df_new.to_csv("data.csv", index=False)
        st.cache_data.clear(); st.cache_resource.clear()
        st.success("✅ retrained on new local Patterns!"); st.rerun()

    if os.path.exists("data.csv"):
        st.markdown("---")
        df_lab = clean_hospital_data(pd.read_csv("data.csv"))
        res_trends = df_lab.groupby('Antibiotic')['Target'].apply(lambda x: (1 - x.mean())*100)
        fig, ax = plt.subplots(figsize=(10, 5))
        res_trends.plot(kind='bar', ax=ax, color='#d62728')
        ax.set_ylabel("Resistance Rate (%)")
        st.pyplot(fig)
        st.metric("Total Records Ingested", len(df_lab))