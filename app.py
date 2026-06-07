import streamlit as st
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
from model import train_inference_engine, clean_hospital_data

# --- 1. SYSTEM SETTINGS & LOGIN ---
st.set_page_config(
    page_title="Zim-AMR Clinical System v3.0",
    layout="wide"
)

# Session State
if "auth" not in st.session_state:
    st.session_state["auth"] = False

if "user" not in st.session_state:
    st.session_state["user"] = None

if "role" not in st.session_state:
    st.session_state["role"] = None

if "accepted_policy" not in st.session_state:
    st.session_state["accepted_policy"] = False


def write_audit(username, role, action, status):
    audit_entry = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Username": username,
        "Role": role,
        "Action": action,
        "Status": status
    }

    pd.DataFrame([audit_entry]).to_csv(
        "system_audit_log.csv",
        mode="a",
        index=False,
        header=not os.path.exists("system_audit_log.csv")
    )


# LOGIN SCREEN
if not st.session_state["auth"]:

    st.title("🔐 Zimbabwe AMR Clinical Decision Support System")
    st.subheader("Secure Stewardship Portal")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if os.path.exists("users.csv"):

            users_df = pd.read_csv("users.csv")
            match = users_df[
                (users_df["username"] == username)
                &
                (users_df["password"] == password)
                &
                (users_df["status"] == "Active")
            ]
            if len(match) > 0:
                st.session_state["auth"] = True
                st.session_state["user"] = match.iloc[0]["username"]
                st.session_state["role"] = match.iloc[0]["role"]
                write_audit(
                    username,
                    match.iloc[0]["role"],
                    "LOGIN",
                    "SUCCESS"
                )
                st.rerun()

            else:

                write_audit(
                    username,
                    "Unknown",
                    "FAILED_LOGIN",
                    "FAILED"
                )

                st.error("Invalid username or password")

        else:
            st.error("users.csv not found")

    st.stop()


# DATA PROTECTION NOTICE
if not st.session_state["accepted_policy"]:

    st.warning(
        '''
        Cyber and Data Protection Act [Chapter 12:07]

        This system contains confidential patient
        and laboratory information.

        Unauthorized access, disclosure,
        modification or misuse of patient data
        is prohibited.

        By continuing, you agree to comply
        with institutional policies and
        Zimbabwean data protection regulations.
        '''
    )

    if st.button("I Acknowledge and Accept"):

        st.session_state["accepted_policy"] = True

        write_audit(
            st.session_state["user"],
            st.session_state["role"],
            "POLICY_ACCEPTED",
            "SUCCESS"
        )

        st.rerun()



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

st.sidebar.markdown("---")

st.sidebar.success(
    f"👤 User: {st.session_state['user']}"
)

st.sidebar.info(
    f"🛡️ Role: {st.session_state['role']}"
)

st.sidebar.markdown("---")

st.sidebar.metric(
    "Training Records",
    metrics["total_n"]
)

st.sidebar.metric(
    "Cross Validation F1",
    f"{metrics['cv_f1']:.1f}%"
)

st.sidebar.markdown("---")

if st.sidebar.button("🚪 Logout"):

    write_audit(
        st.session_state["user"],
        st.session_state["role"],
        "LOGOUT",
        "SUCCESS"
    )

    st.session_state["auth"] = False
    st.session_state["accepted_policy"] = False
    st.session_state["user"] = None
    st.session_state["role"] = None

    st.rerun()
   
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🩺 Clinical Analysis",
    "📊 Scientific Validation",
    "📜 Audit Dashboard",
    "🔬 Lab Surveillance",
    "👥 User Management",
    "📋 Prediction History",
    "📈 Stewardship Analytics"
])


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

        
                prediction_entry = {

                    "Timestamp":
                        datetime.datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),

                    "Clinician":
                        st.session_state["user"],

                    "Patient_ID":
                        p_id,

                    "Patient_Name":
                        p_name,

                    "Antibiotic":
                        anti,

                    "Prediction":
                        "Susceptible"
                        if prob > 0.6
                        else "Resistant",

                    "Probability":
                        round(prob * 100, 2)
                }
                pd.DataFrame(
                    [prediction_entry]
                      ).to_csv(
                      "clinical_prediction_history.csv",
                         mode="a",
                         index=False,
                         header=False,
                          lineterminator="\n"
                            )

                write_audit(
                    st.session_state["user"],
                    st.session_state["role"],
                    "CLINICAL_PREDICTION",
                    "SUCCESS"
                )

                st.session_state["report_text"] = f"""
                                  CLINICAL DECISION SUPPORT REPORT

                                   Patient ID: {p_id}
                                      Patient Name: {p_name}
                                    Age: {p_age}
                         Gender: {p_gender}

                           Organism: {org}
                                Specimen: {spec}

                      Antibiotic: {anti}

                      Prediction: {"Susceptible" if prob > 0.6 else "Resistant"}

                         Probability: {prob*100:.2f}%

                          WHO Group: {AWARE_DATA[anti]["group"]}

                            Clinician: {st.session_state["user"]}
                                  """                   

            else:
                st.warning(
                    "Complete Patient Identification first."
                )

            if "report_text" in st.session_state:
                 st.download_button(
                     "📄 Download Clinical Report",
                         st.session_state["report_text"],
                             file_name="clinical_report.txt"
                                     )
               

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
        st.markdown("---")

    st.subheader("🧹 Data Quality Summary")

    if "cleaning_report" in metrics:

        report = metrics["cleaning_report"]

        st.json(report)

with tab3:

    st.title("📜 System Audit Dashboard")

    if os.path.exists("system_audit_log.csv"):

        audit_df = pd.read_csv(
            "system_audit_log.csv"
        )

        st.metric(
            "Total Audit Events",
            len(audit_df)
        )

        st.markdown("---")

        st.dataframe(
            audit_df,
            use_container_width=True
        )

    else:

        st.warning(
            "No audit logs available."
        )



   
        
with tab4:

    st.title("🔬 Lab Surveillance & Ingestion")

    uploaded_file = st.file_uploader(
        "Upload 2023-2025 Lab Export (CSV)",
        type="csv"
    )

    if uploaded_file and st.button("⚙️ Process and Train"):

        df_new = pd.read_csv(uploaded_file)

        df_new.to_csv("data.csv", index=False)

        st.cache_data.clear()
        st.cache_resource.clear()

        st.success("✅ Retrained on new local patterns!")

        st.rerun()

    if os.path.exists("data.csv"):

        df_lab, cleaning_report = clean_hospital_data(
            pd.read_csv("data.csv")
        )

        st.markdown("---")

        res_trends = df_lab.groupby(
            "Antibiotic"
        )["Target"].apply(
            lambda x: (1 - x.mean()) * 100
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        res_trends.plot(kind="bar", ax=ax)

        st.pyplot(fig)

        st.metric(
            "Total Records Ingested",
            len(df_lab)
        )

        st.subheader("📊 Data Quality Report")

        st.json(cleaning_report)

        st.subheader("📋 Dataset Column Validation")
        expected_columns = [
           "Organism",
           "Specimen",
                "Age",
             "Gender",
               "Ward",
                  "Admission_Status",
             "Prev_Resistance",
                 "Antibiotic",
                       "Result"
                        ]
        validation = []
    for col in expected_columns:

       validation.append({
                   "Column": col,
                   "Present in Dataset":
                 col in df_lab.columns
                       })

    st.dataframe(
                      pd.DataFrame(validation),
                         use_container_width=True
                           )


with tab5:

    st.title("👥 User Management")
    st.write(
        f"Logged in as: {st.session_state['user']}"
    )
    st.write(
        f"Role: {st.session_state['role']}"
    )
    st.markdown("---")

    if st.session_state["role"] != "Administrator":
        st.error(
            "Access Denied. Administrator privileges required."
        )

    else:

        st.success(
            "Administrator Access Granted"
        )

        st.markdown("---")

        st.subheader("➕ Create New User")

        col1, col2 = st.columns(2)

        with col1:

         new_username = st.text_input(
          "New Username"
      )

         new_password = st.text_input(
        "New Password",
        type="password"
    )

        with col2:

         new_role = st.selectbox(
        "Role",
        [
            "Clinician",
            "Laboratory Scientist",
            "Research Officer"
        ]
    )
        

        if st.button("Create User"):

            users_df = pd.read_csv(
                "users.csv"
            )

            if (
                new_username
                in users_df["username"].values
            ):

                st.warning(
                    "Username already exists."
                )

            else:

                new_user = pd.DataFrame([{
                    "username": new_username,
                    "password": new_password,
                    "role": new_role,
                    "status": "Active"
                }])

                users_df = pd.concat(
                    [users_df, new_user],
                    ignore_index=True
                )

                users_df.to_csv(
                    "users.csv",
                    index=False
                )

                write_audit(
                    st.session_state["user"],
                    st.session_state["role"],
                    f"CREATE_USER:{new_username}",
                    "SUCCESS"
                )

                st.success(
                    "User created successfully."
                )

        st.markdown("---")

        st.subheader("🗑️ Delete User")

        users_df = pd.read_csv(
            "users.csv"
        )

        delete_user = st.selectbox(
            "Select User",
            users_df["username"]
        )

        if st.button("Delete User"):

            if delete_user == "ttayengana":

                st.error(
                    "Administrator account cannot be deleted."
                )

            else:

                users_df = users_df[
                    users_df["username"] != delete_user
                ]

                users_df.to_csv(
                    "users.csv",
                    index=False
                )

                write_audit(
                    st.session_state["user"],
                    st.session_state["role"],
                    f"DELETE_USER:{delete_user}",
                    "SUCCESS"
                )

                st.success(
                    "User deleted successfully."
                )
                
        st.markdown("---")

        st.subheader("🔑 Reset Password")

        users_df = pd.read_csv("users.csv")

        reset_user = st.selectbox(
            "Select User",
            users_df["username"],
            key="reset_user"
        )

        new_password = st.text_input(
            "New Password",
            type="password",
            key="new_password"
        )

        if st.button("Reset Password"):

            users_df.loc[
                users_df["username"] == reset_user,
                  "password"
                   ] = new_password,

            users_df.to_csv(
                "users.csv",
                index=False
            )

            write_audit(
                st.session_state["user"],
                st.session_state["role"],
                f"PASSWORD_RESET:{reset_user}",
                "SUCCESS"
            )

            st.success(
                "Password updated successfully."
            )

        st.markdown("---")

        st.subheader("🔒 User Status Management")

        selected_user = st.selectbox(
            "Select User",
            users_df["username"],
            key="status_user"
        )

        new_status = st.selectbox(
            "Status",
            ["Active", "Disabled"],
            key="status_select"
        )

        if st.button("Update Status"):

            users_df.loc[
                users_df["username"] == selected_user,
                "status"
            ] = new_status

            users_df.to_csv(
                "users.csv",
                index=False
            )

            write_audit(
                st.session_state["user"],
                st.session_state["role"],
                f"STATUS_CHANGE:{selected_user}:{new_status}",
                "SUCCESS"
            )

            st.success(
                "User status updated."
            )

        st.markdown("---")

        st.subheader("📋 Registered Users")

        st.dataframe(
            pd.read_csv("users.csv"),
            use_container_width=True
        )




with tab6:

    st.title("📋 Clinical Prediction History")

    if os.path.exists(
        "clinical_prediction_history.csv"
    ):

        history_df = pd.read_csv(
            "clinical_prediction_history.csv"
        )

        st.metric(
            "Total Predictions",
            len(history_df)
        )

        st.markdown("---")

        patient_search = st.text_input(
            "Search Patient ID"
        )

        patient_name_search = st.text_input(
            "Search Patient Name"
        )
        if patient_search:

            history_df = history_df[
                history_df["Patient_ID"]
                .astype(str)
                .str.contains(
                    patient_search,
                    case=False,
                    na=False
                )
            ]
        if patient_name_search:

          history_df = history_df[
               history_df["Patient_Name"]
               .astype(str)
                .str.contains(
                    patient_name_search,
                    case=False,
                   na=False
        )
    ]

        clinician_filter = st.selectbox(
            "Filter by Clinician",
            ["All"] +
            list(
                history_df["Clinician"]
                .dropna()
                .unique()
            )
        )


        antibiotic_filter = st.selectbox(
                "Filter by Antibiotic",
                ["All"] +
                list(
                  history_df["Antibiotic"]
                    .dropna()
                    .unique()
             )
          )


        if clinician_filter != "All":

            history_df = history_df[
                history_df["Clinician"]
                == clinician_filter
            ]

        if antibiotic_filter != "All":

            history_df = history_df[
               history_df["Antibiotic"]
                == antibiotic_filter
                   ]
          
            
            c1, c2, c3 = st.columns(3)

            c1.metric(
                  "Total Records",
            len(history_df)
            )
            c2.metric(
                 "Susceptible",
            len(
        history_df[
            history_df["Prediction"]
            == "Susceptible"
               ]
               )
                
            )

            c3.metric(
                 "Resistant",
            len(
                  history_df[
              history_df["Prediction"]
                  == "Resistant"
                       ]
                        )
                      )


        st.dataframe(
            history_df,
            use_container_width=True
        )

        csv = history_df.to_csv(
            index=False
        )

        st.download_button(
            "📥 Export Results",
            csv,
            "prediction_history.csv",
            "text/csv"
        )

    else:

        st.warning(
            "No prediction history found."
        )
with tab7:

    st.title("📈 Antimicrobial Stewardship Dashboard")

    if os.path.exists(
        "clinical_prediction_history.csv"
    ):

        pred_df = pd.read_csv(
            "clinical_prediction_history.csv"
        )

        st.subheader(
            "Antibiotic Usage Frequency"
        )

        usage_counts = (
            pred_df["Antibiotic"]
            .value_counts()
        )

        st.bar_chart(
            usage_counts
        )

        st.markdown("---")

        aware_summary = []

        for drug in usage_counts.index:

            if drug in AWARE_DATA:

                aware_summary.append({
                    "Antibiotic": drug,
                    "WHO Group":
                        AWARE_DATA[drug]["group"],
                    "Times Used":
                        int(
                            usage_counts[drug]
                        )
                })

        if len(aware_summary) > 0:

            aware_df = pd.DataFrame(
                aware_summary
            )

            st.subheader(
                "WHO AWaRe Classification"
            )

            st.dataframe(
                aware_df,
                use_container_width=True
            )

            st.markdown("---")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "ACCESS",
                len(
                    aware_df[
                        aware_df["WHO Group"]
                        == "ACCESS"
                    ]
                )
            )

            c2.metric(
                "WATCH",
                len(
                    aware_df[
                        aware_df["WHO Group"]
                        == "WATCH"
                    ]
                )
            )

            c3.metric(
                "RESERVE",
                len(
                    aware_df[
                        aware_df["WHO Group"]
                        == "RESERVE"
                    ]
                )
            )

        else:

            st.warning(
                "No stewardship classifications available."
            )

    else:

        st.warning(
            "No prediction history available."
        )

        
