# app.py
import streamlit as st
import pandas as pd
from pycaret.classification import load_model as load_clf_model, predict_model as predict_clf_model
from pycaret.clustering import load_model as load_clu_model, predict_model as predict_clu_model
from genai_prescriptions import generate_prescription
import os
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="GenAI-Powered Phishing SOAR",
    page_icon="🛡️",
    layout="wide"
)

# --- Load Model and Feature Plot ---
@st.cache_resource
def load_assets():
    classification_model_path = 'models/phishing_url_detector'
    clustering_model_path = 'models/threat_actor_profiler'
    plot_path = 'models/feature_importance.png'

    classification_model = None
    clustering_model = None
    plot = None

    if os.path.exists(classification_model_path + '.pkl'):
        classification_model = load_clf_model(classification_model_path)
    if os.path.exists(clustering_model_path + '.pkl'):
        clustering_model = load_clu_model(clustering_model_path)
    if os.path.exists(plot_path):
        plot = plot_path

    return classification_model, clustering_model, plot

classification_model, clustering_model, feature_plot = load_assets()

if not classification_model or not clustering_model:
    st.error(
        "One or more models are missing. Please ensure both `phishing_url_detector.pkl` and `threat_actor_profiler.pkl` exist in the 'models' directory. Run the training script if needed."
    )
    st.stop()

# --- Threat Actor Profile Definitions ---
# These descriptions map cluster IDs to meaningful profiles
THREAT_PROFILES = {
    'Cluster 0': {
        "name": "Organized Cybercrime",
        "icon": "💸",
        "description": "This threat profile is characterized by high-volume, financially motivated attacks. Their methods are often noisy, relying on techniques like URL shortening, IP addresses in URLs, and abnormal structures to overwhelm standard defenses. The primary goal is widespread credential theft or financial fraud."
    },
    'Cluster 1': {
        "name": "State-Sponsored Actor",
        "icon": "🌐",
        "description": "This profile represents a highly sophisticated and targeted attacker. They use subtle, well-crafted techniques, such as valid SSL certificates combined with deceptive sub-domains (e.g., `login.microsoft.com-validate.net`). Their goal is typically espionage, intelligence gathering, or strategic disruption."
    },
    'Cluster 2': {
        "name": "Hacktivist",
        "icon": "📢",
        "description": "This profile is driven by political or social motives. Their attacks are often opportunistic and may leverage current events. The technical sophistication can vary, but a key indicator is often the use of politically charged keywords or themes in the URL itself to lure victims."
    },
}

# --- Sidebar for Inputs ---
with st.sidebar:
    st.title("🔬 URL Feature Input")
    st.write("Describe the characteristics of a suspicious URL below.")

    # Using a dictionary to hold form values
    form_values = {
        'url_length': st.select_slider("URL Length", options=['Short', 'Normal', 'Long'], value='Long'),
        'ssl_state': st.select_slider("SSL Certificate Status", options=['Trusted', 'Suspicious', 'None'], value='Suspicious'),
        'sub_domain': st.select_slider("Sub-domain Complexity", options=['None', 'One', 'Many'], value='One'),
        'prefix_suffix': st.checkbox("URL has a Prefix/Suffix (e.g.,'-')", value=True),
        'has_ip': st.checkbox("URL uses an IP Address", value=False),
        'short_service': st.checkbox("Is it a shortened URL", value=False),
        'at_symbol': st.checkbox("URL contains '@' symbol", value=False),
        'abnormal_url': st.checkbox("Is it an abnormal URL", value=True),
        'political_keyword': st.checkbox("URL contains a political keyword", value=False)
    }

    st.divider()
    genai_provider = st.selectbox("Select GenAI Provider", ["Gemini", "OpenAI", "Grok"])
    submitted = st.button("💥 Analyze & Initiate Response", use_container_width=True, type="primary")

# --- Main Page ---
st.title("🛡️ GenAI-Powered SOAR for Phishing URL Analysis")

if not submitted:
    st.info("Please provide the URL features in the sidebar and click 'Analyze' to begin.")
    if feature_plot:
        st.subheader("Model Feature Importance")
        st.image(feature_plot,
                 caption="Feature importance from the trained RandomForest model. This shows which features the model weighs most heavily when making a prediction.")

else:
    # --- Data Preparation and Risk Scoring ---
    input_dict = {
        'having_IP_Address': 1 if form_values['has_ip'] else -1,
        'URL_Length': -1 if form_values['url_length'] == 'Short' else (
            0 if form_values['url_length'] == 'Normal' else 1),
        'Shortining_Service': 1 if form_values['short_service'] else -1,
        'having_At_Symbol': 1 if form_values['at_symbol'] else -1,
        'double_slash_redirecting': -1,
        'Prefix_Suffix': 1 if form_values['prefix_suffix'] else -1,
        'having_Sub_Domain': -1 if form_values['sub_domain'] == 'None' else (
            0 if form_values['sub_domain'] == 'One' else 1),
        'SSLfinal_State': -1 if form_values['ssl_state'] == 'None' else (
            0 if form_values['ssl_state'] == 'Suspicious' else 1),
        'Abnormal_URL': 1 if form_values['abnormal_url'] else -1,
        'URL_of_Anchor': 0, 'Links_in_tags': 0, 'SFH': 0,
        'has_political_keyword': 1 if form_values['political_keyword'] else -1,
    }
    input_data = pd.DataFrame([input_dict])

    # Simple risk contribution for visualization
    risk_scores = {
        "Bad SSL": 25 if input_dict['SSLfinal_State'] < 1 else 0,
        "Abnormal URL": 20 if input_dict['Abnormal_URL'] == 1 else 0,
        "Prefix/Suffix": 15 if input_dict['Prefix_Suffix'] == 1 else 0,
        "Shortened URL": 15 if input_dict['Shortining_Service'] == 1 else 0,
        "Complex Sub-domain": 10 if input_dict['having_Sub_Domain'] == 1 else 0,
        "Long URL": 10 if input_dict['URL_Length'] == 1 else 0,
        "Uses IP Address": 5 if input_dict['having_IP_Address'] == 1 else 0,
    }
    risk_df = pd.DataFrame(list(risk_scores.items()), columns=['Feature', 'Risk Contribution']).sort_values(
        'Risk Contribution', ascending=False)

    # --- Analysis Workflow ---
    with st.status("Executing SOAR playbook...", expanded=True) as status:
        st.write("▶️ **Step 1: Predictive Analysis** - Running features through classification model.")
        time.sleep(1)
        prediction = predict_clf_model(classification_model, data=input_data)
        is_malicious = prediction['prediction_label'].iloc[0] == 1
        verdict = "MALICIOUS" if is_malicious else "BENIGN"
        st.write(f"▶️ **Step 2: Verdict Interpretation** - Model predicts **{verdict}**.")
        time.sleep(1)

        threat_profile_name = "N/A"
        if is_malicious:
            st.write("▶️ **Step 3: Threat Attribution** - Profiling threat actor with clustering model.")
            # Use only the features the clustering model was trained on
            clustering_features = list(clustering_model.feature_names_in_)
            cluster_prediction = predict_clu_model(clustering_model, data=input_data[clustering_features])
            predicted_cluster_id = cluster_prediction['Cluster'].iloc[0]
            threat_profile_info = THREAT_PROFILES.get(predicted_cluster_id,
                                                      {"name": "Unknown", "description": "No profile matched."})
            threat_profile_name = threat_profile_info["name"]
            st.write(f"▶️ **Step 4: Attribution Complete** - Profiled as **{threat_profile_name}**.")
            time.sleep(1)

            st.write(f"▶️ **Step 5: Prescriptive Analytics** - Engaging **{genai_provider}** for action plan.")
            try:
                prescription = generate_prescription(genai_provider, {k: v for k, v in input_dict.items()})
                status.update(label="✅ SOAR Playbook Executed Successfully!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Failed to generate prescription: {e}")
                prescription = None
                status.update(label="🚨 Error during GenAI prescription!", state="error")
        else:
            prescription = None
            status.update(label="✅ Analysis Complete. No threat found.", state="complete", expanded=False)

    # --- Tabs for Organized Output ---
    tab_list = ["📊 **Analysis Summary**", "🎯 **Threat Attribution**", "📈 **Visual Insights**",
                "📜 **Prescriptive Plan**"]
    summary_tab, attribution_tab, visuals_tab, plan_tab = st.tabs(tab_list)

    with summary_tab:
        st.subheader("Verdict and Key Findings")
        if is_malicious:
            st.error("**Prediction: Malicious Phishing URL**", icon="🚨")
        else:
            st.success("**Prediction: Benign URL**", icon="✅")

        st.metric("Malicious Confidence Score",
                  f"{prediction['prediction_score'].iloc[0]:.2%}" if is_malicious else f"{1 - prediction['prediction_score'].iloc[0]:.2%}")
        st.caption("This score represents the model's confidence in its prediction.")

    with attribution_tab:
        st.subheader("Threat Actor Profiling")
        if is_malicious:
            profile_info = THREAT_PROFILES.get(predicted_cluster_id)
            st.info(f"**Attributed Profile: {profile_info['icon']} {profile_info['name']}**")
            st.write(profile_info['description'])
        else:
            st.info("Threat attribution is only performed on URLs classified as malicious.")

    with visuals_tab:
        st.subheader("Visual Analysis")
        risk_scores = {
            "Bad SSL": 25 if input_dict['SSLfinal_State'] < 1 else 0,
            "Abnormal URL": 20 if input_dict['Abnormal_URL'] == 1 else 0,
            "Prefix/Suffix": 15 if input_dict['Prefix_Suffix'] == 1 else 0,
            "Shortened URL": 15 if input_dict['Shortining_Service'] == 1 else 0,
            "Complex Sub-domain": 10 if input_dict['having_Sub_Domain'] == 1 else 0,
            "Long URL": 10 if input_dict['URL_Length'] == 1 else 0,
            "Uses IP Address": 5 if input_dict['having_IP_Address'] == 1 else 0,
            "Political Keyword": 20 if input_dict['has_political_keyword'] == 1 else 0
        }
        risk_df = pd.DataFrame(list(risk_scores.items()), columns=['Feature', 'Risk Contribution']).sort_values(
            'Risk Contribution', ascending=False)
        st.write("#### Risk Contribution by Feature")
        st.bar_chart(risk_df.set_index('Feature'))
        st.caption("A simplified view of which input features contributed most to a higher risk score.")

    with plan_tab:
        st.subheader("Actionable Response Plan")
        if prescription:
            st.success("A prescriptive response plan has been generated by the AI.", icon="🤖")
            st.json(prescription, expanded=False)

            st.write("#### Recommended Actions (for Security Analyst)")
            for i, action in enumerate(prescription.get("recommended_actions", []), 1):
                st.markdown(f"**{i}.** {action}")

            st.write("#### Communication Draft (for End-User/Reporter)")
            st.text_area("Draft", prescription.get("communication_draft", ""), height=150)
        else:
            st.info("No prescriptive plan was generated because the URL was classified as benign.")

