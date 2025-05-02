import os
from gtts import gTTS
import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Set page config
st.set_page_config(page_title="NetKeeper", layout="centered")

# Add custom CSS for styling
st.markdown(
    """
    <style>
    body, .stApp {
        background: linear-gradient(to bottom, #001a33, #00264d);
        color: #ffffff;
        font-family: 'Poppins', sans-serif;
        background-size: cover;
    }
    .title {
        font-size: 3.5rem;
        font-weight: bold;
        color: #00ccff;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 3px 3px 6px #000d1a;
    }
    .subtitle {
        font-size: 1.8rem;
        color: #66d9ff;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .form-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 1rem;
        text-align: center;
    }
    label {
        color: #ffffff !important;
        font-size: 1rem;
        font-weight: bold;
    }
    input, select, textarea {
        background-color: #00264d;
        color: #ffffff;
        border: 1px solid #00ccff;
        border-radius: 8px;
        padding: 10px;
        font-size: 1rem;
    }
    input:hover, select:hover, textarea:hover {
        background-color: #003366;
        color: #ffffff;
    }
    select option {
        background-color: #001a33;
        color: #ffffff;
        padding: 10px;
    }
    select option:nth-child(odd) {
        background-color: #003366;
    }
    select option:nth-child(even) {
        background-color: #004080;
    }
    select option:hover {
        background-color: #00ccff;
        color: #000000;
    }
    .success {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
    }
    .error {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
    }
    .tip {
        background-color: #001a33;
        border-left: 5px solid #00ccff;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 8px;
        font-size: 1rem;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #00ccff;
        color: #ffffff;
        font-size: 1rem;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        box-shadow: 2px 2px 5px #000d1a;
    }
    .stButton>button:hover {
        background-color: #003366;
        color: #e6ffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Center the logo using Streamlit's `st.image` with a centered layout
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("notebook/logo prj.jpg", width=400)

# Set the title of the app
st.markdown('<div class="title">NetKeeper</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict the Protocol Type Based on Network Traffic Data</div>', unsafe_allow_html=True)

# Mapping predictions to labels, tips, and images
attack_info = {
    0.0: ("🔴 CDP Attack", [
        "Disable unused CDP services on all network devices.",
        "Segment your network and limit CDP to trusted interfaces only.",
        "Regularly audit Layer 2 configurations.",
        "Use secure management protocols like SSH instead of CDP-based discovery."
    ], None),

    2.0: ("🔴 OSPF Attack", [
        "Use MD5 or SHA authentication for OSPF messages.",
        "Implement passive interfaces where routing updates are not needed.",
        "Monitor routing tables for sudden or suspicious changes.",
        "Use route filtering and summarization to control routing updates."
    ], None),

    1.0: ("🔴 ICMP Attack", [
        "Restrict ICMP traffic using firewall rules.",
        "Limit ICMP rate using router access control lists (ACLs).",
        "Monitor for ICMP floods or ping sweeps using IDS.",
        "Disable ICMP redirect messages on gateways."
    ], None),

    3.0: ("🔴 DHCP Attack", [
        "Enable DHCP snooping on switches to filter rogue servers.",
        "Use port security to limit MAC addresses per port.",
        "Configure trusted ports only for legitimate DHCP servers.",
        "Monitor logs for multiple DHCP OFFER messages."
    ], None),

    4.0: ("🟢 Safe", [], None),

    5.0: ("🔴 MAC Flood Attack", [
        "Enable port security to restrict dynamic MAC addresses.",
        "Limit the number of MAC addresses per interface.",
        "Use dynamic ARP inspection to validate MAC-IP mappings.",
        "Deploy IDS/IPS to detect abnormal MAC behavior."
    ], None)
}

# Create the form using Streamlit widgets
st.markdown('<div class="form-title">Please fill out the details below:</div>', unsafe_allow_html=True)
with st.form("prediction_form"):
    no = st.number_input("Packet Number (No.)", min_value=1, step=1)
    time = st.number_input("Time (in seconds)", min_value=0.0, step=0.01)
    protocol = st.selectbox(
        "Protocol",
        ["Select Protocol", "CDP", "ICMP", "STP", "OSPF", "DHCP", "IPv4", "TCP"]
    )
    length = st.number_input("Packet Length (in bytes)", min_value=1, step=1)
    source_type = st.selectbox(
        "Source Type", ["Select Source Type", "MAC_Address", "IP_Address"]
    )
    destination_type = st.selectbox(
        "Destination Type",
        [
            "Select Destination Type",
            "Network_Protocol",
            "Other_Destination",
            "Multicast",
            "Broadcast",
            "Spanning_Tree_Protocol",
        ]
    )

    submitted = st.form_submit_button("Predict Protocol Type")

# Handle form submission
if submitted:
    if (
        protocol == "Select Protocol"
        or source_type == "Select Source Type"
        or destination_type == "Select Destination Type"
    ):
        st.error("Please fill out all the fields correctly.")
    else:
        data = CustomData(
            no=no,
            time=time,
            protocol=protocol,
            length=length,
            source_type=source_type,
            destination_type=destination_type,
        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        st.write("Starting Prediction...")
        try:
            results = predict_pipeline.predict(pred_df)
            pred_class = float(results[0])

            if pred_class in attack_info:
                label, tips, _ = attack_info[pred_class]
                st.markdown(f'<div class="success">Prediction: {label}</div>', unsafe_allow_html=True)
                
                # Generate and play audio for the prediction
                tts = gTTS(text=f"The prediction is {label}", lang='en')
                audio_file = "prediction_audio.mp3"
                tts.save(audio_file)
                st.audio(audio_file, format="audio/mp3")

                # Display tips
                for i, tip in enumerate(tips, start=1):
                    st.markdown(f'<div class="tip">💡 Tip {i}: {tip}</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Unknown prediction result.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")