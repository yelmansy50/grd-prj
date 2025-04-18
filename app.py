from gtts import gTTS
import os
import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Set page config
st.set_page_config(page_title="NetKeeper", layout="centered")

# Center the image using Streamlit's `st.image` with a centered layout
col1, col2, col3 = st.columns([1, 2, 1])  # Create three columns
with col2:  # Place the image in the center column
    st.image("D:/net.jpg", width=800)

# Set the title of the app
st.title("NetKeeper")
st.subheader("Predict the Protocol Type Based on Network Traffic Data")

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
    ], "D:/icmp.jpg"),

    3.0: ("🔴 DHCP Attack", [
        "Enable DHCP snooping on switches to filter rogue servers.",
        "Use port security to limit MAC addresses per port.",
        "Configure trusted ports only for legitimate DHCP servers.",
        "Monitor logs for multiple DHCP OFFER messages."
    ], None),

    4.0: ("🟢 Safe", [], "D:/safe.jpg"),

    5.0: ("🔴 MAC Flood Attack", [
        "Enable port security to restrict dynamic MAC addresses.",
        "Limit the number of MAC addresses per interface.",
        "Use dynamic ARP inspection to validate MAC-IP mappings.",
        "Deploy IDS/IPS to detect abnormal MAC behavior."
    ], "D:/macflood.jpg")
}

# Create the form using Streamlit widgets
with st.form("prediction_form"):
    st.write("Please fill out the details below:")

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
        # st.write("Input DataFrame:", pred_df)

        predict_pipeline = PredictPipeline()
        st.write("Starting Prediction...")
        try:
            results = predict_pipeline.predict(pred_df)
            pred_class = float(results[0])

            if pred_class in attack_info:
                label, tips, image_path = attack_info[pred_class]
                st.success(f"Prediction: {label}")
                
                # Display image if available (before tips)
                if image_path:
                    st.image(image_path, caption=label, use_container_width=True)
                
                # Generate and play audio for the prediction
                tts = gTTS(text=f"The prediction is {label}", lang='en')
                audio_file = "prediction_audio.mp3"
                tts.save(audio_file)
                st.audio(audio_file, format="audio/mp3")

                # Display tips
                for i, tip in enumerate(tips, start=1):
                    st.info(f"💡 Tip {i}: {tip}")
            else:
                st.warning("⚠️ Unknown prediction result.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")