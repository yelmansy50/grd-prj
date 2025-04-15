import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Set the title of the app
st.title("NetKeeper")
st.subheader("Predict the Protocol Type Based on Network Traffic Data")

# Create the form using Streamlit widgets
with st.form("prediction_form"):
    st.write("Please fill out the details below:")

    # No.
    no = st.number_input("Packet Number (No.)", min_value=1, step=1)

    # Time
    time = st.number_input("Time (in seconds)", min_value=0.0, step=0.01)

    # Protocol
    protocol = st.selectbox(
        "Protocol",
        ["Select Protocol", "CDP", "ICMP", "STP", "OSPF", "DHCP", "IPv4", "TCP"],
    )

    # Length
    length = st.number_input("Packet Length (in bytes)", min_value=1, step=1)

    # Source Type
    source_type = st.selectbox(
        "Source Type", ["Select Source Type", "MAC_Address", "IP_Address"]
    )

    # Destination Type
    destination_type = st.selectbox(
        "Destination Type",
        [
            "Select Destination Type",
            "Network_Protocol",
            "Other_Destination",
            "Multicast",
            "Broadcast",
            "Spanning_Tree_Protocol",
        ],
    )

    # Submit button
    submitted = st.form_submit_button("Predict Protocol Type")

# Handle form submission
if submitted:
    # Validate inputs
    if (
        protocol == "Select Protocol"
        or source_type == "Select Source Type"
        or destination_type == "Select Destination Type"
    ):
        st.error("Please fill out all the fields correctly.")
    else:
        # Normalize input categories to match the preprocessor's expectations
        data = CustomData(
            no=no,
            time=time,
            protocol=protocol,
            length=length,
            source_type=source_type,
            destination_type=destination_type,
        )

        # Convert data to DataFrame
        pred_df = data.get_data_as_data_frame()
        st.write("Input DataFrame:", pred_df)

        # Prediction pipeline
        predict_pipeline = PredictPipeline()
        st.write("Starting Prediction...")
        try:
            # Perform prediction
            results = predict_pipeline.predict(pred_df)
            st.success(f"The predicted Protocol Type is: {results[0]}")
        except Exception as e:
            # Handle errors during prediction
            st.error(f"An error occurred during prediction: {e}")