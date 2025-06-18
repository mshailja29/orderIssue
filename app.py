import streamlit as st
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Optional: Debug if key loads
# st.write("API key ends with:", os.getenv("OPENAI_API_KEY")[-5:])

# Streamlit config
st.set_page_config(page_title="Order Issue Command Center", layout="wide")
st.title("📦 Order Issue Command Center")
st.write("Upload order issues and let GPT cluster and summarize them!")

# File upload
uploaded_file = st.file_uploader("Upload CSV with Order ID, Customer Message, and Tracking Log", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

    if st.button("Analyze with GPT"):
        with st.spinner("Analyzing..."):
            try:
                # Build prompt from uploaded data
                issues_text = ""
                for _, row in df.iterrows():
                    issues_text += f"(Order {row['Order ID']}: {row['Customer Message']} Tracking log: {row['Tracking Log']})\n"

                prompt = f"""
You are an AI assistant helping a logistics company triage and summarize customer complaints about order issues.

Group the following customer support issues into common categories like:
- In Transit Delay
- Marked Delivered but Not Received
- Tracking Stuck
- Wrong Item Delivered
- Others

Then, for each group:
- Summarize the key issue
- Suggest one action the support team should take

Here are the order issues:
{issues_text}
                """

                # GPT call using latest SDK
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )

                result = response.choices[0].message.content
                st.markdown("### 🧠 GPT Response")
                st.info(result)

            except Exception as e:
                st.error(f"Error: {e}")
