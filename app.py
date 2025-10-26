import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from fpdf import FPDF
from typing import Dict, Any

# CORRECT IMPORTS: Pulling functions from the local modules
# Ensure rule_generator and data_analyzer are in the same directory
from data_analyzer import perform_eda, get_feature_importance
from rule_generator import generate_rule_based_summary, answer_question

# --- Streamlit Configuration ---
st.set_page_config(layout="wide", page_title="API-Free Insight Generator")

# --- PDF Generation Utility (fpdf2) ---
# app.py (around line 18)
class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15) # CHANGED FONT
        self.cell(0, 10, 'Rule-Based Data Insight Report', 0, 1, 'C')
        self.ln(5)

    # app.py (Line ~25)
    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 12) # Changed from Arial
        self.cell(0, 8, title, 0, 1, 'L')
        # ... rest of function

    # app.py (around line 32)
    # app.py (around line 32)
    def chapter_body(self, body):
    # This loop filters out non-printable/unsafe characters
        safe_body = "".join(c for c in body if 32 <= ord(c) <= 126 or c == '\n')
    
    # We rely on the final output encoding, but use the safe text here
    # Use the clean_body/safe_body variable in multi_cell
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5, safe_body) # Use the sanitized text
        self.ln()

def create_pdf_report(summary_dict: Dict[str, Any], df: pd.DataFrame) -> bytes:
    """Generates a simple PDF from the rule-based summary dictionary."""
    pdf = PDF()
    pdf.add_page()

    pdf.chapter_title("1. Executive Summary")
    pdf.chapter_body(summary_dict.get('executive_summary', 'N/A'))

    pdf.chapter_title("2. Key Relationships & Findings")
    pdf.chapter_body("\n".join([f"- {rel}" for rel in summary_dict.get('key_relationships', ['N/A'])]))

    pdf.chapter_title("3. Data Quality Assessment")
    pdf.chapter_body(summary_dict.get('data_quality_assessment', 'N/A'))

    pdf.chapter_title("4. Strategic Recommendation")
    pdf.chapter_body(summary_dict.get('strategic_recommendation', 'N/A'))

    pdf.chapter_title("5. Data Snapshot")
    pdf.chapter_body(df.head().to_string(max_rows=5, max_cols=5))

    # app.py (Line 62)
    return pdf.output(dest='S')

# --- Streamlit UI and Orchestration ---

st.title("💡 API-Free CSV Insight Generator")
st.markdown("*(Rule-Based Engine running locally in Colab)*")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()

    # --- Session State Setup ---
    if 'df_hash' not in st.session_state or st.session_state.df_hash != hash(uploaded_file.getvalue()):
        st.session_state.df_hash = hash(uploaded_file.getvalue())
        st.session_state.eda_results = perform_eda(df)
        st.session_state.messages = []
        st.session_state.target_col = None

    # CORRECTED LINE (now line 74 in this clean script structure)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # --- MAIN LAYOUT ---
    tab_eda, tab_insights, tab_qa = st.tabs(["📊 Auto EDA & Charts", "🤖 Insight Summary", "💬 Ask Your Data"])

    # === TAB 1: Auto EDA & Charts ===
    with tab_eda:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

        st.subheader("Data Overview")
        st.markdown(f"**Shape:** {st.session_state.eda_results['shape']}")
        st.json(st.session_state.eda_results['null_report'])

        st.subheader("Auto-Generated Visualizations")

        # Correlation Heatmap
        if len(numeric_cols) >= 2:
            st.markdown("##### Feature Correlation Heatmap")
            fig_corr = px.imshow(df[numeric_cols].corr(), text_auto=True, aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)

        # Target Impact Analysis setup
        if numeric_cols:
            st.session_state.target_col = st.selectbox("Select Target Column for Feature Importance", numeric_cols)
            if st.session_state.target_col:
                importance_list = get_feature_importance(df, st.session_state.target_col)
                if importance_list:
                    st.markdown(f"##### Top Features Impacting `{st.session_state.target_col}` (Random Forest)")
                    importance_df = pd.DataFrame(importance_list)
                    fig_imp = px.bar(importance_df, x='feature', y='importance', title=f"Feature Importance for {st.session_state.target_col}")
                    st.plotly_chart(fig_imp)
                else:
                    st.warning("Feature Importance requires at least one numerical feature besides the target.")

    # === TAB 2: Rule-Based Insight Summary ===
    with tab_insights:
        current_importance = []
        if st.session_state.target_col:
             current_importance = get_feature_importance(df, st.session_state.target_col)

        summary_dict = generate_rule_based_summary(st.session_state.eda_results, current_importance)

        st.markdown("### 📄 Automated Insight Report")

        st.info(f"**Executive Summary:** {summary_dict.get('executive_summary')}")
        st.success(f"**Strategic Recommendation:** {summary_dict.get('strategic_recommendation')}")

        with st.expander("Detailed Report Sections"):
            st.markdown("**Key Relationships:**")
            for rel in summary_dict.get('key_relationships', []):
                st.write(f"- {rel}")
            st.markdown(f"**Data Quality Assessment:** {summary_dict.get('data_quality_assessment')}")

        # PDF Download Button
        pdf_bytes = create_pdf_report(summary_dict, df)
        st.download_button(
            label="⬇️ Download Full PDF Insight Report",
            data=pdf_bytes,
            file_name="Insight_Report_Rule_Based.pdf",
            mime="application/pdf"
        )


    # === TAB 3: Ask Your Data ===
    with tab_qa:
        st.subheader("Ask Rule-Based Questions")
        st.info("The Q&A uses simple rules. Ask about **Feature Impact**, **Correlations**, or **Data Quality**.")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("E.g., Which feature affects the target most?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response_text = answer_question(
                    prompt,
                    st.session_state.eda_results,
                    get_feature_importance(df, st.session_state.target_col) if st.session_state.target_col else []
                )

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

else:
    st.info("👆 Please upload a CSV file in the sidebar to begin the analysis.")
