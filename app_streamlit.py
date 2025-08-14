import os
import io
import streamlit as st
from dotenv import load_dotenv
from generate_doc import generate_functional_doc

st.set_page_config(page_title="Source → Functional Doc (GenAI)", layout="wide")
st.title("📘 Source → Functional Doc (GenAI)")

with st.expander("⚙️ Configuration", expanded=False):
    st.write("Set API keys in environment or Streamlit secrets.")
    st.code("GROQ_API_KEY or OPENAI_API_KEY", language="bash")

uploaded = st.file_uploader("Upload a ZIP of your project repository", type=["zip"])

if st.button("Generate Documentation", disabled=not uploaded):
    with st.spinner("Indexing code and generating documentation..."):
        zip_bytes = uploaded.read()
        md_path, docx_path = generate_functional_doc(zip_bytes, workdir="./work")
    st.success("Done!")
    st.download_button("⬇️ Download Markdown", data=open(md_path,"rb").read(), file_name="functional_doc.md")
    st.download_button("⬇️ Download Word (.docx)", data=open(docx_path,"rb").read(), file_name="functional_doc.docx")

st.markdown("---")
st.caption("Tip: Add a README to your repo; it improves the overview section.")

