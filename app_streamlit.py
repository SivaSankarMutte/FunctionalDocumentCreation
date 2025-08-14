import streamlit as st
from generate_doc import generate_functional_doc

st.set_page_config(page_title="Functional Doc Generator", page_icon="📄", layout="wide")

st.title("📄 Functional Documentation Generator")
st.write("Upload a ZIP of your codebase to generate a functional documentation report.")

uploaded_file = st.file_uploader("Upload ZIP file", type=["zip"])

if uploaded_file is not None:
    with st.spinner("Generating documentation... This may take a few minutes."):
        try:
            md_path, docx_path = generate_functional_doc(uploaded_file.read(), workdir="./work")

            # Download buttons
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            st.download_button(
                "⬇️ Download as Markdown",
                md_content,
                file_name="functional_doc.md",
                mime="text/markdown"
            )

            with open(docx_path, "rb") as f:
                st.download_button(
                    "⬇️ Download as DOCX",
                    f,
                    file_name="functional_doc.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

            # Preview in Streamlit
            st.subheader("📖 Documentation Preview")
            st.markdown(md_content)

        except Exception as e:
            st.error(f"Error: {e}")
