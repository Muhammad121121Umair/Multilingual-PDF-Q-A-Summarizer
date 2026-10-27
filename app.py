import os
import re
import io
import traceback
import streamlit as st
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from groq import Groq

MODEL_ID = "openai/gpt-oss-120b"

st.set_page_config(
    page_title="AI Office Suite (Excel & PowerPoint)",
    page_icon="📑",
    layout="wide"
)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def extract_python_code(text: str) -> str:
    """Extract code block if wrapped in markdown fences, or return raw string."""
    pattern = r"```(?:python)?\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Workspace Setup")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Reads GROQ_API_KEY from Streamlit secrets or manual entry."
    )
    
    st.markdown("---")
    mode = st.radio(
        "Target Deliverable", 
        ["PowerPoint Presentation (.pptx)", "Excel Workbook (.xlsx)"]
    )
    
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Optional Reference Data (CSV/Excel)", 
        type=["csv", "xlsx", "xls"],
        help="Upload data if you want slides or sheets generated from existing numbers."
    )

# ---------------------------------------------------------
# Main Interface
# ---------------------------------------------------------
st.header("📑 AI Office Assistant: Excel & PowerPoint Builder")
st.caption(f"Powered by `{MODEL_ID}` on Groq. Generates native, downloadable files.")

df = None
schema_info = "No dataset uploaded. Build entirely from user prompt instructions."

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.write("**Dataset Preview:**")
        st.dataframe(df.head(3), use_container_width=True)
        
        buf = io.StringIO()
        df.info(buf=buf)
        schema_info = f"Columns & Types:\n{buf.getvalue()}\nSample Rows:\n{df.head(2).to_dict()}"
    except Exception as e:
        st.error(f"Error reading file: {e}")

prompt_placeholder = (
    "e.g., Create a 4-slide executive briefing on procurement compliance with key criteria and timelines."
    if "PowerPoint" in mode
    else "e.g., Create a clean budget sheet with revenue, operational expenses, totals, and formatted profit margins."
)

user_query = st.text_area("What do you want built?", placeholder=prompt_placeholder, height=100)

if st.button("Generate File", type="primary", use_container_width=True):
    if not api_key:
        st.error("Please provide your Groq API key in the sidebar.")
        st.stop()
    if not user_query.strip():
        st.warning("Please provide instructions for what to create.")
        st.stop()

    groq_client = Groq(api_key=api_key)

    if "PowerPoint" in mode:
        system_prompt = (
            "You are an expert Python developer specialized in `python-pptx`.\n"
            "Generate pure executable Python code that builds a PowerPoint presentation.\n\n"
            "MANDATORY EXECUTION CONVENTIONS:\n"
            "1. Available imports: `from pptx import Presentation`, `from pptx.util import Inches, Pt`, `from pptx.dml.color import RGBColor`.\n"
            "2. Always start by creating the presentation object: `prs = Presentation()`.\n"
            "3. TYPE SAFETY: Wrap all text assignments in `str(...)`.\n"
            "4. Build multi-slide decks with clean layout, slide titles, subtitles, formatted bullets, or tables.\n"
            "5. SAVING: You MUST end the script with this exact line:\n"
            "   prs.save(output_buffer)\n"
            "6. Output ONLY executable Python inside a ```python block. No introductory comments or explanations."
        )
    else:
        system_prompt = (
            "You are an expert Python developer specialized in `pandas` and `openpyxl`.\n"
            "Generate pure executable Python code that builds or formats an Excel workbook.\n\n"
            "MANDATORY EXECUTION CONVENTIONS:\n"
            "1. Available imports: `import pandas as pd`, `import openpyxl`, `from openpyxl.styles import Font, PatternFill, Alignment, Border, Side`.\n"
            "2. If `df` is provided, manipulate or enrich it. If creating from scratch, build a dataframe `result_df = pd.DataFrame(...)` or an openpyxl workbook `wb = openpyxl.Workbook()`.\n"
            "3. SAVING: You MUST write the final data directly into `output_buffer` using ONE of these two exact patterns:\n"
            "   Pattern A (Pandas):\n"
            "       with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:\n"
            "           result_df.to_excel(writer, index=False, sheet_name='Sheet1')\n"
            "   Pattern B (Openpyxl Workbook):\n"
            "       wb.save(output_buffer)\n"
            "4. NEVER call writer.save() under any circumstances.\n"
            "5. Output ONLY executable Python inside a ```python block. No introductory comments or explanations."
        )

    task_payload = (
        f"AVAILABLE REFERENCE DATA:\n{schema_info}\n\n"
        f"USER REQUEST: {user_query}\n\n"
        "Generate the complete Python script to create and save the output into `output_buffer`."
    )

    with st.spinner("Writing and compiling your file with Groq..."):
        try:
            res = groq_client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task_payload}
                ],
                temperature=0.1,
                max_tokens=2500,
            )
            code = extract_python_code(res.choices[0].message.content)

            # Compatibility patch for legacy writer.save()
            if not hasattr(pd.ExcelWriter, "save"):
                pd.ExcelWriter.save = lambda self: self.close()

            output_buffer = io.BytesIO()
            exec_scope = {
                "output_buffer": output_buffer,
                "df": df,
                "Presentation": Presentation,
                "Inches": Inches,
                "Pt": Pt,
                "RGBColor": RGBColor,
                "pd": pd,
                "openpyxl": openpyxl,
                "Font": Font,
                "PatternFill": PatternFill,
                "Alignment": Alignment,
                "Border": Border,
                "Side": Side
            }

            # Execute generated code
            exec(code, {}, exec_scope)

            # -------------------------------------------------------------
            # Fallback Auto-Recovery: Intercept if output_buffer is empty
            # -------------------------------------------------------------
            file_bytes = output_buffer.getvalue()

            if len(file_bytes) == 0:
                # 1. Did the code define an openpyxl workbook `wb` or `workbook`?
                for var_name in ["wb", "workbook"]:
                    if var_name in exec_scope and hasattr(exec_scope[var_name], "save"):
                        exec_scope[var_name].save(output_buffer)
                        file_bytes = output_buffer.getvalue()
                        break

                # 2. Did the code define a PowerPoint presentation `prs` or `presentation`?
                if len(file_bytes) == 0:
                    for var_name in ["prs", "presentation"]:
                        if var_name in exec_scope and hasattr(exec_scope[var_name], "save"):
                            exec_scope[var_name].save(output_buffer)
                            file_bytes = output_buffer.getvalue()
                            break

                # 3. Did the code create a new DataFrame `result_df` or update `df`?
                if len(file_bytes) == 0 and "Excel" in mode:
                    target_df = exec_scope.get("result_df", exec_scope.get("df", None))
                    if isinstance(target_df, pd.DataFrame):
                        with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
                            target_df.to_excel(writer, index=False)
                        file_bytes = output_buffer.getvalue()

            # Final validation
            if len(file_bytes) == 0:
                st.error("Output buffer was empty. Here is the generated code for inspection:")
                st.code(code, language="python")
            else:
                st.success(f"File created successfully! ({len(file_bytes):,} bytes)")

                if "PowerPoint" in mode:
                    st.download_button(
                        label="📥 Download Presentation (.pptx)",
                        data=file_bytes,
                        file_name="generated_presentation.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        use_container_width=True
                    )
                else:
                    st.download_button(
                        label="📥 Download Workbook (.xlsx)",
                        data=file_bytes,
                        file_name="generated_sheet.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

                with st.expander("View Generated Automation Code"):
                    st.code(code, language="python")

        except Exception as e:
            st.error(f"Execution Error: {e}")
            with st.expander("Show Traceback Details"):
                st.code(traceback.format_exc(), language="python")
