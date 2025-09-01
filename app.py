# app.py
# Streamlit app: Tower Spec → Auto Material List (Ringlock-focused)
# Author: ChatGPT (for Renier)
# Notes:
# - Upload a tower drawing/spec PDF. The app will try to parse key parameters (count, plan 4x4m, bay spacing 2m, height 12m, cladding, ballast markers like "1493kg").
# - You can override anything in the confirmation form.
# - The BOM rules are parametric and editable from the sidebar.
# - Exports XLSX and PDF in a "Project Material List" style.
#
# Tip: For OCR of drawings, ensure you have Tesseract installed on the host machine.
#   macOS (brew): brew install tesseract
#   Ubuntu: sudo apt-get install tesseract-ocr
#   Windows: https://github.com/UB-Mannheim/tesseract/wiki

import io
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Optional text extraction backends
try:
    import pdfplumber  # text extraction from vector PDFs
except Exception:  # pragma: no cover
    pdfplumber = None

try:
    from pdf2image import convert_from_bytes  # OCR fallback
    import pytesseract
    from PIL import Image
except Exception:  # pragma: no cover
    convert_from_bytes = None
    pytesseract = None
    Image = None

# ------------------------------------------------------------
# Data classes
# ------------------------------------------------------------
@dataclass
class TowerParams:
    system: str = "Ringlock"
    tower_name: str = "Tower LX"
    tower_count: int = 1
    width_m: float = 4.0
    depth_m: float = 4.0
    bay_m: float = 2.0
    height_m: float = 12.0
    lift_m: float = 2.0
    platforms_at_lifts: List[int] = None  # e.g. [6] for top only
    cladding: str = "Netting"
    barrier_1493_per_tower: int = 0  # Concrete Jersey Barrier 1493kg
    notes: str = ""

    def __post_init__(self):
        if self.platforms_at_lifts is None:
            self.platforms_at_lifts = [self.lifts]  # default platform at top

    @property
    def lifts(self) -> int:
        return int(math.ceil(self.height_m / self.lift_m))


@dataclass
class BracePolicy:
    # density = 1 means brace present at *every* lift set we check; 0.5 means alternate lifts, etc.
    plan_brace_alt_lifts: bool = True
    diag_brace_alt_lifts: bool = True


# ------------------------------------------------------------
# Parsing utilities
# ------------------------------------------------------------
TOWER_RE = re.compile(r"Tower\s+([A-Za-z0-9\- ]+)\s*\((\d+)\s*nos?\.?\)", re.IGNORECASE)
CLADDING_RE = re.compile(r"Cladding\s*type\s*([A-Za-z]+)", re.IGNORECASE)


def parse_pdf_text(file_bytes: bytes) -> str:
    """Try pdfplumber text first; then OCR if available."""
    text = ""
    if pdfplumber is not None:
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except Exception:
            pass
    if text.strip():
        return text

    # OCR fallback
    if convert_from_bytes is not None and pytesseract is not None:
        try:
            images = convert_from_bytes(file_bytes, dpi=200)
            for im in images:
                text += pytesseract.image_to_string(im)
        except Exception:
            pass

    return text


def smart_number_candidates(text: str) -> List[int]:
    """Return a list of integer candidates found in the text, sorted by frequency (desc)."""
    nums = re.findall(r"\b\d{3,5}\b", text)  # 3-5 digit numbers like 2000, 4000, 12000
    vals = list(map(int, nums))
    # frequency sort
    freq: Dict[int, int] = {}
    for v in vals:
        freq[v] = freq.get(v, 0) + 1
    return [k for k, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))]


def extract_params_from_text(text: str) -> Tuple[Optional[str], Optional[int], Optional[float], Optional[float], Optional[float], Optional[float], Optional[str], int]:
    """Heuristic extraction of key values from drawing text.

    Returns: (tower_name, tower_count, width_m, depth_m, bay_m, height_m, cladding, barrier_1493_hits)
    """
    tower_name = None
    tower_count = None
    m = TOWER_RE.search(text)
    if m:
        tower_name = m.group(1).strip()
        tower_count = int(m.group(2))

    # Dimensions: look for common 2000/4000/12000 pattern (mm)
    candidates = smart_number_candidates(text)
    width_m = depth_m = bay_m = height_m = None
    # very common in plans: 2000, 4000, 12000
    if 12000 in candidates:
        height_m = 12.0
    # choose 4000 as width/depth and 2000 as bay
    if 4000 in candidates:
        width_m = depth_m = 4.0
    if 2000 in candidates:
        bay_m = 2.0

    # cladding
    cladding = None
    m2 = CLADDING_RE.search(text)
    if m2:
        cladding = m2.group(1).capitalize()

    barrier_hits = len(re.findall(r"1493\s*kg|1493kg", text, flags=re.IGNORECASE))

    return tower_name, tower_count, width_m, depth_m, bay_m, height_m, cladding, barrier_hits


# ------------------------------------------------------------
# BOM Computation (Ringlock-focused baseline)
# ------------------------------------------------------------
def compute_ringlock_bom(p: TowerParams, policy: BracePolicy) -> pd.DataFrame:
    bays_x = int(round(p.width_m / p.bay_m))
    bays_y = int(round(p.depth_m / p.bay_m))
    posts = (bays_x + 1) * (bays_y + 1)
    lifts = p.lifts

    # Standards: use 2.0m segments for simplicity.
    standards_2m = posts * lifts

    # Ledgers along X & Y at every lift
    ledgers_x = bays_x * (bays_y + 1) * lifts
    ledgers_y = bays_y * (bays_x + 1) * lifts
    ledgers_2m = ledgers_x + ledgers_y

    # Plan braces (assume 1 per bay square on alternate lifts if enabled)
    plan_levels = math.ceil(lifts / 2) if policy.plan_brace_alt_lifts else lifts
    plan_braces = bays_x * bays_y * plan_levels

    # Diagonal braces on elevations (faces)
    diag_levels = math.ceil(lifts / 2) if policy.diag_brace_alt_lifts else lifts
    diag_lines = 2 * bays_x + 2 * bays_y  # braces per level across four faces
    diag_braces = diag_lines * diag_levels

    # Base items
    base_plate = posts
    adjustable_jack = posts

    # Jersey barriers (if specified)
    jersey_barriers = p.barrier_1493_per_tower

    data = [
        ("Ringlock 2000 STANDARD", standards_2m),
        ("Ringlock 2.0m LEDGER", ledgers_2m),
        ("Ringlock 2.0m x 2.0m PLAN BRACE", plan_braces),
        ("Ringlock 2.0m x 2.0m DIAGONAL BRACE", diag_braces),
        ("Ringlock BASE PLATE", base_plate),
        ("Ringlock ADJUSTABLE JACK LONG", adjustable_jack),
        ("Accessories CONCRETE JERSEY BARRIER 1493kg", jersey_barriers),
    ]

    df = pd.DataFrame(data, columns=["Product", "Qty per Tower"])
    df["Towers"] = p.tower_count
    df["Total (All Towers)"] = df["Qty per Tower"] * df["Towers"]
    return df


def apply_extras(df: pd.DataFrame, extras_pct: float) -> pd.DataFrame:
    out = df.copy()
    out["Extras %"] = extras_pct
    out["Extras Qty"] = np.ceil(out["Total (All Towers)"] * extras_pct / 100.0).astype(int)
    out["Total for Delivery"] = out["Total (All Towers)"] + out["Extras Qty"]
    return out


def export_excel(project_header: Dict[str, str], df_final: pd.DataFrame) -> bytes:
    """Export a simple XLSX with a header sheet and a BOM sheet."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        # Header
        hdr_df = pd.DataFrame(list(project_header.items()), columns=["Field", "Value"])
        hdr_df.to_excel(writer, index=False, sheet_name="Header")

        # BOM
        df_final.to_excel(writer, index=False, sheet_name="BOM")

        # Formatting
        wb = writer.book
        ws_hdr = writer.sheets["Header"]
        ws_bom = writer.sheets["BOM"]
        for ws in (ws_hdr, ws_bom):
            ws.set_column(0, 0, 28)
            ws.set_column(1, 20, 18)
    return buffer.getvalue()


def export_pdf(project_header: Dict[str, str], df_final: pd.DataFrame) -> bytes:
    """Export a compact PDF using reportlab (optional dependency)."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
    except Exception:
        # If reportlab not installed, fall back to an empty PDF
        return b"%PDF-1.4\n%EOF"

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=24, rightMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Project Material List", styles["Title"]))
    story.append(Spacer(1, 8))
    for k, v in project_header.items():
        story.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Build table
    headers = list(df_final.columns)
    data = [headers] + df_final.astype(str).values.tolist()
    tbl = Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ]))
    story.append(tbl)

    doc.build(story)
    return buffer.getvalue()


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="Tower Spec → Project Material List", page_icon="🏗️", layout="wide")
st.title("🏗️ Tower Spec → Auto Project Material List")

with st.sidebar:
    st.header("Rules & Policies")
    with st.expander("Bracing policy"):
        plan_alt = st.checkbox("Plan braces on alternate lifts", value=True)
        diag_alt = st.checkbox("Diagonal braces on alternate lifts", value=True)
    extras_pct = st.slider("Extras uplift (%)", min_value=0, max_value=20, value=8, step=1)
    policy = BracePolicy(plan_brace_alt_lifts=plan_alt, diag_brace_alt_lifts=diag_alt)

st.markdown(
    """
This tool reads a tower drawing/spec and converts it to a **Project Material List** using parametric rules.  
You can override any detected value before generating the BOM.
"""
)

colA, colB = st.columns(2)
with colA:
    uploaded_spec = st.file_uploader("Upload tower drawing/spec (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])

with colB:
    st.info("Tip: You can still use the app without a drawing — just type values below.")

# Defaults (inspired by the example drawing showing a 4×4 m plan with 2 m bays and 12 m height)
params = TowerParams()
autofilled = False

if uploaded_spec is not None:
    file_bytes = uploaded_spec.read()
    text = parse_pdf_text(file_bytes)
    tower_name, tower_count, width_m, depth_m, bay_m, height_m, cladding, barrier_hits = extract_params_from_text(text)

    if tower_name: params.tower_name = tower_name
    if tower_count: params.tower_count = tower_count
    if width_m: params.width_m = width_m
    if depth_m: params.depth_m = depth_m
    if bay_m: params.bay_m = bay_m
    if height_m: params.height_m = height_m
    if cladding: params.cladding = cladding
    # We only use barrier hits as a hint for the user (count is engineering-driven)
    if barrier_hits and params.barrier_1493_per_tower == 0:
        params.barrier_1493_per_tower = 0  # do not guess; user must confirm
    autofilled = True

with st.form("confirm_params"):
    st.subheader("1) Confirm / Edit Detected Parameters")
    c1, c2, c3, c4 = st.columns(4)
    params.system = c1.selectbox("System", ["Ringlock", "Cuplok", "Other"], index=0)
    params.tower_name = c2.text_input("Structure Name", params.tower_name)
    params.tower_count = c3.number_input("Number of Towers", min_value=1, value=params.tower_count, step=1)
    params.cladding = c4.text_input("Cladding", params.cladding)

    c5, c6, c7, c8 = st.columns(4)
    params.width_m = c5.number_input("Plan Width (m)", min_value=1.0, value=float(params.width_m), step=0.5)
    params.depth_m = c6.number_input("Plan Depth (m)", min_value=1.0, value=float(params.depth_m), step=0.5)
    params.bay_m = c7.number_input("Bay Spacing (m)", min_value=0.5, value=float(params.bay_m), step=0.5)
    params.height_m = c8.number_input("Tower Height (m)", min_value=2.0, value=float(params.height_m), step=0.5)

    c9, c10, c11 = st.columns(3)
    params.lift_m = c9.number_input("Lift Interval (m)", min_value=1.0, value=float(params.lift_m), step=0.5)
    top_only = c10.checkbox("Platform at top only", value=True)
    barrier_1493 = c11.number_input("Concrete Jersey Barrier 1493kg per tower", min_value=0, value=int(params.barrier_1493_per_tower), step=1)
    params.barrier_1493_per_tower = barrier_1493
    params.platforms_at_lifts = [params.lifts] if top_only else list(range(1, params.lifts + 1))

    params.notes = st.text_area("Notes", value=("Autofilled from drawing." if autofilled else ""))

    submitted = st.form_submit_button("Generate BOM")

if submitted:
    if params.system != "Ringlock":
        st.warning("This POC computes Ringlock BOM rules. You can still export and then adjust your catalog mapping.")
    df = compute_ringlock_bom(params, policy)
    df_final = apply_extras(df, extras_pct)

    st.subheader("2) BOM Result")
    st.dataframe(df_final, use_container_width=True)

    # Project header for exports
    header = {
        "Structure": params.tower_name,
        "System": params.system,
        "Towers": str(params.tower_count),
        "Plan (m)": f"{params.width_m:.1f} × {params.depth_m:.1f} @ {params.bay_m:.1f}",
        "Height / Lift (m)": f"{params.height_m:.1f} / {params.lift_m:.1f} → {params.lifts} lifts",
        "Cladding": params.cladding,
        "Notes": params.notes,
    }

    # Export buttons
    excel_bytes = export_excel(header, df_final)
    pdf_bytes = export_pdf(header, df_final)

    col1, col2 = st.columns(2)
    col1.download_button(
        "⬇️ Download Excel (.xlsx)",
        data=excel_bytes,
        file_name="Project_Material_List.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    col2.download_button(
        "⬇️ Download PDF (.pdf)",
        data=pdf_bytes,
        file_name="Project_Material_List.pdf",
        mime="application/pdf",
    )

    # Calculation details for transparency
    with st.expander("How counts were calculated"):
        bays_x = int(round(params.width_m / params.bay_m))
        bays_y = int(round(params.depth_m / params.bay_m))
        posts = (bays_x + 1) * (bays_y + 1)
        st.write({
            "bays_x": bays_x,
            "bays_y": bays_y,
            "posts": posts,
            "lifts": params.lifts,
            "ledgers_x_per_lift": bays_x * (bays_y + 1),
            "ledgers_y_per_lift": bays_y * (bays_x + 1),
            "plan_brace_levels": math.ceil(params.lifts / 2) if policy.plan_brace_alt_lifts else params.lifts,
            "diag_brace_levels": math.ceil(params.lifts / 2) if policy.diag_brace_alt_lifts else params.lifts,
        })

st.caption("Engineered counts require review by a competent person before issue.")