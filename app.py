# app.py
# Simplified Enhanced Streamlit app: Tower Spec → Auto Material List
# Author: Claude (Simplified Enhanced version)
# Works with basic dependencies: streamlit, pandas, numpy

import io
import math
import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st

# Optional dependencies for enhanced functionality
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from pdf2image import convert_from_bytes
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError:
    convert_from_bytes = None
    pytesseract = None
    Image = None

# ============================================================================
# ENHANCED DATA STRUCTURES
# ============================================================================

class SystemType(Enum):
    RINGLOCK = "Ringlock"
    CUPLOK = "Cuplok"
    LION_DECK = "Lion Deck"
    HYBRID = "Hybrid"

class LoadCategory(Enum):
    LIGHT = "Light (≤3.0 kN/m²)"
    MEDIUM = "Medium (3.0-5.0 kN/m²)"
    HEAVY = "Heavy (>5.0 kN/m²)"

class CladdingType(Enum):
    NONE = "None"
    NETTING = "Netting"
    SHEETING = "Sheeting"
    SOLID_PANELS = "Solid Panels"
    MESH = "Mesh"

@dataclass
class WindLoading:
    design_speed_ms: float = 25.0
    exposure_category: str = "B"  # A, B, C, D
    terrain_factor: float = 1.0
    height_factor: float = 1.0
    
    def get_design_pressure(self, height_m: float) -> float:
        """Calculate wind pressure based on height and exposure"""
        base_pressure = 0.6 * (self.design_speed_ms ** 2) / 1000  # kN/m²
        height_factor = min(1.2, 1.0 + (height_m - 10) * 0.02)
        return base_pressure * self.terrain_factor * height_factor

@dataclass
class TowerParams:
    # Basic geometry
    system: SystemType = SystemType.RINGLOCK
    tower_name: str = "Tower LX"
    tower_count: int = 1
    width_m: float = 4.0
    depth_m: float = 4.0
    bay_m: float = 2.0
    height_m: float = 12.0
    lift_m: float = 2.0
    
    # Loading and access
    load_category: LoadCategory = LoadCategory.MEDIUM
    platform_levels: List[int] = field(default_factory=lambda: [6])
    access_levels: List[int] = field(default_factory=lambda: [6])
    cladding: CladdingType = CladdingType.NETTING
    
    # Foundation and ballast
    base_condition: str = "Level concrete"
    ballast_750kg: int = 0
    ballast_1000kg: int = 0
    barrier_1493kg: int = 0
    
    # Environmental
    wind_loading: WindLoading = field(default_factory=WindLoading)
    exposure_class: str = "External"
    
    # Special requirements
    cantilever_m: float = 0.0
    roof_loading: bool = False
    seismic_zone: str = "Low"
    
    # Validation flags
    engineering_approved: bool = False
    notes: str = ""

    @property
    def lifts(self) -> int:
        return max(1, int(math.ceil(self.height_m / self.lift_m)))
    
    @property
    def total_area_m2(self) -> float:
        return self.width_m * self.depth_m * self.tower_count

@dataclass
class BOMRules:
    # Bracing policies
    plan_brace_alt_lifts: bool = True
    diag_brace_alt_lifts: bool = True
    ledger_every_lift: bool = True
    
    # Safety requirements
    tie_frequency_lifts: int = 4
    min_ballast_per_tower: int = 2
    guardrail_required: bool = True
    
    # Load factors
    live_load_factor: float = 1.4
    wind_load_factor: float = 1.2
    dead_load_factor: float = 1.2
    
    # System-specific factors
    ringlock_efficiency: float = 1.0
    cuplok_efficiency: float = 1.1
    hybrid_complexity_factor: float = 1.2

# ============================================================================
# ENHANCED DRAWING ANALYSIS
# ============================================================================

class DrawingAnalyzer:
    def __init__(self):
        self.patterns = {
            'tower_info': [
                re.compile(r"Tower\s+([A-Za-z0-9\-\s]+)\s*\((\d+)\s*nos?\.?\)", re.IGNORECASE),
                re.compile(r"Structure\s*:\s*([A-Za-z0-9\-\s]+)", re.IGNORECASE),
                re.compile(r"([A-Za-z0-9\-\s]+\s*Tower)", re.IGNORECASE)
            ],
            'dimensions': [
                re.compile(r"(\d{3,5})\s*(?:mm|MM)", re.IGNORECASE),
                re.compile(r"(\d+\.?\d*)\s*(?:m|M)\s*[x×]\s*(\d+\.?\d*)\s*(?:m|M)", re.IGNORECASE),
                re.compile(r"Plan\s*:\s*(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)", re.IGNORECASE)
            ],
            'ballast': [
                re.compile(r"(\d+)\s*(?:kg|KG).*ballast", re.IGNORECASE),
                re.compile(r"ballast.*(\d+)\s*(?:kg|KG)", re.IGNORECASE),
                re.compile(r"barrier.*(\d{4})\s*(?:kg|KG)", re.IGNORECASE)
            ],
            'cladding': [
                re.compile(r"Cladding\s*(?:type\s*)?:?\s*([A-Za-z]+)", re.IGNORECASE),
                re.compile(r"([A-Za-z]+)\s*cladding", re.IGNORECASE)
            ],
            'system': [
                re.compile(r"(Ringlock|Cuplok|Lion\s*Deck)", re.IGNORECASE),
                re.compile(r"System\s*:\s*([A-Za-z\s]+)", re.IGNORECASE)
            ]
        }
    
    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        """Enhanced text extraction with preprocessing"""
        text = ""
        
        # Try pdfplumber first
        if pdfplumber is not None:
            try:
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                st.warning(f"PDF text extraction failed: {e}")
        
        # OCR fallback
        if not text.strip() and convert_from_bytes and pytesseract:
            try:
                images = convert_from_bytes(file_bytes, dpi=300)
                for img in images:
                    if Image:
                        # Enhance image for better OCR
                        if img.mode != 'L':
                            img = img.convert('L')
                        enhancer = ImageEnhance.Contrast(img)
                        img = enhancer.enhance(1.5)
                    text += pytesseract.image_to_string(img, config='--psm 6') + "\n"
            except Exception as e:
                st.warning(f"OCR extraction failed: {e}")
        
        return text
    
    def analyze_drawing(self, text: str) -> Dict[str, Any]:
        """Comprehensive drawing analysis"""
        results = {
            'tower_name': None,
            'tower_count': 1,
            'system_type': SystemType.RINGLOCK,
            'dimensions': {'width_m': None, 'depth_m': None, 'height_m': None},
            'bay_spacing': None,
            'loads': [],
            'ballast_items': [],
            'cladding_type': CladdingType.NETTING,
            'confidence_score': 0.0
        }
        
        confidence_points = 0
        max_points = 10
        
        # Extract tower information
        for pattern in self.patterns['tower_info']:
            match = pattern.search(text)
            if match:
                if len(match.groups()) >= 2:
                    results['tower_name'] = match.group(1).strip()
                    try:
                        results['tower_count'] = int(match.group(2))
                        confidence_points += 2
                    except ValueError:
                        pass
                else:
                    results['tower_name'] = match.group(1).strip()
                    confidence_points += 1
                break
        
        # Extract dimensions
        dimensions_found = self._extract_dimensions(text)
        if dimensions_found:
            results['dimensions'].update(dimensions_found)
            confidence_points += 2
        
        # Extract system type
        for pattern in self.patterns['system']:
            match = pattern.search(text)
            if match:
                system_text = match.group(1).lower()
                if 'ringlock' in system_text:
                    results['system_type'] = SystemType.RINGLOCK
                elif 'cuplok' in system_text:
                    results['system_type'] = SystemType.CUPLOK
                elif 'lion' in system_text:
                    results['system_type'] = SystemType.LION_DECK
                confidence_points += 1
                break
        
        # Extract ballast information
        ballast_items = self._extract_ballast_info(text)
        if ballast_items:
            results['ballast_items'] = ballast_items
            confidence_points += 1
        
        # Extract cladding type
        for pattern in self.patterns['cladding']:
            match = pattern.search(text)
            if match:
                cladding_text = match.group(1).lower()
                if 'netting' in cladding_text:
                    results['cladding_type'] = CladdingType.NETTING
                elif 'sheet' in cladding_text:
                    results['cladding_type'] = CladdingType.SHEETING
                elif 'mesh' in cladding_text:
                    results['cladding_type'] = CladdingType.MESH
                elif 'panel' in cladding_text:
                    results['cladding_type'] = CladdingType.SOLID_PANELS
                confidence_points += 1
                break
        
        results['confidence_score'] = confidence_points / max_points
        return results
    
    def _extract_dimensions(self, text: str) -> Dict[str, float]:
        """Extract dimensional information"""
        dims = {}
        
        # Look for plan dimensions
        plan_match = re.search(r"(\d+)\s*[x×]\s*(\d+)", text)
        if plan_match:
            try:
                dim1, dim2 = int(plan_match.group(1)), int(plan_match.group(2))
                # Convert mm to m if needed
                if dim1 > 100:  # Assume mm
                    dim1, dim2 = dim1/1000, dim2/1000
                dims['width_m'] = float(dim1)
                dims['depth_m'] = float(dim2)
            except ValueError:
                pass
        
        # Look for height
        height_match = re.search(r"(\d{4,5})", text)
        if height_match:
            try:
                height = int(height_match.group(1))
                if height > 1000:  # Assume mm
                    dims['height_m'] = height / 1000
                else:
                    dims['height_m'] = float(height)
            except ValueError:
                pass
        
        return dims
    
    def _extract_ballast_info(self, text: str) -> List[Dict[str, Any]]:
        """Extract ballast and foundation information"""
        ballast_items = []
        
        # Look for specific ballast weights
        ballast_matches = re.findall(r"(\d+)\s*(?:kg|KG)", text)
        for match in ballast_matches:
            weight = int(match)
            if weight in [750, 1000, 1493]:
                ballast_items.append({
                    'type': f"Ballast {weight}kg",
                    'weight_kg': weight,
                    'quantity': text.count(f"{weight}kg")
                })
        
        return ballast_items

# ============================================================================
# ENHANCED BOM CALCULATION ENGINE
# ============================================================================

class BOMCalculator:
    def __init__(self, rules: BOMRules):
        self.rules = rules
    
    def calculate_comprehensive_bom(self, params: TowerParams) -> pd.DataFrame:
        """Calculate comprehensive BOM based on system type"""
        if params.system == SystemType.RINGLOCK:
            return self._calculate_ringlock_bom(params)
        elif params.system == SystemType.CUPLOK:
            return self._calculate_cuplok_bom(params)
        elif params.system == SystemType.LION_DECK:
            return self._calculate_lion_deck_bom(params)
        elif params.system == SystemType.HYBRID:
            return self._calculate_hybrid_bom(params)
        else:
            raise ValueError(f"Unsupported system type: {params.system}")
    
    def _calculate_ringlock_bom(self, params: TowerParams) -> pd.DataFrame:
        """Enhanced Ringlock BOM calculation"""
        bays_x = max(1, int(round(params.width_m / params.bay_m)))
        bays_y = max(1, int(round(params.depth_m / params.bay_m)))
        posts = (bays_x + 1) * (bays_y + 1)
        lifts = params.lifts
        
        # Calculate components
        components = []
        
        # Standards - use 2.0m for consistency
        standard_length = "2.0m"
        standards_qty = posts * lifts
        components.append((f"Ringlock STANDARD {standard_length}", standards_qty))
        
        # Ledgers
        ledger_length = "2.0m"
        ledgers_x = bays_x * (bays_y + 1) * lifts
        ledgers_y = bays_y * (bays_x + 1) * lifts
        total_ledgers = ledgers_x + ledgers_y
        components.append((f"Ringlock LEDGER {ledger_length}", total_ledgers))
        
        # Bracing calculations with wind consideration
        wind_pressure = params.wind_loading.get_design_pressure(params.height_m)
        bracing_factor = min(1.5, 1.0 + wind_pressure * 0.1)
        
        # Plan braces
        plan_levels = math.ceil(lifts / 2) if self.rules.plan_brace_alt_lifts else lifts
        plan_braces = int((bays_x * bays_y * plan_levels) * bracing_factor)
        components.append(("Ringlock PLAN BRACE 2.0m x 2.0m", plan_braces))
        
        # Diagonal braces
        diag_levels = math.ceil(lifts / 2) if self.rules.diag_brace_alt_lifts else lifts
        perimeter_bays = 2 * (bays_x + bays_y)
        diag_braces = int((perimeter_bays * diag_levels) * bracing_factor)
        components.append(("Ringlock DIAGONAL BRACE 2.0m x 2.0m", diag_braces))
        
        # Base components
        components.append(("Ringlock BASE PLATE", posts))
        components.append(("Ringlock ADJUSTABLE JACK LONG", posts))
        
        # Platform components for specified levels
        for level in params.platform_levels:
            platform_area = params.width_m * params.depth_m
            decking_panels = math.ceil(platform_area / 2.0)
            guardrail_length = 2 * (params.width_m + params.depth_m)
            guardrail_sections = math.ceil(guardrail_length / 2.0)
            
            components.append((f"Steel Deck 4' x 4' (Level {level})", decking_panels))
            components.append((f"Guardrail 2.0m (Level {level})", guardrail_sections))
        
        # Access components
        if params.access_levels:
            stair_flights = len(params.access_levels)
            components.append(("Staircase 2.0m width", stair_flights))
            components.append(("Landing Frame", stair_flights))
        
        # Ballast and anchoring
        if params.barrier_1493kg > 0:
            components.append(("Concrete Jersey Barrier 1493kg", params.barrier_1493kg))
        if params.ballast_1000kg > 0:
            components.append(("Ballast 1000kg", params.ballast_1000kg))
        if params.ballast_750kg > 0:
            components.append(("Ballast 750kg", params.ballast_750kg))
        
        # Ties and anchors (based on height)
        if params.height_m > 8.0:
            tie_levels = math.ceil(params.lifts / self.rules.tie_frequency_lifts)
            perimeter_ties = 2 * (math.ceil(params.width_m / 4) + math.ceil(params.depth_m / 4))
            tie_points = tie_levels * perimeter_ties
            components.append(("Tie Rods with Anchors", tie_points))
        
        # Cladding attachments
        if params.cladding != CladdingType.NONE:
            cladding_area = 2 * (params.width_m + params.depth_m) * params.height_m
            cladding_fixings = math.ceil(cladding_area * 2)  # 2 fixings per m²
            components.append((f"Cladding Fixings ({params.cladding.value})", cladding_fixings))
        
        # Create DataFrame
        df = pd.DataFrame(components, columns=["Product", "Qty per Tower"])
        df["Towers"] = params.tower_count
        df["Total (All Towers)"] = df["Qty per Tower"] * df["Towers"]
        
        return df
    
    def _calculate_cuplok_bom(self, params: TowerParams) -> pd.DataFrame:
        """Cuplok system BOM calculation"""
        bays_x = max(1, int(round(params.width_m / params.bay_m)))
        bays_y = max(1, int(round(params.depth_m / params.bay_m)))
        posts = (bays_x + 1) * (bays_y + 1)
        lifts = params.lifts
        
        components = []
        
        # Cuplok standards
        standards_qty = posts * lifts
        components.append(("Cuplok STANDARD 2.0m", standards_qty))
        
        # Cuplok ledgers
        ledgers_x = bays_x * (bays_y + 1) * lifts
        ledgers_y = bays_y * (bays_x + 1) * lifts
        total_ledgers = ledgers_x + ledgers_y
        components.append(("Cuplok LEDGER 2.0m", total_ledgers))
        
        # Base components
        components.append(("Cuplok ADJUSTABLE BASE JACK", posts))
        
        # Tube and fitting bracing
        perimeter_bays = 2 * (bays_x + bays_y)
        tf_braces = perimeter_bays * math.ceil(lifts / 2)
        components.append(("T&F TUBE 2.0m", tf_braces))
        components.append(("T&F SWIVELS", tf_braces * 2))
        
        df = pd.DataFrame(components, columns=["Product", "Qty per Tower"])
        df["Towers"] = params.tower_count
        df["Total (All Towers)"] = df["Qty per Tower"] * df["Towers"]
        
        return df
    
    def _calculate_lion_deck_bom(self, params: TowerParams) -> pd.DataFrame:
        """Lion Deck system BOM calculation"""
        components = []
        
        total_platform_area = 0
        for level in params.platform_levels:
            platform_area = params.width_m * params.depth_m
            total_platform_area += platform_area
        
        if total_platform_area > 0:
            # Primary beams
            primary_beam_count = math.ceil(total_platform_area / 4.0)
            components.append(("Lion Deck PRIMARY BEAM 2.0m", primary_beam_count))
            
            # Secondary beams
            secondary_beam_count = primary_beam_count * 2
            components.append(("Lion Deck SECONDARY BEAM 2.0m", secondary_beam_count))
            
            # Plywood decking
            ply_panels = math.ceil(total_platform_area / 2.0)
            components.append(("Lion Deck PLY 2.0m x 1.0m", ply_panels))
            
            # Handrail system
            handrail_length = len(params.platform_levels) * 2 * (params.width_m + params.depth_m)
            handrail_sections = math.ceil(handrail_length / 2.0)
            components.append(("Lion Deck HANDRAIL STEEL FRAME 2.0m", handrail_sections))
        
        df = pd.DataFrame(components, columns=["Product", "Qty per Tower"])
        df["Towers"] = params.tower_count
        df["Total (All Towers)"] = df["Qty per Tower"] * df["Towers"]
        
        return df
    
    def _calculate_hybrid_bom(self, params: TowerParams) -> pd.DataFrame:
        """Hybrid system BOM calculation"""
        # Combine Ringlock structure with Lion Deck platforms
        ringlock_bom = self._calculate_ringlock_bom(params)
        lion_deck_bom = self._calculate_lion_deck_bom(params)
        
        combined_components = []
        
        # Add Ringlock structural components
        for _, row in ringlock_bom.iterrows():
            if any(x in row["Product"].lower() for x in ["standard", "ledger", "brace", "base", "jack"]):
                combined_components.append((row["Product"], row["Qty per Tower"]))
        
        # Add Lion Deck platform components
        for _, row in lion_deck_bom.iterrows():
            combined_components.append((row["Product"], row["Qty per Tower"]))
        
        # Apply complexity factor
        for i in range(len(combined_components)):
            qty = int(combined_components[i][1] * self.rules.hybrid_complexity_factor)
            combined_components[i] = (combined_components[i][0], qty)
        
        df = pd.DataFrame(combined_components, columns=["Product", "Qty per Tower"])
        df["Towers"] = params.tower_count
        df["Total (All Towers)"] = df["Qty per Tower"] * df["Towers"]
        
        return df

# ============================================================================
# ENGINEERING VALIDATION
# ============================================================================

class EngineeringValidator:
    def __init__(self):
        self.max_height_m = 50.0
        self.max_area_m2 = 400.0
        self.max_unsupported_height_m = 12.0
    
    def validate_structure(self, params: TowerParams) -> Dict[str, Any]:
        """Comprehensive structural validation"""
        warnings = []
        errors = []
        recommendations = []
        
        # Height checks
        if params.height_m > self.max_height_m:
            errors.append(f"Height {params.height_m}m exceeds maximum allowed {self.max_height_m}m")
        elif params.height_m > self.max_unsupported_height_m:
            warnings.append(f"Height {params.height_m}m requires adequate tying for stability")
        
        # Area checks
        total_area = params.total_area_m2
        if total_area > self.max_area_m2:
            warnings.append(f"Total area {total_area:.1f}m² is large - verify foundation capacity")
        
        # Aspect ratio checks
        aspect_ratio = max(params.width_m, params.depth_m) / min(params.width_m, params.depth_m)
        if aspect_ratio > 3.0:
            warnings.append(f"High aspect ratio {aspect_ratio:.1f} may cause stability issues")
        
        # Bay spacing checks
        if params.bay_m > 3.0:
            warnings.append(f"Bay spacing {params.bay_m}m is large - check beam capacity")
        
        # Wind loading assessment
        wind_pressure = params.wind_loading.get_design_pressure(params.height_m)
        if wind_pressure > 1.5:
            warnings.append(f"High wind pressure {wind_pressure:.2f} kN/m² - additional bracing recommended")
        
        # Ballast recommendations
        total_ballast = (params.ballast_750kg * 750 + 
                        params.ballast_1000kg * 1000 + 
                        params.barrier_1493kg * 1493)
        structure_weight = params.total_area_m2 * params.height_m * 10  # kg estimate
        ballast_ratio = total_ballast / max(structure_weight, 1000)
        
        if ballast_ratio < 0.5:
            recommendations.append("Consider additional ballast for overturning stability")
        
        # Calculate safety rating
        if len(errors) > 0:
            rating = "UNSAFE"
        elif len(warnings) > 2:
            rating = "CAUTION"
        elif len(warnings) > 0:
            rating = "REVIEW REQUIRED"
        else:
            rating = "ACCEPTABLE"
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'recommendations': recommendations,
            'overall_rating': rating,
            'wind_pressure': wind_pressure,
            'ballast_ratio': ballast_ratio
        }

# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

st.set_page_config(
    page_title="Enhanced Tower BOM Generator", 
    page_icon="🏗️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'current_params' not in st.session_state:
    st.session_state.current_params = TowerParams()
if 'bom_calculated' not in st.session_state:
    st.session_state.bom_calculated = False

# Main title
st.title("🏗️ Enhanced Tower Specification → Automated BOM Generator")
st.markdown("*Multi-system scaffolding BOM generator with engineering validation*")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # BOM Rules
    with st.expander("📋 BOM Rules", expanded=False):
        plan_brace_alt = st.checkbox("Plan braces on alternate lifts", value=True)
        diag_brace_alt = st.checkbox("Diagonal braces on alternate lifts", value=True)
        tie_frequency = st.slider("Tie frequency (lifts)", 3, 8, 4)
        min_ballast = st.slider("Min ballast per tower", 0, 10, 2)
    
    # Safety factors
    with st.expander("🛡️ Safety Factors", expanded=False):
        live_load_factor = st.slider("Live load factor", 1.0, 2.0, 1.4, 0.1)
        wind_load_factor = st.slider("Wind load factor", 1.0, 2.0, 1.2, 0.1)
        extras_pct = st.slider("Extras percentage", 0, 20, 8, 1)
    
    # Create rules object
    bom_rules = BOMRules(
        plan_brace_alt_lifts=plan_brace_alt,
        diag_brace_alt_lifts=diag_brace_alt,
        tie_frequency_lifts=tie_frequency,
        min_ballast_per_tower=min_ballast,
        live_load_factor=live_load_factor,
        wind_load_factor=wind_load_factor
    )

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📄 Drawing Analysis", "⚙️ Parameters", "📊 BOM Generation", "🔍 Validation"])

# ============================================================================
# TAB 1: DRAWING ANALYSIS
# ============================================================================
with tab1:
    st.header("Drawing Analysis & Parameter Extraction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload tower drawing/specification", 
            type=["pdf", "png", "jpg", "jpeg"],
            help="Upload technical drawings, specifications, or photos of drawings"
        )
        
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            
            with st.spinner("Analyzing drawing..."):
                analyzer = DrawingAnalyzer()
                text_content = analyzer.extract_text_from_pdf(file_bytes)
                analysis_results = analyzer.analyze_drawing(text_content)
                st.session_state.analysis_results = analysis_results
                
                # Update session state parameters based on analysis
                if analysis_results['tower_name']:
                    st.session_state.current_params.tower_name = analysis_results['tower_name']
                if analysis_results['tower_count']:
                    st.session_state.current_params.tower_count = analysis_results['tower_count']
                if analysis_results['system_type']:
                    st.session_state.current_params.system = analysis_results['system_type']
                
                # Update dimensions with validation
                dims = analysis_results['dimensions']
                if dims.get('width_m') and dims['width_m'] >= 1.0:
                    st.session_state.current_params.width_m = dims['width_m']
                if dims.get('depth_m') and dims['depth_m'] >= 1.0:
                    st.session_state.current_params.depth_m = dims['depth_m']
                if dims.get('height_m') and dims['height_m'] >= 2.0:
                    st.session_state.current_params.height_m = dims['height_m']
                elif dims.get('height_m') and dims['height_m'] < 2.0:
                    # If detected height is too low, scale it up (might be in wrong units)
                    scaled_height = dims['height_m'] * 1000 / 1000  # Keep as is but add warning
                    if scaled_height >= 2.0:
                        st.session_state.current_params.height_m = scaled_height
                    else:
                        st.warning(f"Detected height {dims['height_m']:.1f}m seems too low. Please verify in Parameters tab.")
                
                st.success(f"Drawing analyzed! Confidence: {analysis_results['confidence_score']:.1%}")
    
    with col2:
        if st.session_state.analysis_results:
            st.subheader("🔍 Detection Results")
            results = st.session_state.analysis_results
            
            # Display confidence score
            confidence = results['confidence_score']
            if confidence > 0.7:
                st.success(f"High confidence: {confidence:.1%}")
            elif confidence > 0.4:
                st.warning(f"Medium confidence: {confidence:.1%}")
            else:
                st.error(f"Low confidence: {confidence:.1%}")
            
            # Display extracted parameters
            st.write("**Detected Parameters:**")
            if results['tower_name']:
                st.write(f"• Tower: {results['tower_name']}")
            if results['tower_count']:
                st.write(f"• Quantity: {results['tower_count']}")
            if results['system_type']:
                st.write(f"• System: {results['system_type'].value}")
            
            dims = results['dimensions']
            if any(dims.values()):
                st.write("**Dimensions:**")
                if dims.get('width_m'):
                    st.write(f"• Width: {dims['width_m']:.1f}m")
                if dims.get('depth_m'):
                    st.write(f"• Depth: {dims['depth_m']:.1f}m")
                if dims.get('height_m'):
                    st.write(f"• Height: {dims['height_m']:.1f}m")

# ============================================================================
# TAB 2: PARAMETERS
# ============================================================================
with tab2:
    st.header("Structure Parameters")
    
    with st.form("parameter_form"):
        # Basic Information
        st.subheader("🏗️ Basic Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            system_type = st.selectbox(
                "System Type",
                options=[s.value for s in SystemType],
                index=[s.value for s in SystemType].index(st.session_state.current_params.system.value)
            )
            st.session_state.current_params.system = SystemType(system_type)
        
        with col2:
            st.session_state.current_params.tower_name = st.text_input(
                "Tower Name", 
                value=st.session_state.current_params.tower_name
            )
        
        with col3:
            st.session_state.current_params.tower_count = st.number_input(
                "Number of Towers", 
                min_value=1, 
                value=st.session_state.current_params.tower_count
            )
        
        with col4:
            load_category = st.selectbox(
                "Load Category",
                options=[l.value for l in LoadCategory],
                index=1  # Default to Medium
            )
            st.session_state.current_params.load_category = LoadCategory(load_category)
        
        # Geometry
        st.subheader("📐 Geometry")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.session_state.current_params.width_m = st.number_input(
                "Width (m)", 
                min_value=1.0, 
                value=st.session_state.current_params.width_m, 
                step=0.5
            )
        
        with col2:
            st.session_state.current_params.depth_m = st.number_input(
                "Depth (m)", 
                min_value=1.0, 
                value=st.session_state.current_params.depth_m, 
                step=0.5
            )
        
        with col3:
            # Use max() to ensure we don't go below minimum
            current_height = max(2.0, st.session_state.current_params.height_m)
            st.session_state.current_params.height_m = st.number_input(
                "Height (m)", 
                min_value=0.5,  # Lower minimum to accommodate detection errors
                max_value=100.0,
                value=current_height, 
                step=0.5,
                help="Detected values below 2.0m may indicate unit conversion issues"
            )
        
        with col4:
            st.session_state.current_params.lift_m = st.number_input(
                "Lift Height (m)", 
                min_value=1.0, 
                value=st.session_state.current_params.lift_m, 
                step=0.5
            )
        
        # Bay spacing
        st.session_state.current_params.bay_m = st.slider(
            "Bay Spacing (m)", 
            min_value=1.0, 
            max_value=3.0, 
            value=st.session_state.current_params.bay_m, 
            step=0.5
        )
        
        # Platforms and Access
        st.subheader("🚶 Platforms & Access")
        col1, col2 = st.columns(2)
        
        with col1:
            max_lifts = st.session_state.current_params.lifts
            platform_levels = st.multiselect(
                "Platform Levels (Lift Numbers)",
                options=list(range(1, max_lifts + 1)),
                default=[max_lifts] if max_lifts > 0 else [1]
            )
            st.session_state.current_params.platform_levels = platform_levels
        
        with col2:
            access_levels = st.multiselect(
                "Access Levels (Lift Numbers)",
                options=list(range(1, max_lifts + 1)),
                default=platform_levels[:1] if platform_levels else [1]
            )
            st.session_state.current_params.access_levels = access_levels
        
        # Environmental Conditions
        st.subheader("🌬️ Environmental Conditions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cladding_type = st.selectbox(
                "Cladding Type",
                options=[c.value for c in CladdingType],
                index=1  # Default to Netting
            )
            st.session_state.current_params.cladding = CladdingType(cladding_type)
        
        with col2:
            wind_speed = st.slider(
                "Design Wind Speed (m/s)", 
                min_value=15, 
                max_value=50, 
                value=25
            )
            st.session_state.current_params.wind_loading.design_speed_ms = wind_speed
        
        with col3:
            exposure = st.selectbox(
                "Exposure Category",
                options=["A", "B", "C", "D"],
                index=1
            )
            st.session_state.current_params.wind_loading.exposure_category = exposure
        
        # Foundation & Ballast
        st.subheader("⚓ Foundation & Ballast")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.session_state.current_params.base_condition = st.selectbox(
                "Base Condition",
                options=["Level concrete", "Uneven ground", "Slope", "Soft ground"]
            )
        
        with col2:
            st.session_state.current_params.ballast_750kg = st.number_input(
                "750kg Ballast per Tower", 
                min_value=0, 
                value=st.session_state.current_params.ballast_750kg
            )
        
        with col3:
            st.session_state.current_params.ballast_1000kg = st.number_input(
                "1000kg Ballast per Tower", 
                min_value=0, 
                value=st.session_state.current_params.ballast_1000kg
            )
        
        with col4:
            st.session_state.current_params.barrier_1493kg = st.number_input(
                "1493kg Barriers per Tower", 
                min_value=0, 
                value=st.session_state.current_params.barrier_1493kg
            )
        
        # Notes
        st.session_state.current_params.notes = st.text_area(
            "Additional Notes",
            value=st.session_state.current_params.notes,
            height=100
        )
        
        # Form submission
        submitted = st.form_submit_button("🔄 Update Parameters", type="primary")
        
        if submitted:
            st.success("Parameters updated successfully!")
            st.session_state.bom_calculated = False

# ============================================================================
# TAB 3: BOM GENERATION
# ============================================================================
with tab3:
    st.header("Bill of Materials Generation")
    
    if st.button("🧮 Calculate BOM", type="primary", use_container_width=True):
        with st.spinner("Calculating comprehensive BOM..."):
            calculator = BOMCalculator(bom_rules)
            bom_df = calculator.calculate_comprehensive_bom(st.session_state.current_params)
            
            # Apply extras
            bom_df["Extras %"] = extras_pct
            bom_df["Extras Qty"] = np.ceil(bom_df["Total (All Towers)"] * extras_pct / 100.0).astype(int)
            bom_df["Total for Delivery"] = bom_df["Total (All Towers)"] + bom_df["Extras Qty"]
            
            st.session_state.bom_df = bom_df
            st.session_state.bom_calculated = True
    
    if st.session_state.bom_calculated and 'bom_df' in st.session_state:
        bom_df = st.session_state.bom_df
        
        # Display BOM summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Line Items", len(bom_df))
        with col2:
            st.metric("Total Components", bom_df["Total for Delivery"].sum())
        with col3:
            estimated_weight = bom_df["Total for Delivery"].sum() * 5
            st.metric("Estimated Weight (kg)", f"{estimated_weight:,}")
        
        # Display BOM table
        st.subheader("📋 Detailed Bill of Materials")
        
        # Add filtering options
        col1, col2 = st.columns(2)
        with col1:
            search_term = st.text_input("🔍 Search components:", placeholder="e.g., Standard, Ledger, Brace")
        with col2:
            min_qty = st.number_input("Min quantity filter:", min_value=0, value=0)
        
        # Filter DataFrame
        filtered_df = bom_df.copy()
        if search_term:
            mask = filtered_df['Product'].str.contains(search_term, case=False, na=False)
            filtered_df = filtered_df[mask]
        if min_qty > 0:
            filtered_df = filtered_df[filtered_df['Total for Delivery'] >= min_qty]
        
        # Display filtered results
        st.dataframe(filtered_df, use_container_width=True, height=400)
        
        # Export options
        st.subheader("📤 Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV export
            csv_buffer = io.StringIO()
            filtered_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                "📄 Download CSV",
                data=csv_buffer.getvalue(),
                file_name=f"BOM_{st.session_state.current_params.tower_name.replace(' ', '_')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # JSON export
            json_data = {
                "project_info": {
                    "name": st.session_state.current_params.tower_name,
                    "system": st.session_state.current_params.system.value,
                    "towers": st.session_state.current_params.tower_count,
                    "generated": pd.Timestamp.now().isoformat()
                },
                "bom": filtered_df.to_dict('records')
            }
            
            st.download_button(
                "🔗 Download JSON",
                data=json.dumps(json_data, indent=2),
                file_name=f"BOM_{st.session_state.current_params.tower_name.replace(' ', '_')}.json",
                mime="application/json"
            )
        
        with col3:
            # Text summary
            summary_text = f"""Project: {st.session_state.current_params.tower_name}
System: {st.session_state.current_params.system.value}
Towers: {st.session_state.current_params.tower_count}
Dimensions: {st.session_state.current_params.width_m}×{st.session_state.current_params.depth_m}×{st.session_state.current_params.height_m}m

Bill of Materials:
"""
            for _, row in filtered_df.iterrows():
                summary_text += f"{row['Product']}: {row['Total for Delivery']}\n"
            
            st.download_button(
                "📄 Download Summary",
                data=summary_text,
                file_name=f"Summary_{st.session_state.current_params.tower_name.replace(' ', '_')}.txt",
                mime="text/plain"
            )

# ============================================================================
# TAB 4: VALIDATION
# ============================================================================
with tab4:
    st.header("Engineering Validation & Analysis")
    
    # Engineering validation
    st.subheader("🛡️ Structural Validation")
    
    validator = EngineeringValidator()
    validation_results = validator.validate_structure(st.session_state.current_params)
    
    # Display validation status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if validation_results['is_valid']:
            st.success("✅ Structure Validated")
        else:
            st.error("❌ Validation Failed")
    
    with col2:
        rating = validation_results['overall_rating']
        if rating == "ACCEPTABLE":
            st.success(f"Rating: {rating}")
        elif rating == "REVIEW REQUIRED":
            st.warning(f"Rating: {rating}")
        else:
            st.error(f"Rating: {rating}")
    
    with col3:
        wind_pressure = validation_results['wind_pressure']
        st.metric("Wind Pressure", f"{wind_pressure:.2f} kN/m²")
    
    with col4:
        ballast_ratio = validation_results['ballast_ratio']
        st.metric("Ballast Ratio", f"{ballast_ratio:.2f}")
    
    # Display detailed validation results
    if validation_results['errors']:
        st.error("**Critical Errors:**")
        for error in validation_results['errors']:
            st.error(f"• {error}")
    
    if validation_results['warnings']:
        st.warning("**Warnings:**")
        for warning in validation_results['warnings']:
            st.warning(f"• {warning}")
    
    if validation_results['recommendations']:
        st.info("**Recommendations:**")
        for rec in validation_results['recommendations']:
            st.info(f"• {rec}")
    
    st.divider()
    
    # Structure summary
    st.subheader("📊 Structure Summary")
    params = st.session_state.current_params
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Height", f"{params.height_m:.1f} m")
        st.metric("Plan Area", f"{params.width_m * params.depth_m:.1f} m²")
        st.metric("Number of Lifts", params.lifts)
    
    with col2:
        st.metric("Bay Spacing", f"{params.bay_m:.1f} m")
        st.metric("Platform Levels", len(params.platform_levels))
        st.metric("System Type", params.system.value)
    
    with col3:
        st.metric("Load Category", params.load_category.value)
        st.metric("Cladding", params.cladding.value)
        total_ballast = (params.ballast_750kg * 750 + 
                        params.ballast_1000kg * 1000 + 
                        params.barrier_1493kg * 1493)
        st.metric("Total Ballast", f"{total_ballast:,} kg")
    
    # Simple text-based analysis
    if st.session_state.bom_calculated and 'bom_df' in st.session_state:
        st.subheader("📈 BOM Analysis")
        bom_df = st.session_state.bom_df
        
        # Component category analysis
        standards = bom_df[bom_df['Product'].str.contains('STANDARD', case=False, na=False)]['Total for Delivery'].sum()
        ledgers = bom_df[bom_df['Product'].str.contains('LEDGER', case=False, na=False)]['Total for Delivery'].sum()
        braces = bom_df[bom_df['Product'].str.contains('BRACE', case=False, na=False)]['Total for Delivery'].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Standards", standards)
        with col2:
            st.metric("Ledgers", ledgers)
        with col3:
            st.metric("Braces", braces)
        
        # Show top components
        st.subheader("Top Components by Quantity")
        top_components = bom_df.nlargest(5, 'Total for Delivery')[['Product', 'Total for Delivery']]
        st.dataframe(top_components, use_container_width=True)
    
    else:
        st.info("Generate BOM first to see detailed analysis.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    <p>Enhanced Scaffolding BOM Generator v2.0 | 
    ⚠️ All calculations require verification by a competent person before use | 
    🔒 Engineering approval required for critical applications</p>
</div>
""", unsafe_allow_html=True)
