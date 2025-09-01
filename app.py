# app.py
# Multi-Structure Streamlit app: Tower/Bridge/Wall Spec → Auto Material List
# Author: Claude (Multi-Structure version)
# Handles: Towers, Bridges, Walls, and other scaffolding structures

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

# Optional dependencies
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
# ENHANCED DATA STRUCTURES FOR MULTIPLE STRUCTURE TYPES
# ============================================================================

class StructureType(Enum):
    TOWER = "Tower"
    BRIDGE = "Bridge" 
    WALL = "Wall"
    GANTRY = "Gantry"
    PLATFORM = "Platform"
    CANTILEVER = "Cantilever"

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
    exposure_category: str = "B"
    terrain_factor: float = 1.0
    height_factor: float = 1.0
    
    def get_design_pressure(self, height_m: float) -> float:
        base_pressure = 0.6 * (self.design_speed_ms ** 2) / 1000
        height_factor = min(1.2, 1.0 + (height_m - 10) * 0.02)
        return base_pressure * self.terrain_factor * height_factor

@dataclass
class StructureParams:
    # Structure identification
    structure_type: StructureType = StructureType.TOWER
    system: SystemType = SystemType.RINGLOCK
    structure_name: str = "Structure"
    structure_count: int = 1
    
    # Basic dimensions
    width_m: float = 4.0
    depth_m: float = 4.0
    height_m: float = 12.0
    length_m: float = 0.0  # For linear structures (walls, bridges)
    
    # Grid and spacing
    bay_m: float = 2.0
    lift_m: float = 2.0
    
    # Structure-specific parameters
    spans: List[float] = field(default_factory=list)  # For bridges
    wall_thickness_m: float = 2.0  # For walls
    cantilever_length_m: float = 0.0  # For cantilevers
    
    # Loading and access
    load_category: LoadCategory = LoadCategory.MEDIUM
    platform_levels: List[int] = field(default_factory=lambda: [6])
    access_levels: List[int] = field(default_factory=lambda: [6])
    working_levels: List[int] = field(default_factory=lambda: [6])
    
    # Environmental
    cladding: CladdingType = CladdingType.NETTING
    wind_loading: WindLoading = field(default_factory=WindLoading)
    exposure_class: str = "External"
    
    # Foundation and ballast
    base_condition: str = "Level concrete"
    ballast_750kg: int = 0
    ballast_1000kg: int = 0
    barrier_1493kg: int = 0
    
    # Special features
    has_stairs: bool = True
    has_hoists: bool = False
    has_weather_protection: bool = False
    tied_structure: bool = False
    
    # Engineering
    engineering_approved: bool = False
    notes: str = ""

    @property
    def lifts(self) -> int:
        return max(1, int(math.ceil(self.height_m / self.lift_m)))
    
    @property 
    def total_area_m2(self) -> float:
        if self.structure_type == StructureType.WALL:
            return self.length_m * self.wall_thickness_m * self.structure_count
        elif self.structure_type == StructureType.BRIDGE:
            return sum(self.spans) * self.width_m if self.spans else self.length_m * self.width_m
        else:  # Tower, Platform, etc.
            return self.width_m * self.depth_m * self.structure_count

@dataclass
class BOMRules:
    # System efficiency factors
    ringlock_efficiency: float = 1.0
    cuplok_efficiency: float = 1.1
    hybrid_complexity_factor: float = 1.2
    
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
    
    # Structure-specific factors
    bridge_spanning_factor: float = 1.3  # Extra beams for spanning
    wall_continuity_factor: float = 0.9  # Less material per linear meter
    cantilever_support_factor: float = 1.8  # Heavy back-propping required

# ============================================================================
# ADVANCED DRAWING ANALYSIS FOR MULTIPLE STRUCTURE TYPES
# ============================================================================

class MultiStructureAnalyzer:
    def __init__(self):
        self.patterns = {
            'structure_info': [
                # Towers
                re.compile(r"Tower\s+([A-Za-z0-9\-\s]+)\s*\((\d+)\s*nos?\.?\)", re.IGNORECASE),
                # Bridges 
                re.compile(r"([A-Za-z0-9\-\s]*Bridge)", re.IGNORECASE),
                re.compile(r"Cable\s+Bridge", re.IGNORECASE),
                # Walls
                re.compile(r"Wall", re.IGNORECASE),
                re.compile(r"Linear\s+Structure", re.IGNORECASE),
                # Generic
                re.compile(r"Structure\s*:\s*([A-Za-z0-9\-\s]+)", re.IGNORECASE),
            ],
            'structure_type_keywords': {
                StructureType.TOWER: ['tower', 'vertical', 'platform'],
                StructureType.BRIDGE: ['bridge', 'cable', 'span', 'crossing'],
                StructureType.WALL: ['wall', 'linear', 'barrier', 'screen'],
                StructureType.GANTRY: ['gantry', 'frame', 'portal'],
                StructureType.PLATFORM: ['platform', 'deck', 'stage'],
                StructureType.CANTILEVER: ['cantilever', 'overhang', 'projection']
            },
            'dimensions': [
                # Standard dimension patterns
                re.compile(r"(\d{3,5})\s*(?:mm|MM)", re.IGNORECASE),
                re.compile(r"(\d+\.?\d*)\s*(?:m|M)\s*[x×]\s*(\d+\.?\d*)\s*(?:m|M)", re.IGNORECASE),
                
                # Bridge-specific patterns
                re.compile(r"(\d{4,5})\s*(?:total|span|length)", re.IGNORECASE),
                re.compile(r"spans?\s*[:=]\s*([\d\s,+-]+)", re.IGNORECASE),
                
                # Wall-specific patterns
                re.compile(r"(\d{5})\s*(?:length|wide)", re.IGNORECASE),
                re.compile(r"linear\s*[\w\s]*(\d{4,5})", re.IGNORECASE),
            ],
            'span_patterns': [
                # Individual spans like "1300 5140 1300 7390 1300"
                re.compile(r"(\d{4})\s+(\d{4})\s+(\d{4})\s+(\d{4})\s+(\d{4})", re.IGNORECASE),
                re.compile(r"(\d{4})\s*[-–+]\s*(\d{4})\s*[-–+]\s*(\d{4})", re.IGNORECASE),
            ],
            'ballast': [
                re.compile(r"(\d+)\s*(?:kg|KG).*ballast", re.IGNORECASE),
                re.compile(r"ballast.*(\d+)\s*(?:kg|KG)", re.IGNORECASE),
                re.compile(r"barrier.*(\d{4})\s*(?:kg|KG)", re.IGNORECASE),
                re.compile(r"(\d{4})\s*kg", re.IGNORECASE),
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
        """Enhanced text extraction"""
        text = ""
        
        if pdfplumber is not None:
            try:
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                            
                            # Also try to extract from tables
                            tables = page.extract_tables()
                            for table in tables or []:
                                for row in table:
                                    text += " ".join([str(cell) for cell in row if cell]) + "\n"
                                    
            except Exception as e:
                st.warning(f"PDF text extraction failed: {e}")
        
        # OCR fallback 
        if not text.strip() and convert_from_bytes and pytesseract:
            try:
                images = convert_from_bytes(file_bytes, dpi=300)
                for img in images:
                    if Image:
                        if img.mode != 'L':
                            img = img.convert('L')
                        enhancer = ImageEnhance.Contrast(img)
                        img = enhancer.enhance(1.5)
                    text += pytesseract.image_to_string(img, config='--psm 6') + "\n"
            except Exception as e:
                st.warning(f"OCR extraction failed: {e}")
        
        return text
    
    def analyze_drawing(self, text: str) -> Dict[str, Any]:
        """Comprehensive multi-structure analysis"""
        results = {
            'structure_type': StructureType.TOWER,  # Default
            'structure_name': None,
            'structure_count': 1,
            'system_type': SystemType.RINGLOCK,
            'dimensions': {
                'width_m': None, 
                'depth_m': None, 
                'height_m': None, 
                'length_m': None,
                'total_length_m': None
            },
            'spans': [],
            'bay_spacing': None,
            'ballast_items': [],
            'cladding_type': CladdingType.NETTING,
            'confidence_score': 0.0,
            'detection_details': []
        }
        
        confidence_points = 0
        max_points = 15
        detection_details = []
        
        # 1. DETECT STRUCTURE TYPE
        structure_type = self._detect_structure_type(text)
        if structure_type:
            results['structure_type'] = structure_type
            confidence_points += 3
            detection_details.append(f"Detected structure type: {structure_type.value}")
        
        # 2. EXTRACT STRUCTURE NAME AND COUNT
        name_info = self._extract_structure_name(text, structure_type)
        if name_info['name']:
            results['structure_name'] = name_info['name']
            confidence_points += 2
            detection_details.append(f"Structure name: {name_info['name']}")
        if name_info['count']:
            results['structure_count'] = name_info['count']
            confidence_points += 1
            detection_details.append(f"Count: {name_info['count']}")
        
        # 3. EXTRACT DIMENSIONS BASED ON STRUCTURE TYPE
        if structure_type == StructureType.BRIDGE:
            dims, spans = self._extract_bridge_dimensions(text)
            results['dimensions'].update(dims)
            results['spans'] = spans
            if dims or spans:
                confidence_points += 4
                detection_details.append(f"Bridge dims: {dims}, spans: {len(spans)}")
        
        elif structure_type == StructureType.WALL:
            dims = self._extract_wall_dimensions(text)
            results['dimensions'].update(dims)
            if dims:
                confidence_points += 3
                detection_details.append(f"Wall dims: {dims}")
        
        else:  # Tower and other vertical structures
            dims = self._extract_tower_dimensions(text)
            results['dimensions'].update(dims)
            if dims:
                confidence_points += 2
                detection_details.append(f"Tower dims: {dims}")
        
        # 4. EXTRACT SYSTEM TYPE
        system_type = self._extract_system_type(text)
        if system_type:
            results['system_type'] = system_type
            confidence_points += 1
            detection_details.append(f"System: {system_type.value}")
        
        # 5. EXTRACT BALLAST INFORMATION
        ballast_items = self._extract_ballast_info(text)
        if ballast_items:
            results['ballast_items'] = ballast_items
            confidence_points += 2
            detection_details.append(f"Ballast items: {len(ballast_items)}")
        
        # 6. EXTRACT CLADDING
        cladding = self._extract_cladding_type(text)
        if cladding:
            results['cladding_type'] = cladding
            confidence_points += 1
        
        results['confidence_score'] = confidence_points / max_points
        results['detection_details'] = detection_details
        
        return results
    
    def _detect_structure_type(self, text: str) -> Optional[StructureType]:
        """Detect the primary structure type"""
        text_lower = text.lower()
        
        # Score each structure type based on keyword frequency
        scores = {}
        for struct_type, keywords in self.patterns['structure_type_keywords'].items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            scores[struct_type] = score
        
        # Additional specific patterns
        if re.search(r"cable\s*bridge", text_lower):
            scores[StructureType.BRIDGE] = scores.get(StructureType.BRIDGE, 0) + 5
        if re.search(r"tower.*\(\d+\s*nos?\.?\)", text_lower):
            scores[StructureType.TOWER] = scores.get(StructureType.TOWER, 0) + 5
        if re.search(r"\d{5}.*(?:wall|linear)", text_lower):  # Large numbers suggest wall length
            scores[StructureType.WALL] = scores.get(StructureType.WALL, 0) + 3
        
        # Return highest scoring type if above threshold
        if scores:
            best_type = max(scores, key=scores.get)
            if scores[best_type] > 0:
                return best_type
        
        return StructureType.TOWER  # Default
    
    def _extract_structure_name(self, text: str, structure_type: StructureType) -> Dict[str, Any]:
        """Extract structure name and count"""
        result = {'name': None, 'count': 1}
        
        for pattern in self.patterns['structure_info']:
            match = pattern.search(text)
            if match:
                if len(match.groups()) >= 2:
                    result['name'] = match.group(1).strip()
                    try:
                        result['count'] = int(match.group(2))
                    except (ValueError, IndexError):
                        pass
                else:
                    result['name'] = match.group(1).strip()
                break
        
        # Fallback: look for structure type name in text
        if not result['name']:
            type_match = re.search(rf"({structure_type.value})", text, re.IGNORECASE)
            if type_match:
                result['name'] = type_match.group(1)
        
        return result
    
    def _extract_bridge_dimensions(self, text: str) -> Tuple[Dict[str, float], List[float]]:
        """Extract bridge-specific dimensions and spans"""
        dims = {}
        spans = []
        
        # Look for total length (5-6 digit numbers)
        total_length_matches = re.findall(r"(\d{5,6})", text)
        if total_length_matches:
            # Take the largest as likely total length
            total_length_mm = max(int(x) for x in total_length_matches)
            dims['total_length_m'] = total_length_mm / 1000
            dims['length_m'] = total_length_mm / 1000
        
        # Look for individual spans (4-digit patterns)
        span_candidates = re.findall(r"(\d{4})", text)
        if span_candidates:
            # Filter to reasonable span ranges (1-10m when converted)
            valid_spans = []
            for span_str in span_candidates:
                span_mm = int(span_str)
                if 1000 <= span_mm <= 10000:  # 1m to 10m spans
                    valid_spans.append(span_mm / 1000)
            
            # Remove duplicates and sort
            spans = sorted(list(set(valid_spans)))
            
            # If we have multiple spans, use them
            if len(spans) >= 3:
                dims['width_m'] = max(spans)  # Longest span as width
            
        # Look for height (typically 2-4m for bridges)
        height_candidates = [int(x) for x in re.findall(r"(\d{4})", text) 
                           if 1500 <= int(x) <= 6000]  # 1.5-6m height range
        if height_candidates:
            dims['height_m'] = max(height_candidates) / 1000
        
        return dims, spans
    
    def _extract_wall_dimensions(self, text: str) -> Dict[str, float]:
        """Extract wall-specific dimensions"""
        dims = {}
        
        # Look for large dimensions (walls are typically long)
        length_matches = re.findall(r"(\d{5})", text)  # 5-digit numbers for length
        if length_matches:
            length_mm = max(int(x) for x in length_matches)
            dims['length_m'] = length_mm / 1000
            dims['width_m'] = length_mm / 1000  # Use length as width for BOM calc
        
        # Look for height
        height_candidates = [int(x) for x in re.findall(r"(\d{4})", text)
                           if 3000 <= int(x) <= 15000]  # 3-15m height range
        if height_candidates:
            dims['height_m'] = max(height_candidates) / 1000
        
        # Look for thickness/depth (typically 2-4m)
        thickness_candidates = [int(x) for x in re.findall(r"(\d{4})", text)
                              if 1500 <= int(x) <= 4000]  # 1.5-4m depth
        if thickness_candidates:
            dims['depth_m'] = min(thickness_candidates) / 1000  # Smallest reasonable thickness
        
        return dims
    
    def _extract_tower_dimensions(self, text: str) -> Dict[str, float]:
        """Extract tower-specific dimensions"""
        dims = {}
        
        # Look for plan dimensions (WxD)
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
        height_candidates = [int(x) for x in re.findall(r"(\d{4,5})", text)]
        if height_candidates:
            # For towers, look for the largest reasonable height
            for height_mm in sorted(height_candidates, reverse=True):
                if 3000 <= height_mm <= 50000:  # 3-50m range
                    dims['height_m'] = height_mm / 1000
                    break
        
        return dims
    
    def _extract_system_type(self, text: str) -> Optional[SystemType]:
        """Extract scaffolding system type"""
        for pattern in self.patterns['system']:
            match = pattern.search(text)
            if match:
                system_text = match.group(1).lower()
                if 'ringlock' in system_text:
                    return SystemType.RINGLOCK
                elif 'cuplok' in system_text:
                    return SystemType.CUPLOK
                elif 'lion' in system_text:
                    return SystemType.LION_DECK
        return SystemType.RINGLOCK  # Default
    
    def _extract_ballast_info(self, text: str) -> List[Dict[str, Any]]:
        """Extract ballast information with counting"""
        ballast_items = []
        
        # Count occurrences of specific weights
        ballast_weights = [750, 1000, 1493]
        for weight in ballast_weights:
            count = len(re.findall(rf"{weight}\s*kg", text, re.IGNORECASE))
            if count > 0:
                ballast_items.append({
                    'type': f"Ballast {weight}kg",
                    'weight_kg': weight,
                    'quantity': count
                })
        
        return ballast_items
    
    def _extract_cladding_type(self, text: str) -> Optional[CladdingType]:
        """Extract cladding type"""
        for pattern in self.patterns['cladding']:
            match = pattern.search(text)
            if match:
                cladding_text = match.group(1).lower()
                if 'netting' in cladding_text:
                    return CladdingType.NETTING
                elif 'sheet' in cladding_text:
                    return CladdingType.SHEETING
                elif 'mesh' in cladding_text:
                    return CladdingType.MESH
                elif 'panel' in cladding_text:
                    return CladdingType.SOLID_PANELS
        return CladdingType.NETTING  # Default

# ============================================================================
# MULTI-STRUCTURE BOM CALCULATION ENGINE
# ============================================================================

class MultiStructureBOMCalculator:
    def __init__(self, rules: BOMRules):
        self.rules = rules
    
    def calculate_comprehensive_bom(self, params: StructureParams) -> pd.DataFrame:
        """Calculate BOM based on structure type"""
        if params.structure_type == StructureType.TOWER:
            return self._calculate_tower_bom(params)
        elif params.structure_type == StructureType.BRIDGE:
            return self._calculate_bridge_bom(params)
        elif params.structure_type == StructureType.WALL:
            return self._calculate_wall_bom(params)
        elif params.structure_type == StructureType.GANTRY:
            return self._calculate_gantry_bom(params)
        elif params.structure_type == StructureType.PLATFORM:
            return self._calculate_platform_bom(params)
        elif params.structure_type == StructureType.CANTILEVER:
            return self._calculate_cantilever_bom(params)
        else:
            return self._calculate_tower_bom(params)  # Fallback
    
    def _calculate_tower_bom(self, params: StructureParams) -> pd.DataFrame:
        """Enhanced tower BOM calculation"""
        bays_x = max(1, int(round(params.width_m / params.bay_m)))
        bays_y = max(1, int(round(params.depth_m / params.bay_m)))
        posts = (bays_x + 1) * (bays_y + 1)
        lifts = params.lifts
        
        components = []
        
        # Standards 
        standards_qty = posts * lifts
        components.append((f"{params.system.value} STANDARD 2.0m", standards_qty))
        
        # Ledgers
        ledgers_x = bays_x * (bays_y + 1) * lifts
        ledgers_y = bays_y * (bays_x + 1) * lifts
        total_ledgers = ledgers_x + ledgers_y
        components.append((f"{params.system.value} LEDGER 2.0m", total_ledgers))
        
        # Bracing with wind consideration
        wind_pressure = params.wind_loading.get_design_pressure(params.height_m)
        bracing_factor = min(1.5, 1.0 + wind_pressure * 0.1)
        
        # Plan braces
        plan_levels = math.ceil(lifts / 2) if self.rules.plan_brace_alt_lifts else lifts
        plan_braces = int((bays_x * bays_y * plan_levels) * bracing_factor)
        components.append((f"{params.system.value} PLAN BRACE 2.0m x 2.0m", plan_braces))
        
        # Diagonal braces
        diag_levels = math.ceil(lifts / 2) if self.rules.diag_brace_alt_lifts else lifts
        perimeter_bays = 2 * (bays_x + bays_y)
        diag_braces = int((perimeter_bays * diag_levels) * bracing_factor)
        components.append((f"{params.system.value} DIAGONAL BRACE 2.0m x 2.0m", diag_braces))
        
        # Base components
        components.append((f"{params.system.value} BASE PLATE", posts))
        components.append((f"{params.system.value} ADJUSTABLE JACK LONG", posts))
        
        # Platform and access components
        self._add_platform_components(components, params)
        self._add_access_components(components, params)
        self._add_ballast_components(components, params)
        self._add_tie_components(components, params)
        self._add_cladding_components(components, params)
        
        return self._create_bom_dataframe(components, params)
    
    def _calculate_bridge_bom(self, params: StructureParams) -> pd.DataFrame:
        """Bridge-specific BOM calculation"""
        components = []
        
        # Determine number of support points
        if params.spans:
            support_points = len(params.spans) + 1
            total_span = sum(params.spans)
        else:
            support_points = max(2, int(params.length_m / 8))  # Support every ~8m
            total_span = params.length_m
        
        # Main spanning beams - heavy duty for bridge loads
        primary_beams = support_points - 1  # One beam per span
        components.append((f"{params.system.value} PRIMARY BEAM 8.0m", primary_beams))
        
        # Secondary beams for deck support
        secondary_beam_spacing = 2.0  # Every 2m along bridge
        secondary_beams = int(total_span / secondary_beam_spacing) + 1
        components.append((f"{params.system.value} SECONDARY BEAM 4.0m", secondary_beams))
        
        # Support towers at each support point
        tower_height_lifts = max(2, int(params.height_m / params.lift_m))
        support_standards = support_points * tower_height_lifts * 4  # 4 posts per support
        components.append((f"{params.system.value} STANDARD 2.0m", support_standards))
        
        # Horizontal bracing along bridge length
        horizontal_braces = int(total_span / 4) * 2  # Top and bottom chord bracing
        components.append((f"{params.system.value} HORIZONTAL BRACE 4.0m", horizontal_braces))
        
        # Vertical/diagonal bracing at supports
        vertical_braces = support_points * 4  # Cross bracing at each support
        components.append((f"{params.system.value} DIAGONAL BRACE 3.0m x 3.0m", vertical_braces))
        
        # Cable supports and tensioning (if cable bridge)
        if "cable" in params.structure_name.lower():
            cable_length = total_span * 2  # Main cables
            components.append((f"BRIDGE CABLE ø20mm (per meter)", int(cable_length)))
            components.append((f"CABLE ANCHOR POINTS", support_points * 2))
            components.append((f"CABLE TENSIONING DEVICES", support_points))
        
        # Bridge deck
        deck_area = total_span * params.width_m
        deck_panels = math.ceil(deck_area / 8)  # 8m² per panel
        components.append((f"BRIDGE DECKING PANEL 4m x 2m", deck_panels))
        
        # Guardrails
        guardrail_length = total_span * 2  # Both sides
        guardrail_sections = math.ceil(guardrail_length / 2.0)
        components.append((f"BRIDGE GUARDRAIL 2.0m", guardrail_sections))
        
        # Foundation/base components
        base_plates = support_points * 4
        components.append((f"{params.system.value} BASE PLATE", base_plates))
        components.append((f"{params.system.value} ADJUSTABLE JACK LONG", base_plates))
        
        # Apply bridge spanning factor
        components = [(desc, int(qty * self.rules.bridge_spanning_factor)) 
                     for desc, qty in components]
        
        self._add_ballast_components(components, params)
        
        return self._create_bom_dataframe(components, params)
    
    def _calculate_wall_bom(self, params: StructureParams) -> pd.DataFrame:
        """Wall-specific BOM calculation"""
        components = []
        
        # Wall is essentially a linear structure
        wall_length = params.length_m or params.width_m
        wall_bays = max(1, int(wall_length / params.bay_m))
        posts_per_lift = wall_bays + 1
        lifts = params.lifts
        
        # Standards - linear arrangement
        total_standards = posts_per_lift * lifts
        components.append((f"{params.system.value} STANDARD 2.0m", total_standards))
        
        # Ledgers - run continuously along wall
        ledgers_per_lift = wall_bays  # Horizontal ledgers
        if params.wall_thickness_m > 2.0:  # If thick wall, add cross ledgers
            ledgers_per_lift += wall_bays  # Cross braces
        total_ledgers = ledgers_per_lift * lifts
        components.append((f"{params.system.value} LEDGER 2.0m", total_ledgers))
        
        # Bracing - focus on stability along length
        # Diagonal bracing every 8m or so
        brace_sets = max(2, wall_bays // 4)
        brace_lifts = math.ceil(lifts / 2) if self.rules.diag_brace_alt_lifts else lifts
        diagonal_braces = brace_sets * brace_lifts
        components.append((f"{params.system.value} DIAGONAL BRACE 4.0m x 2.0m", diagonal_braces))
        
        # Plan bracing for wall stability (wind loading)
        plan_braces = brace_sets * brace_lifts
        components.append((f"{params.system.value} PLAN BRACE 2.0m x 2.0m", plan_braces))
        
        # Base components
        base_plates = posts_per_lift
        components.append((f"{params.system.value} BASE PLATE", base_plates))
        components.append((f"{params.system.value} ADJUSTABLE JACK LONG", base_plates))
        
        # Working platforms on wall (if specified)
        if params.working_levels:
            platform_area = wall_length * params.wall_thickness_m
            deck_panels = math.ceil(platform_area / 8)
            components.append((f"WALL PLATFORM DECKING 4m x 2m", deck_panels))
            
            # Guardrails for working levels
            guardrail_length = wall_length * 2  # Both sides
            guardrail_sections = math.ceil(guardrail_length / 2.0)
            components.append((f"GUARDRAIL 2.0m", guardrail_sections))
        
        # Apply wall continuity factor (walls use less material per linear meter)
        components = [(desc, max(1, int(qty * self.rules.wall_continuity_factor)))
                     for desc, qty in components]
        
        # Ballast - critical for wall stability
        self._add_ballast_components(components, params)
        
        return self._create_bom_dataframe(components, params)
    
    def _calculate_gantry_bom(self, params: StructureParams) -> pd.DataFrame:
        """Gantry/portal frame BOM calculation"""
        components = []
        
        # Gantry is like a bridge but with vertical supports
        span = params.width_m
        height = params.height_m
        
        # Main frame beams
        horizontal_beams = 2  # Top and bottom chord
        components.append((f"{params.system.value} PRIMARY BEAM 8.0m", horizontal_beams))
        
        # Vertical posts
        posts_per_side = max(2, int(height / params.lift_m))
        total_posts = posts_per_side * 2  # Two sides
        components.append((f"{params.system.value} STANDARD 2.0m", total_posts))
        
        # Cross bracing
        cross_braces = posts_per_side
        components.append((f"{params.system.value} DIAGONAL BRACE 4.0m x 4.0m", cross_braces))
        
        # Base components
        components.append((f"{params.system.value} BASE PLATE", 4))  # Four corners
        components.append((f"{params.system.value} ADJUSTABLE JACK LONG", 4))
        
        return self._create_bom_dataframe(components, params)
    
    def _calculate_platform_bom(self, params: StructureParams) -> pd.DataFrame:
        """Platform-specific BOM calculation"""
        components = []
        
        # Platform is like a low tower with emphasis on decking
        bays_x = max(1, int(round(params.width_m / params.bay_m)))
        bays_y = max(1, int(round(params.depth_m / params.bay_m)))
        posts = (bays_x + 1) * (bays_y + 1)
        
        # Minimal height structure
        lifts = max(1, int(params.height_m / params.lift_m))
        
        # Standards
        components.append((f"{params.system.value} STANDARD 1.5m", posts * lifts))
        
        # Ledgers
        ledgers_x = bays_x * (bays_y + 1) * lifts
        ledgers_y = bays_y * (bays_x + 1) * lifts
        components.append((f"{params.system.value} LEDGER 2.0m", ledgers_x + ledgers_y))
        
        # Heavy-duty decking
        platform_area = params.width_m * params.depth_m
        deck_panels = math.ceil(platform_area / 4)  # Smaller panels for platforms
        components.append((f"PLATFORM DECK 2m x 2m", deck_panels))
        
        # Guardrails
        perimeter = 2 * (params.width_m + params.depth_m)
        guardrail_sections = math.ceil(perimeter / 2.0)
        components.append((f"GUARDRAIL 2.0m", guardrail_sections))
        
        # Base components
        components.append((f"{params.system.value} BASE PLATE", posts))
        components.append((f"{params.system.value} ADJUSTABLE JACK LONG", posts))
        
        return self._create_bom_dataframe(components, params)
    
    def _calculate_cantilever_bom(self, params: StructureParams) -> pd.DataFrame:
        """Cantilever-specific BOM calculation"""
        components = []
        
        # Cantilevers require heavy back-propping
        cantilever_length = params.cantilever_length_m or params.width_m / 2
        
        # Main cantilever beam
        cantilever_beams = max(2, int(params.depth_m / 2))  # Beams across width
        components.append((f"{params.system.value} CANTILEVER BEAM 6.0m", cantilever_beams))
        
        # Back-propping (critical for cantilevers)
        backprop_posts = cantilever_beams * 2  # Two rows of backprops
        backprop_lifts = max(2, int(params.height_m / params.lift_m))
        total_backprops = backprop_posts * backprop_lifts
        components.append((f"{params.system.value} STANDARD 2.0m", total_backprops))
        
        # Heavy duty bracing
        brace_qty = int(cantilever_beams * backprop_lifts * 1.5)
        components.append((f"{params.system.value} DIAGONAL BRACE 3.0m x 2.0m", brace_qty))
        
        # Cantilever decking
        cantilever_area = cantilever_length * params.depth_m
        deck_panels = math.ceil(cantilever_area / 4)
        components.append((f"CANTILEVER DECK 2m x 2m", deck_panels))
        
        # Extra ballast for overturning resistance
        base_plates = backprop_posts
        components.append((f"{params.system.value} BASE PLATE", base_plates))
        components.append((f"{params.system.value} ADJUSTABLE JACK LONG", base_plates))
        
        # Apply cantilever support factor
        components = [(desc, int(qty * self.rules.cantilever_support_factor))
                     for desc, qty in components]
        
        return self._create_bom_dataframe(components, params)
    
    def _add_platform_components(self, components: List, params: StructureParams):
        """Add platform-specific components"""
        for level in params.platform_levels:
            platform_area = params.width_m * params.depth_m
            decking_panels = math.ceil(platform_area / 8)  # 8m² per panel
            guardrail_length = 2 * (params.width_m + params.depth_m)
            guardrail_sections = math.ceil(guardrail_length / 2.0)
            
            components.append((f"Steel Deck 4' x 4' (Level {level})", decking_panels))
            components.append((f"Guardrail 2.0m (Level {level})", guardrail_sections))
    
    def _add_access_components(self, components: List, params: StructureParams):
        """Add access components"""
        if params.has_stairs and params.access_levels:
            stair_flights = len(params.access_levels)
            components.append((f"Staircase 2.0m width", stair_flights))
            components.append((f"Landing Frame", stair_flights))
        
        if params.has_hoists:
            components.append((f"Hoist Tower Extension", 1))
            components.append((f"Hoist Beam and Supports", 1))
    
    def _add_ballast_components(self, components: List, params: StructureParams):
        """Add ballast components"""
        if params.barrier_1493kg > 0:
            components.append((f"Concrete Jersey Barrier 1493kg", params.barrier_1493kg))
        if params.ballast_1000kg > 0:
            components.append((f"Ballast 1000kg", params.ballast_1000kg))
        if params.ballast_750kg > 0:
            components.append((f"Ballast 750kg", params.ballast_750kg))
    
    def _add_tie_components(self, components: List, params: StructureParams):
        """Add tie components for tall structures"""
        if params.height_m > 8.0 or params.tied_structure:
            tie_levels = math.ceil(params.lifts / self.rules.tie_frequency_lifts)
            
            if params.structure_type == StructureType.WALL:
                # Wall ties every 8m along length
                wall_length = params.length_m or params.width_m
                tie_points = tie_levels * max(1, int(wall_length / 8))
            else:
                # Tower/other structure ties around perimeter
                perimeter_ties = 2 * (math.ceil(params.width_m / 4) + math.ceil(params.depth_m / 4))
                tie_points = tie_levels * perimeter_ties
            
            if tie_points > 0:
                components.append((f"Tie Rods with Anchors", tie_points))
    
    def _add_cladding_components(self, components: List, params: StructureParams):
        """Add cladding components"""
        if params.cladding != CladdingType.NONE:
            if params.structure_type == StructureType.WALL:
                cladding_area = (params.length_m or params.width_m) * params.height_m
            else:
                cladding_area = 2 * (params.width_m + params.depth_m) * params.height_m
            
            cladding_fixings = math.ceil(cladding_area * 2)
            components.append((f"Cladding Fixings ({params.cladding.value})", cladding_fixings))
    
    def _create_bom_dataframe(self, components: List, params: StructureParams) -> pd.DataFrame:
        """Create standardized BOM DataFrame"""
        df = pd.DataFrame(components, columns=["Product", "Qty per Structure"])
        df["Structures"] = params.structure_count
        df["Total (All Structures)"] = df["Qty per Structure"] * df["Structures"]
        return df

# ============================================================================
# ENHANCED ENGINEERING VALIDATION
# ============================================================================

class MultiStructureValidator:
    def __init__(self):
        self.limits = {
            StructureType.TOWER: {'max_height': 50.0, 'max_area': 400.0, 'min_ballast_ratio': 0.3},
            StructureType.BRIDGE: {'max_span': 30.0, 'max_total_length': 100.0, 'min_supports': 2},
            StructureType.WALL: {'max_length': 100.0, 'max_height': 20.0, 'min_thickness': 1.5},
            StructureType.GANTRY: {'max_span': 20.0, 'max_height': 15.0},
            StructureType.PLATFORM: {'max_area': 200.0, 'max_height': 8.0},
            StructureType.CANTILEVER: {'max_cantilever': 4.0, 'min_backprop_ratio': 2.0}
        }
    
    def validate_structure(self, params: StructureParams) -> Dict[str, Any]:
        """Structure-type-specific validation"""
        warnings = []
        errors = []
        recommendations = []
        
        struct_limits = self.limits.get(params.structure_type, {})
        
        # General validations
        if params.structure_type == StructureType.TOWER:
            self._validate_tower(params, struct_limits, errors, warnings, recommendations)
        elif params.structure_type == StructureType.BRIDGE:
            self._validate_bridge(params, struct_limits, errors, warnings, recommendations)
        elif params.structure_type == StructureType.WALL:
            self._validate_wall(params, struct_limits, errors, warnings, recommendations)
        
        # Wind loading assessment
        wind_pressure = params.wind_loading.get_design_pressure(params.height_m)
        if wind_pressure > 1.5:
            warnings.append(f"High wind pressure {wind_pressure:.2f} kN/m² - additional bracing recommended")
        
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
            'wind_pressure': wind_pressure
        }
    
    def _validate_tower(self, params, limits, errors, warnings, recommendations):
        """Tower-specific validation"""
        if params.height_m > limits.get('max_height', 50.0):
            errors.append(f"Tower height {params.height_m}m exceeds maximum {limits['max_height']}m")
        
        if params.total_area_m2 > limits.get('max_area', 400.0):
            warnings.append(f"Large tower area {params.total_area_m2:.1f}m² - verify foundation capacity")
        
        aspect_ratio = max(params.width_m, params.depth_m) / min(params.width_m, params.depth_m)
        if aspect_ratio > 3.0:
            warnings.append(f"High aspect ratio {aspect_ratio:.1f} may cause stability issues")
    
    def _validate_bridge(self, params, limits, errors, warnings, recommendations):
        """Bridge-specific validation"""
        total_span = sum(params.spans) if params.spans else params.length_m
        
        if total_span > limits.get('max_total_length', 100.0):
            errors.append(f"Bridge total length {total_span:.1f}m exceeds maximum {limits['max_total_length']}m")
        
        if params.spans:
            max_span = max(params.spans)
            if max_span > limits.get('max_span', 30.0):
                errors.append(f"Maximum span {max_span:.1f}m exceeds limit {limits['max_span']}m")
        
        support_count = len(params.spans) + 1 if params.spans else max(2, int(total_span / 10))
        if support_count < limits.get('min_supports', 2):
            errors.append(f"Insufficient supports - minimum {limits['min_supports']} required")
        
        recommendations.append("Bridge structures require specialized foundation design")
        recommendations.append("Consider deflection limits and dynamic loading")
    
    def _validate_wall(self, params, limits, errors, warnings, recommendations):
        """Wall-specific validation"""
        wall_length = params.length_m or params.width_m
        
        if wall_length > limits.get('max_length', 100.0):
            warnings.append(f"Wall length {wall_length:.1f}m is very long - consider expansion joints")
        
        if params.height_m > limits.get('max_height', 20.0):
            errors.append(f"Wall height {params.height_m}m exceeds maximum {limits['max_height']}m")
        
        if params.wall_thickness_m < limits.get('min_thickness', 1.5):
            warnings.append(f"Wall thickness {params.wall_thickness_m:.1f}m may be insufficient")
        
        # Wall-specific stability check
        height_to_thickness = params.height_m / params.wall_thickness_m
        if height_to_thickness > 6.0:
            warnings.append(f"High height/thickness ratio {height_to_thickness:.1f} - check stability")
        
        recommendations.append("Wall structures require adequate ballasting along length")
        recommendations.append("Consider wind loading perpendicular to wall face")

# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

st.set_page_config(
    page_title="Multi-Structure BOM Generator", 
    page_icon="🏗️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'current_params' not in st.session_state:
    st.session_state.current_params = StructureParams()
if 'bom_calculated' not in st.session_state:
    st.session_state.bom_calculated = False

# Main title
st.title("🏗️ Multi-Structure BOM Generator")
st.markdown("*Comprehensive scaffolding BOM generator for Towers, Bridges, Walls, and more*")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # BOM Rules
    with st.expander("📋 BOM Rules", expanded=False):
        plan_brace_alt = st.checkbox("Plan braces on alternate lifts", value=True)
        diag_brace_alt = st.checkbox("Diagonal braces on alternate lifts", value=True)
        tie_frequency = st.slider("Tie frequency (lifts)", 3, 8, 4)
        bridge_span_factor = st.slider("Bridge spanning factor", 1.0, 2.0, 1.3, 0.1)
        wall_continuity_factor = st.slider("Wall continuity factor", 0.7, 1.2, 0.9, 0.1)
        cantilever_support_factor = st.slider("Cantilever support factor", 1.2, 2.5, 1.8, 0.1)
    
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
        live_load_factor=live_load_factor,
        wind_load_factor=wind_load_factor,
        bridge_spanning_factor=bridge_span_factor,
        wall_continuity_factor=wall_continuity_factor,
        cantilever_support_factor=cantilever_support_factor
    )

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📄 Drawing Analysis", "⚙️ Parameters", "📊 BOM Generation", "🔍 Validation"])

# ============================================================================
# TAB 1: DRAWING ANALYSIS
# ============================================================================
with tab1:
    st.header("Multi-Structure Drawing Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload structure drawing/specification", 
            type=["pdf", "png", "jpg", "jpeg"],
            help="Supports towers, bridges, walls, gantries, platforms, and cantilevers"
        )
        
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            
            with st.spinner("Analyzing multi-structure drawing..."):
                analyzer = MultiStructureAnalyzer()
                text_content = analyzer.extract_text_from_pdf(file_bytes)
                analysis_results = analyzer.analyze_drawing(text_content)
                st.session_state.analysis_results = analysis_results
                
                # Update session state parameters based on analysis
                if analysis_results['structure_name']:
                    st.session_state.current_params.structure_name = analysis_results['structure_name']
                if analysis_results['structure_count']:
                    st.session_state.current_params.structure_count = analysis_results['structure_count']
                if analysis_results['structure_type']:
                    st.session_state.current_params.structure_type = analysis_results['structure_type']
                if analysis_results['system_type']:
                    st.session_state.current_params.system = analysis_results['system_type']
                
                # Update dimensions with validation
                dims = analysis_results['dimensions']
                if dims.get('width_m') and dims['width_m'] >= 1.0:
                    st.session_state.current_params.width_m = dims['width_m']
                if dims.get('depth_m') and dims['depth_m'] >= 1.0:
                    st.session_state.current_params.depth_m = dims['depth_m']
                if dims.get('height_m') and dims['height_m'] >= 0.5:
                    st.session_state.current_params.height_m = dims['height_m']
                if dims.get('length_m'):
                    st.session_state.current_params.length_m = dims['length_m']
                
                # Update spans for bridges
                if analysis_results['spans']:
                    st.session_state.current_params.spans = analysis_results['spans']
                
                # Update ballast info
                ballast_info = analysis_results.get('ballast_items', [])
                for item in ballast_info:
                    if item['weight_kg'] == 1493:
                        st.session_state.current_params.barrier_1493kg = max(
                            st.session_state.current_params.barrier_1493kg, 
                            item['quantity']
                        )
                
                confidence = analysis_results['confidence_score']
                if confidence > 0.5:
                    st.success(f"Analysis complete! Confidence: {confidence:.1%}")
                elif confidence > 0.2:
                    st.warning(f"Partial analysis. Confidence: {confidence:.1%}")
                else:
                    st.info(f"Limited detection. Confidence: {confidence:.1%} - Please verify parameters manually")
    
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
            
            # Display structure type
            st.write(f"**Structure Type:** {results['structure_type'].value}")
            
            # Display extracted parameters
            if results['structure_name']:
                st.write(f"**Name:** {results['structure_name']}")
            if results['structure_count'] > 1:
                st.write(f"**Quantity:** {results['structure_count']}")
            st.write(f"**System:** {results['system_type'].value}")
            
            # Display dimensions based on structure type
            dims = results['dimensions']
            if any(dims.values()):
                st.write("**Dimensions:**")
                if dims.get('width_m'):
                    st.write(f"• Width: {dims['width_m']:.1f}m")
                if dims.get('depth_m'):
                    st.write(f"• Depth: {dims['depth_m']:.1f}m") 
                if dims.get('height_m'):
                    st.write(f"• Height: {dims['height_m']:.1f}m")
                if dims.get('length_m'):
                    st.write(f"• Length: {dims['length_m']:.1f}m")
                if dims.get('total_length_m'):
                    st.write(f"• Total: {dims['total_length_m']:.1f}m")
            
            # Display spans for bridges
            if results['spans']:
                st.write(f"**Spans:** {len(results['spans'])} spans")
                span_text = ", ".join([f"{s:.1f}m" for s in results['spans'][:5]])
                if len(results['spans']) > 5:
                    span_text += "..."
                st.write(f"• {span_text}")
            
            # Display detection details
            if results.get('detection_details'):
                with st.expander("Detection Details"):
                    for detail in results['detection_details']:
                        st.write(f"• {detail}")

# ============================================================================
# TAB 2: PARAMETERS
# ============================================================================
with tab2:
    st.header("Structure Parameters")
    
    with st.form("parameter_form"):
        # Structure Type Selection
        st.subheader("🏗️ Structure Type & Basic Info")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            structure_type = st.selectbox(
                "Structure Type",
                options=[s.value for s in StructureType],
                index=[s.value for s in StructureType].index(st.session_state.current_params.structure_type.value)
            )
            st.session_state.current_params.structure_type = StructureType(structure_type)
        
        with col2:
            system_type = st.selectbox(
                "System Type",
                options=[s.value for s in SystemType],
                index=[s.value for s in SystemType].index(st.session_state.current_params.system.value)
            )
            st.session_state.current_params.system = SystemType(system_type)
        
        with col3:
            st.session_state.current_params.structure_name = st.text_input(
                "Structure Name", 
                value=st.session_state.current_params.structure_name
            )
        
        with col4:
            st.session_state.current_params.structure_count = st.number_input(
                "Number of Structures", 
                min_value=1, 
                value=st.session_state.current_params.structure_count
            )
        
        # Dimensions (adapt based on structure type)
        st.subheader("📐 Dimensions")
        
        if st.session_state.current_params.structure_type in [StructureType.TOWER, StructureType.PLATFORM, StructureType.GANTRY]:
            # Square/rectangular structures
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.session_state.current_params.width_m = st.number_input(
                    "Width (m)", min_value=1.0, value=st.session_state.current_params.width_m, step=0.5
                )
            with col2:
                st.session_state.current_params.depth_m = st.number_input(
                    "Depth (m)", min_value=1.0, value=st.session_state.current_params.depth_m, step=0.5
                )
            with col3:
                st.session_state.current_params.height_m = st.number_input(
                    "Height (m)", min_value=0.5, value=max(0.5, st.session_state.current_params.height_m), step=0.5
                )
            with col4:
                st.session_state.current_params.bay_m = st.slider(
                    "Bay Spacing (m)", min_value=1.0, max_value=3.0, value=st.session_state.current_params.bay_m, step=0.5
                )
        
        elif st.session_state.current_params.structure_type == StructureType.BRIDGE:
            # Linear structure with spans
            col1, col2, col3 = st.columns(3)
            with col1:
                st.session_state.current_params.length_m = st.number_input(
                    "Total Length (m)", min_value=5.0, value=max(5.0, st.session_state.current_params.length_m), step=0.5
                )
            with col2:
                st.session_state.current_params.width_m = st.number_input(
                    "Bridge Width (m)", min_value=1.0, value=st.session_state.current_params.width_m, step=0.5
                )
            with col3:
                st.session_state.current_params.height_m = st.number_input(
                    "Bridge Height (m)", min_value=0.5, value=max(0.5, st.session_state.current_params.height_m), step=0.5
                )
            
            # Spans input
            spans_text = st.text_input(
                "Individual Spans (m) - comma separated", 
                value=", ".join([f"{s:.1f}" for s in st.session_state.current_params.spans]) if st.session_state.current_params.spans else ""
            )
            if spans_text:
                try:
                    spans = [float(s.strip()) for s in spans_text.split(",") if s.strip()]
                    st.session_state.current_params.spans = spans
                except ValueError:
                    st.warning("Invalid span format - use numbers separated by commas")
        
        elif st.session_state.current_params.structure_type == StructureType.WALL:
            # Linear wall structure
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.session_state.current_params.length_m = st.number_input(
                    "Wall Length (m)", min_value=3.0, value=max(3.0, st.session_state.current_params.length_m), step=0.5
                )
            with col2:
                st.session_state.current_params.wall_thickness_m = st.number_input(
                    "Wall Thickness (m)", min_value=1.0, value=st.session_state.current_params.wall_thickness_m, step=0.5
                )
            with col3:
                st.session_state.current_params.height_m = st.number_input(
                    "Wall Height (m)", min_value=0.5, value=max(0.5, st.session_state.current_params.height_m), step=0.5
                )
            with col4:
                st.session_state.current_params.bay_m = st.slider(
                    "Bay Spacing (m)", min_value=1.0, max_value=3.0, value=st.session_state.current_params.bay_m, step=0.5
                )
        
        elif st.session_state.current_params.structure_type == StructureType.CANTILEVER:
            # Cantilever structure
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.session_state.current_params.width_m = st.number_input(
                    "Back Structure Width (m)", min_value=2.0, value=max(2.0, st.session_state.current_params.width_m), step=0.5
                )
            with col2:
                st.session_state.current_params.cantilever_length_m = st.number_input(
                    "Cantilever Length (m)", min_value=1.0, value=st.session_state.current_params.cantilever_length_m, step=0.5
                )
            with col3:
                st.session_state.current_params.depth_m = st.number_input(
                    "Structure Depth (m)", min_value=1.0, value=st.session_state.current_params.depth_m, step=0.5
                )
            with col4:
                st.session_state.current_params.height_m = st.number_input(
                    "Structure Height (m)", min_value=0.5, value=max(0.5, st.session_state.current_params.height_m), step=0.5
                )
        
        # Working Levels and Access (adapt to structure type)
        st.subheader("🚶 Working Levels & Access")
        col1, col2 = st.columns(2)
        
        with col1:
            max_lifts = max(1, int(st.session_state.current_params.height_m / st.session_state.current_params.lift_m))
            
            if st.session_state.current_params.structure_type in [StructureType.TOWER, StructureType.PLATFORM]:
                platform_levels = st.multiselect(
                    "Platform Levels (Lift Numbers)",
                    options=list(range(1, max_lifts + 1)),
                    default=[max_lifts] if max_lifts > 0 else [1]
                )
                st.session_state.current_params.platform_levels = platform_levels
            
            elif st.session_state.current_params.structure_type == StructureType.WALL:
                working_levels = st.multiselect(
                    "Working Levels (Lift Numbers)",
                    options=list(range(1, max_lifts + 1)),
                    default=[max_lifts] if max_lifts > 0 else [1]
                )
                st.session_state.current_params.working_levels = working_levels
        
        with col2:
            st.session_state.current_params.has_stairs = st.checkbox("Include Stairs", value=st.session_state.current_params.has_stairs)
            st.session_state.current_params.has_hoists = st.checkbox("Include Hoist Points", value=st.session_state.current_params.has_hoists)
        
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
            wind_speed = st.slider("Design Wind Speed (m/s)", min_value=15, max_value=50, value=25)
            st.session_state.current_params.wind_loading.design_speed_ms = wind_speed
        
        with col3:
            st.session_state.current_params.tied_structure = st.checkbox(
                "Tied Structure", 
                value=st.session_state.current_params.tied_structure,
                help="Structure is tied to existing building"
            )
        
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
                "750kg Ballast per Structure", min_value=0, value=st.session_state.current_params.ballast_750kg
            )
        
        with col3:
            st.session_state.current_params.ballast_1000kg = st.number_input(
                "1000kg Ballast per Structure", min_value=0, value=st.session_state.current_params.ballast_1000kg
            )
        
        with col4:
            st.session_state.current_params.barrier_1493kg = st.number_input(
                "1493kg Barriers per Structure", min_value=0, value=st.session_state.current_params.barrier_1493kg
            )
        
        # Notes
        st.session_state.current_params.notes = st.text_area(
            "Additional Notes", value=st.session_state.current_params.notes, height=100
        )
        
        # Form submission
        submitted = st.form_submit_button("🔄 Update Parameters", type="primary")
        
        if submitted:
            st.success(f"Parameters updated for {st.session_state.current_params.structure_type.value} structure!")
            st.session_state.bom_calculated = False

# ============================================================================
# TAB 3: BOM GENERATION
# ============================================================================
with tab3:
    st.header("Multi-Structure BOM Generation")
    
    # Show structure summary
    params = st.session_state.current_params
    st.info(f"**Structure:** {params.structure_type.value} - {params.structure_name} ({params.structure_count} units)")
    
    if st.button("🧮 Calculate BOM", type="primary", use_container_width=True):
        with st.spinner(f"Calculating {params.structure_type.value} BOM..."):
            try:
                calculator = MultiStructureBOMCalculator(bom_rules)
                bom_df = calculator.calculate_comprehensive_bom(params)
                
                # Apply extras
                bom_df["Extras %"] = extras_pct
                bom_df["Extras Qty"] = np.ceil(bom_df["Total (All Structures)"] * extras_pct / 100.0).astype(int)
                bom_df["Total for Delivery"] = bom_df["Total (All Structures)"] + bom_df["Extras Qty"]
                
                st.session_state.bom_df = bom_df
                st.session_state.bom_calculated = True
                
                st.success(f"{params.structure_type.value} BOM calculated successfully!")
                
            except Exception as e:
                st.error(f"Error calculating BOM: {str(e)}")
                st.info("Please check your parameters and try again.")
    
    if st.session_state.bom_calculated and 'bom_df' in st.session_state:
        bom_df = st.session_state.bom_df
        
        # Display BOM summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Structure Type", params.structure_type.value)
        with col2:
            st.metric("Total Line Items", len(bom_df))
        with col3:
            st.metric("Total Components", bom_df["Total for Delivery"].sum())
        with col4:
            estimated_weight = bom_df["Total for Delivery"].sum() * 5
            st.metric("Estimated Weight (kg)", f"{estimated_weight:,}")
        
        # Display BOM table
        st.subheader(f"📋 {params.structure_type.value} Bill of Materials")
        
        # Add filtering options
        col1, col2, col3 = st.columns(3)
        with col1:
            search_term = st.text_input("🔍 Search components:", placeholder="e.g., Standard, Ledger, Brace")
        with col2:
            min_qty = st.number_input("Min quantity filter:", min_value=0, value=0)
        with col3:
            system_filter = st.selectbox("System filter:", ["All"] + [s.value for s in SystemType])
        
        # Filter DataFrame
        filtered_df = bom_df.copy()
        if search_term:
            mask = filtered_df['Product'].str.contains(search_term, case=False, na=False)
            filtered_df = filtered_df[mask]
        if min_qty > 0:
            filtered_df = filtered_df[filtered_df['Total for Delivery'] >= min_qty]
        if system_filter != "All":
            mask = filtered_df['Product'].str.contains(system_filter, case=False, na=False)
            filtered_df = filtered_df[mask]
        
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
                file_name=f"BOM_{params.structure_type.value}_{params.structure_name.replace(' ', '_')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # JSON export
            json_data = {
                "project_info": {
                    "name": params.structure_name,
                    "type": params.structure_type.value,
                    "system": params.system.value,
                    "count": params.structure_count,
                    "generated": pd.Timestamp.now().isoformat()
                },
                "bom": filtered_df.to_dict('records')
            }
            
            st.download_button(
                "🔗 Download JSON",
                data=json.dumps(json_data, indent=2),
                file_name=f"BOM_{params.structure_type.value}_{params.structure_name.replace(' ', '_')}.json",
                mime="application/json"
            )
        
        with col3:
            # Text summary
            summary_text = f"""{params.structure_type.value}: {params.structure_name}
System: {params.system.value}
Count: {params.structure_count}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

Bill of Materials:
"""
            for _, row in filtered_df.iterrows():
                summary_text += f"{row['Product']}: {row['Total for Delivery']}\n"
            
            st.download_button(
                "📄 Download Summary",
                data=summary_text,
                file_name=f"Summary_{params.structure_type.value}_{params.structure_name.replace(' ', '_')}.txt",
                mime="text/plain"
            )

# ============================================================================
# TAB 4: VALIDATION
# ============================================================================
with tab4:
    st.header("Multi-Structure Engineering Validation")
    
    # Structure-specific validation
    st.subheader(f"🛡️ {st.session_state.current_params.structure_type.value} Validation")
    
    validator = MultiStructureValidator()
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
        total_issues = len(validation_results['errors']) + len(validation_results['warnings'])
        st.metric("Issues Found", total_issues)
    
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
    st.subheader(f"📊 {params.structure_type.value} Summary")
    params = st.session_state.current_params
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Structure Type", params.structure_type.value)
        st.metric("System Type", params.system.value)
        if params.structure_type == StructureType.BRIDGE and params.spans:
            st.metric("Number of Spans", len(params.spans))
        elif params.structure_type == StructureType.WALL:
            st.metric("Wall Length", f"{params.length_m:.1f} m")
        else:
            st.metric("Plan Area", f"{params.width_m * params.depth_m:.1f} m²")
    
    with col2:
        st.metric("Height", f"{params.height_m:.1f} m")
        st.metric("Bay Spacing", f"{params.bay_m:.1f} m")
        if hasattr(params, 'lifts'):
            st.metric("Number of Lifts", params.lifts)
        else:
            st.metric("Lift Height", f"{params.lift_m:.1f} m")
    
    with col3:
        st.metric("Load Category", params.load_category.value)
        st.metric("Cladding", params.cladding.value)
        total_ballast = (params.ballast_750kg * 750 + 
                        params.ballast_1000kg * 1000 + 
                        params.barrier_1493kg * 1493)
        st.metric("Total Ballast", f"{total_ballast:,} kg")
    
    # Structure-specific metrics
    if params.structure_type == StructureType.BRIDGE and params.spans:
        st.subheader("Bridge Span Details")
        spans_df = pd.DataFrame([
            {"Span": i+1, "Length (m)": span, "Support Load": f"{span * 2:.1f} kN"} 
            for i, span in enumerate(params.spans)
        ])
        st.dataframe(spans_df, use_container_width=True)
    
    # BOM analysis
    if st.session_state.bom_calculated and 'bom_df' in st.session_state:
        st.subheader("📈 BOM Analysis")
        bom_df = st.session_state.bom_df
        
        # Component category analysis based on structure type
        if params.structure_type == StructureType.TOWER:
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
        
        elif params.structure_type == StructureType.BRIDGE:
            beams = bom_df[bom_df['Product'].str.contains('BEAM', case=False, na=False)]['Total for Delivery'].sum()
            cables = bom_df[bom_df['Product'].str.contains('CABLE', case=False, na=False)]['Total for Delivery'].sum()
            deck = bom_df[bom_df['Product'].str.contains('DECK', case=False, na=False)]['Total for Delivery'].sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Beams", beams)
            with col2:
                st.metric("Cables", cables)
            with col3:
                st.metric("Decking", deck)
        
        elif params.structure_type == StructureType.WALL:
            standards = bom_df[bom_df['Product'].str.contains('STANDARD', case=False, na=False)]['Total for Delivery'].sum()
            ledgers = bom_df[bom_df['Product'].str.contains('LEDGER', case=False, na=False)]['Total for Delivery'].sum()
            ballast = bom_df[bom_df['Product'].str.contains('BALLAST|BARRIER', case=False, na=False)]['Total for Delivery'].sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Standards", standards)
            with col2:
                st.metric("Ledgers", ledgers)
            with col3:
                st.metric("Ballast Items", ballast)
        
        # Show top components
        st.subheader("Top Components by Quantity")
        top_components = bom_df.nlargest(10, 'Total for Delivery')[['Product', 'Total for Delivery']]
        st.dataframe(top_components, use_container_width=True)
    
    else:
        st.info("Generate BOM first to see detailed analysis.")

# Footer
st.divider()
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    <p>Multi-Structure BOM Generator v3.0 | Supports {", ".join([s.value for s in StructureType])} | 
    ⚠️ All calculations require verification by a competent person before use | 
    🔒 Engineering approval required for critical applications</p>
</div>
""", unsafe_allow_html=True)
