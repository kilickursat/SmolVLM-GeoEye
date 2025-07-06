#!/usr/bin/env python3
"""
SmolVLM-GeoEye Data Extraction Module
====================================

Advanced numerical data extraction from geotechnical documents.
Handles extraction of SPT values, bearing capacity, soil properties, etc.

Author: SmolVLM-GeoEye Team
Version: 3.1.0
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class ExtractedValue:
    """Represents an extracted numerical value with context"""
    value: float
    unit: str
    context: str
    confidence: float
    parameter_type: str
    depth: Optional[float] = None
    depth_unit: Optional[str] = None

class EnhancedGeotechnicalDataExtractor:
    """Enhanced data extraction for geotechnical parameters"""
    
    def __init__(self):
        # Enhanced patterns for various geotechnical parameters
        self.extraction_patterns = {
            'spt_values': [
                r'SPT[^\d]*N[\s-]*value[s]?[\s:=]*(\d+(?:\.\d+)?)',
                r'N[\s-]*value[s]?[\s:=]*(\d+(?:\.\d+)?)\s*(?:at|@)\s*(\d+(?:\.\d+)?)\s*(m|ft|meter|feet)',
                r'SPT[\s:=]*(\d+(?:\.\d+)?)\s*(?:at|@)\s*(\d+(?:\.\d+)?)\s*(m|ft)',
                r'N\s*=\s*(\d+(?:\.\d+)?)\s*(?:at|@)\s*(\d+(?:\.\d+)?)\s*(m|ft)',
                r'Standard\s+Penetration\s+Test[^\d]*(\d+(?:\.\d+)?)',
                r'blow\s*count[s]?[\s:=]*(\d+(?:\.\d+)?)',
            ],
            'bearing_capacity': [
                r'bearing\s+capacity[\s:=]*(\d+(?:\.\d+)?)\s*(kPa|MPa|psf|ksf|kN/m²|MN/m²)',
                r'allowable\s+bearing[\s:=]*(\d+(?:\.\d+)?)\s*(kPa|MPa|psf|ksf)',
                r'ultimate\s+bearing[\s:=]*(\d+(?:\.\d+)?)\s*(kPa|MPa|psf|ksf)',
                r'q[_]?allow[\s:=]*(\d+(?:\.\d+)?)\s*(kPa|MPa|psf|ksf)',
                r'q[_]?ult[\s:=]*(\d+(?:\.\d+)?)\s*(kPa|MPa|psf|ksf)',
                r'safe\s+bearing[\s:=]*(\d+(?:\.\d+)?)\s*(kPa|MPa|psf|ksf)',
            ],
            'density': [
                r'density[\s:=]*(\d+(?:\.\d+)?)\s*(g/cm³|kg/m³|pcf|g/cc)',
                r'unit\s+weight[\s:=]*(\d+(?:\.\d+)?)\s*(kN/m³|pcf|lb/ft³)',
                r'dry\s+density[\s:=]*(\d+(?:\.\d+)?)\s*(g/cm³|kg/m³|pcf)',
                r'bulk\s+density[\s:=]*(\d+(?:\.\d+)?)\s*(g/cm³|kg/m³|pcf)',
                r'γ[\s:=]*(\d+(?:\.\d+)?)\s*(kN/m³|pcf)',
                r'rho[\s:=]*(\d+(?:\.\d+)?)\s*(g/cm³|kg/m³)',
            ],
            'moisture_content': [
                r'moisture\s+content[\s:=]*(\d+(?:\.\d+)?)\s*(%|percent)',
                r'water\s+content[\s:=]*(\d+(?:\.\d+)?)\s*(%|percent)',
                r'w[\s:=]*(\d+(?:\.\d+)?)\s*%',
                r'MC[\s:=]*(\d+(?:\.\d+)?)\s*%',
            ],
            'cohesion': [
                r'cohesion[\s:=]*(\d+(?:\.\d+)?)\s*(kPa|MPa|psf|ksf)',
                r'c[\s:=]*(\d+(?:\.\d+)?)\s*(kPa|MPa|psf|ksf)',
                r'undrained\s+shear[\s:=]*(\d+(?:\.\d+)?)\s*(kPa|MPa|psf|ksf)',
                r'Su[\s:=]*(\d+(?:\.\d+)?)\s*(kPa|MPa|psf|ksf)',
            ],
            'friction_angle': [
                r'friction\s+angle[\s:=]*(\d+(?:\.\d+)?)\s*(°|deg|degree)',
                r'phi[\s:=]*(\d+(?:\.\d+)?)\s*(°|deg|degree)',
                r'φ[\s:=]*(\d+(?:\.\d+)?)\s*(°|deg|degree)',
                r'angle\s+of\s+friction[\s:=]*(\d+(?:\.\d+)?)\s*(°|deg|degree)',
            ],
            'settlement': [
                r'settlement[\s:=]*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|ft)',
                r'total\s+settlement[\s:=]*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|ft)',
                r'immediate\s+settlement[\s:=]*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|ft)',
                r'consolidation\s+settlement[\s:=]*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|ft)',
            ],
            'rqd': [
                r'RQD[\s:=]*(\d+(?:\.\d+)?)\s*%',
                r'Rock\s+Quality\s+Designation[\s:=]*(\d+(?:\.\d+)?)\s*%',
            ],
            'ucs': [
                r'UCS[\s:=]*(\d+(?:\.\d+)?)\s*(MPa|kPa|psi)',
                r'unconfined\s+compressive[\s:=]*(\d+(?:\.\d+)?)\s*(MPa|kPa|psi)',
                r'compressive\s+strength[\s:=]*(\d+(?:\.\d+)?)\s*(MPa|kPa|psi)',
            ],
            'plasticity_index': [
                r'PI[\s:=]*(\d+(?:\.\d+)?)',
                r'plasticity\s+index[\s:=]*(\d+(?:\.\d+)?)',
                r'Ip[\s:=]*(\d+(?:\.\d+)?)',
            ],
            'liquid_limit': [
                r'LL[\s:=]*(\d+(?:\.\d+)?)\s*%',
                r'liquid\s+limit[\s:=]*(\d+(?:\.\d+)?)\s*%',
                r'wL[\s:=]*(\d+(?:\.\d+)?)\s*%',
            ],
            'plastic_limit': [
                r'PL[\s:=]*(\d+(?:\.\d+)?)\s*%',
                r'plastic\s+limit[\s:=]*(\d+(?:\.\d+)?)\s*%',
                r'wP[\s:=]*(\d+(?:\.\d+)?)\s*%',
            ],
        }
        
    def extract_numerical_data_from_text(self, text: str) -> Dict[str, List[ExtractedValue]]:
        """Extract all numerical data from text"""
        if not text:
            return {}
        
        extracted_data = {}
        text_lower = text.lower()
        
        for param_type, patterns in self.extraction_patterns.items():
            values = []
            
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                
                for match in matches:
                    try:
                        # Extract value and unit
                        groups = match.groups()
                        if len(groups) >= 2:
                            value = float(groups[0])
                            unit = groups[1]
                            
                            # Check for depth information
                            depth = None
                            depth_unit = None
                            if len(groups) >= 3:
                                depth = float(groups[1])
                                depth_unit = groups[2]
                                unit = groups[2] if len(groups) == 3 else groups[1]
                            
                            # Get context (surrounding text)
                            start = max(0, match.start() - 50)
                            end = min(len(text), match.end() + 50)
                            context = text[start:end].strip()
                            
                            # Calculate confidence based on pattern specificity
                            confidence = self._calculate_confidence(pattern, context)
                            
                            extracted_value = ExtractedValue(
                                value=value,
                                unit=unit,
                                context=context,
                                confidence=confidence,
                                parameter_type=param_type,
                                depth=depth,
                                depth_unit=depth_unit
                            )
                            
                            # Avoid duplicates
                            if not self._is_duplicate(extracted_value, values):
                                values.append(extracted_value)
                        
                        elif len(groups) == 1:
                            # Handle single value patterns
                            value = float(groups[0])
                            unit = self._infer_unit(param_type)
                            
                            start = max(0, match.start() - 50)
                            end = min(len(text), match.end() + 50)
                            context = text[start:end].strip()
                            
                            confidence = self._calculate_confidence(pattern, context)
                            
                            extracted_value = ExtractedValue(
                                value=value,
                                unit=unit,
                                context=context,
                                confidence=confidence,
                                parameter_type=param_type
                            )
                            
                            if not self._is_duplicate(extracted_value, values):
                                values.append(extracted_value)
                    
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Error extracting {param_type}: {e}")
                        continue
            
            if values:
                extracted_data[param_type] = values
        
        return extracted_data
    
    def extract_from_structured_data(self, data: pd.DataFrame) -> Dict[str, List[ExtractedValue]]:
        """Extract numerical data from structured sources like CSV/Excel"""
        extracted_data = {}
        
        # Common column patterns for geotechnical data
        column_patterns = {
            'spt_values': ['spt', 'n-value', 'n_value', 'blow count', 'penetration'],
            'bearing_capacity': ['bearing', 'capacity', 'allowable', 'ultimate'],
            'density': ['density', 'unit weight', 'gamma'],
            'moisture_content': ['moisture', 'water content', 'mc', 'w%'],
            'cohesion': ['cohesion', 'c', 'su', 'undrained'],
            'friction_angle': ['friction', 'phi', 'angle'],
            'depth': ['depth', 'elevation', 'level'],
        }
        
        # Identify relevant columns
        for param_type, patterns in column_patterns.items():
            for col in data.columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in patterns):
                    # Extract values from this column
                    values = []
                    for idx, value in data[col].items():
                        try:
                            if pd.notna(value) and isinstance(value, (int, float)):
                                # Try to find associated depth
                                depth = None
                                depth_unit = 'm'
                                
                                for depth_col in data.columns:
                                    if 'depth' in depth_col.lower():
                                        depth_val = data.loc[idx, depth_col]
                                        if pd.notna(depth_val):
                                            depth = float(depth_val)
                                            break
                                
                                extracted_value = ExtractedValue(
                                    value=float(value),
                                    unit=self._infer_unit_from_column(col),
                                    context=f"Row {idx}: {col}",
                                    confidence=0.9,
                                    parameter_type=param_type,
                                    depth=depth,
                                    depth_unit=depth_unit if depth else None
                                )
                                values.append(extracted_value)
                        except:
                            continue
                    
                    if values:
                        if param_type not in extracted_data:
                            extracted_data[param_type] = []
                        extracted_data[param_type].extend(values)
        
        return extracted_data
    
    def _calculate_confidence(self, pattern: str, context: str) -> float:
        """Calculate confidence score for extracted value"""
        confidence = 0.5
        
        # Higher confidence for more specific patterns
        if 'at' in pattern or '@' in pattern:
            confidence += 0.2
        if any(keyword in context.lower() for keyword in ['test', 'result', 'measured', 'recorded']):
            confidence += 0.1
        if len(pattern) > 50:  # More complex patterns
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _infer_unit(self, param_type: str) -> str:
        """Infer unit based on parameter type"""
        unit_map = {
            'spt_values': 'blows/ft',
            'bearing_capacity': 'kPa',
            'density': 'g/cm³',
            'moisture_content': '%',
            'cohesion': 'kPa',
            'friction_angle': '°',
            'settlement': 'mm',
            'rqd': '%',
            'ucs': 'MPa',
            'plasticity_index': '',
            'liquid_limit': '%',
            'plastic_limit': '%',
        }
        return unit_map.get(param_type, '')
    
    def _infer_unit_from_column(self, column_name: str) -> str:
        """Infer unit from column name"""
        col_lower = column_name.lower()
        
        if 'kpa' in col_lower:
            return 'kPa'
        elif 'mpa' in col_lower:
            return 'MPa'
        elif '%' in col_lower or 'percent' in col_lower:
            return '%'
        elif 'deg' in col_lower or '°' in col_lower:
            return '°'
        elif 'g/cm' in col_lower:
            return 'g/cm³'
        elif 'mm' in col_lower:
            return 'mm'
        elif 'm' in col_lower and 'meter' in col_lower:
            return 'm'
        else:
            return ''
    
    def _is_duplicate(self, new_value: ExtractedValue, existing_values: List[ExtractedValue]) -> bool:
        """Check if value is duplicate"""
        for existing in existing_values:
            if (abs(existing.value - new_value.value) < 0.001 and 
                existing.unit == new_value.unit and
                abs(len(existing.context) - len(new_value.context)) < 10):
                return True
        return False
    
    def get_statistical_summary(self, extracted_data: Dict[str, List[ExtractedValue]]) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of extracted data"""
        summary = {}
        
        for param_type, values in extracted_data.items():
            if values:
                nums = [v.value for v in values]
                summary[param_type] = {
                    'count': len(nums),
                    'min': min(nums),
                    'max': max(nums),
                    'mean': np.mean(nums),
                    'std': np.std(nums),
                    'median': np.median(nums),
                }
        
        return summary
    
    def export_to_dataframe(self, extracted_data: Dict[str, List[ExtractedValue]]) -> pd.DataFrame:
        """Export extracted data to pandas DataFrame"""
        rows = []
        
        for param_type, values in extracted_data.items():
            for value in values:
                row = asdict(value)
                rows.append(row)
        
        return pd.DataFrame(rows)
