#!/usr/bin/env python3
"""
SmolVLM-GeoEye Engineering Agents Module
=======================================

Specialized AI agents for geotechnical engineering tasks.
Implements SmolAgent-based reasoning for domain-specific analysis.

Author: SmolVLM-GeoEye Team
Version: 3.1.0
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import json
import re
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class AgentResponse:
    """Standardized agent response structure"""
    agent_type: str
    query: str
    response: str
    confidence: float
    recommendations: List[str]
    warnings: List[str]
    data_used: Dict[str, Any]
    document_based: bool
    timestamp: str

class BaseGeotechnicalAgent(ABC):
    """Base class for all geotechnical agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.expertise_areas = []
        
    @abstractmethod
    def analyze(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        """Analyze query with given context"""
        pass
    
    def can_handle(self, query: str) -> float:
        """Return confidence score (0-1) for handling this query"""
        query_lower = query.lower()
        score = 0.0
        
        for area in self.expertise_areas:
            if area.lower() in query_lower:
                score += 0.3
        
        return min(score, 1.0)
    
    def search_document_content(self, query: str, documents: Dict[str, Any]) -> str:
        """Search through document content for relevant information"""
        query_lower = query.lower()
        relevant_sections = []
        
        for doc_name, doc_data in documents.items():
            if doc_data.get('document_type') == 'image' and 'content' in doc_data:
                ai_response = doc_data['content'].get('response', '')
                
                # Split response into paragraphs for better matching
                paragraphs = ai_response.split('\n\n')
                
                for paragraph in paragraphs:
                    paragraph_lower = paragraph.lower()
                    # Check if any word from query appears in paragraph
                    query_words = query_lower.split()
                    if any(word in paragraph_lower for word in query_words if len(word) > 3):
                        relevant_sections.append({
                            'source': doc_name,
                            'content': paragraph
                        })
            
            elif doc_data.get('document_type') == 'pdf' and 'content' in doc_data:
                pdf_text = doc_data['content'].get('text', '')
                # Search in PDF text
                paragraphs = pdf_text.split('\n\n')
                
                for paragraph in paragraphs[:20]:  # Limit to first 20 paragraphs
                    paragraph_lower = paragraph.lower()
                    query_words = query_lower.split()
                    if any(word in paragraph_lower for word in query_words if len(word) > 3):
                        relevant_sections.append({
                            'source': doc_name,
                            'content': paragraph[:500]  # Limit length
                        })
        
        return relevant_sections

class SoilAnalysisAgent(BaseGeotechnicalAgent):
    """Agent specialized in soil mechanics and analysis"""
    
    def __init__(self):
        super().__init__("Soil Analysis Expert")
        self.expertise_areas = [
            "soil", "spt", "bearing capacity", "density", "moisture",
            "classification", "grain size", "atterberg limits", "compaction",
            "permeability", "consolidation", "shear strength"
        ]
        
    def analyze(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        """Analyze soil-related queries"""
        from datetime import datetime
        
        response_text = ""
        recommendations = []
        warnings = []
        data_used = {}
        
        # First, search through document content for relevant information
        if context.get('processed_documents'):
            relevant_sections = self.search_document_content(query, context['processed_documents'])
            if relevant_sections:
                response_text = "Based on the uploaded documents:\n\n"
                for section in relevant_sections[:3]:  # Limit to top 3 relevant sections
                    response_text += f"**From {section['source']}:**\n{section['content']}\n\n"
        
        # Extract numerical data
        numerical_data = self._extract_numerical_data(context)
        
        # Add specific analysis based on query type
        if "spt" in query.lower() or "n-value" in query.lower():
            analysis, recs, warns = self._analyze_spt_data(numerical_data)
            if analysis and "No SPT data" not in analysis:
                response_text += "\n" + analysis
            recommendations.extend(recs)
            warnings.extend(warns)
            
        elif "bearing capacity" in query.lower():
            analysis, recs, warns = self._analyze_bearing_capacity(numerical_data)
            if analysis and "No bearing capacity data" not in analysis:
                response_text += "\n" + analysis
            recommendations.extend(recs)
            warnings.extend(warns)
            
        elif "classification" in query.lower():
            analysis = self._analyze_soil_classification(numerical_data)
            if analysis:
                response_text += "\n" + analysis
            
        elif "settlement" in query.lower():
            analysis, recs = self._analyze_settlement(numerical_data)
            if analysis:
                response_text += "\n" + analysis
            recommendations.extend(recs)
        
        # If no specific analysis was triggered, provide comprehensive document insights
        if not response_text and context.get('processed_documents'):
            response_text = self._comprehensive_document_analysis(query, context['processed_documents'])
        
        # If still no response, provide general analysis
        if not response_text:
            response_text = self._general_soil_analysis(numerical_data)
        
        return AgentResponse(
            agent_type=self.name,
            query=query,
            response=response_text or "No specific soil data found to analyze. Please upload geotechnical documents for detailed analysis.",
            confidence=0.85 if response_text else 0.5,
            recommendations=recommendations,
            warnings=warnings,
            data_used=numerical_data,
            document_based=bool(context.get('processed_documents')),
            timestamp=datetime.now().isoformat()
        )
    
    def _extract_numerical_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract numerical data from context"""
        data = {}
        
        # Get from processed documents
        for doc_name, doc_data in context.get('processed_documents', {}).items():
            if 'numerical_data' in doc_data:
                for param_type, values in doc_data['numerical_data'].items():
                    if param_type not in data:
                        data[param_type] = []
                    data[param_type].extend(values)
        
        return data
    
    def _analyze_spt_data(self, data: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
        """Analyze SPT N-values"""
        spt_values = data.get('spt_values', [])
        recommendations = []
        warnings = []
        
        if not spt_values:
            return "No SPT data available for analysis.", [], []
        
        # Extract values
        n_values = [v['value'] if isinstance(v, dict) else v.value for v in spt_values]
        
        # Statistical analysis
        min_n = min(n_values)
        max_n = max(n_values)
        avg_n = np.mean(n_values)
        
        response = f"**SPT Analysis Results:**\n\n"
        response += f"- Number of tests: {len(n_values)}\n"
        response += f"- N-value range: {min_n:.0f} to {max_n:.0f}\n"
        response += f"- Average N-value: {avg_n:.1f}\n\n"
        
        # Soil consistency interpretation
        response += "**Soil Consistency Interpretation:**\n"
        
        for i, n in enumerate(n_values):
            depth_info = ""
            if i < len(spt_values) and hasattr(spt_values[i], 'depth'):
                depth_info = f" at {spt_values[i].depth}m"
            
            if n < 5:
                consistency = "Very loose/Very soft"
                warnings.append(f"Very low SPT value ({n}){depth_info} - potential stability issues")
            elif n < 10:
                consistency = "Loose/Soft"
            elif n < 30:
                consistency = "Medium dense/Medium stiff"
            elif n < 50:
                consistency = "Dense/Stiff"
            else:
                consistency = "Very dense/Hard"
            
            response += f"- N={n}{depth_info}: {consistency}\n"
        
        # Recommendations
        if avg_n < 10:
            recommendations.append("Consider ground improvement techniques for low SPT values")
            recommendations.append("Deep foundations may be required")
        elif avg_n < 20:
            recommendations.append("Shallow foundations possible with adequate design")
            recommendations.append("Monitor settlement carefully")
        else:
            recommendations.append("Good bearing capacity expected")
            recommendations.append("Shallow foundations generally suitable")
        
        return response, recommendations, warnings
    
    def _analyze_bearing_capacity(self, data: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
        """Analyze bearing capacity data"""
        bearing_data = data.get('bearing_capacity', [])
        recommendations = []
        warnings = []
        
        if not bearing_data:
            return "No bearing capacity data available.", [], []
        
        # Extract values
        values = [v['value'] if isinstance(v, dict) else v.value for v in bearing_data]
        units = [v.get('unit', 'kPa') if isinstance(v, dict) else v.unit for v in bearing_data]
        
        # Convert to kPa for consistency
        values_kpa = []
        for val, unit in zip(values, units):
            if 'mpa' in unit.lower():
                values_kpa.append(val * 1000)
            else:
                values_kpa.append(val)
        
        min_bc = min(values_kpa)
        max_bc = max(values_kpa)
        avg_bc = np.mean(values_kpa)
        
        response = f"**Bearing Capacity Analysis:**\n\n"
        response += f"- Bearing capacity range: {min_bc:.0f} to {max_bc:.0f} kPa\n"
        response += f"- Average: {avg_bc:.0f} kPa\n\n"
        
        # Safety factor analysis
        if avg_bc < 100:
            warnings.append("Low bearing capacity - careful foundation design required")
            recommendations.append("Consider pile foundations or ground improvement")
        elif avg_bc < 200:
            recommendations.append("Moderate bearing capacity - suitable for light structures")
        else:
            recommendations.append("Good bearing capacity for most structures")
        
        # Check if values are allowable or ultimate
        if any('ultimate' in str(v).lower() for v in bearing_data):
            recommendations.append("Apply appropriate safety factor (typically 3.0) for design")
        
        return response, recommendations, warnings
    
    def _analyze_soil_classification(self, data: Dict[str, Any]) -> str:
        """Analyze soil classification parameters"""
        response = "**Soil Classification Analysis:**\n\n"
        
        # Check for Atterberg limits
        ll_data = data.get('liquid_limit', [])
        pl_data = data.get('plastic_limit', [])
        pi_data = data.get('plasticity_index', [])
        
        if ll_data and pl_data:
            ll = np.mean([v['value'] if isinstance(v, dict) else v.value for v in ll_data])
            pl = np.mean([v['value'] if isinstance(v, dict) else v.value for v in pl_data])
            pi = ll - pl
            
            response += f"- Liquid Limit (LL): {ll:.1f}%\n"
            response += f"- Plastic Limit (PL): {pl:.1f}%\n"
            response += f"- Plasticity Index (PI): {pi:.1f}%\n\n"
            
            # USCS Classification
            if pi < 7:
                if ll < 50:
                    response += "Classification: ML (Silt of low plasticity)\n"
                else:
                    response += "Classification: MH (Silt of high plasticity)\n"
            else:
                if ll < 50:
                    response += "Classification: CL (Clay of low plasticity)\n"
                else:
                    response += "Classification: CH (Clay of high plasticity)\n"
        
        # Check density and moisture
        density_data = data.get('density', [])
        moisture_data = data.get('moisture_content', [])
        
        if density_data:
            density = np.mean([v['value'] if isinstance(v, dict) else v.value for v in density_data])
            response += f"\nDry Density: {density:.2f} g/cm³\n"
        
        if moisture_data:
            moisture = np.mean([v['value'] if isinstance(v, dict) else v.value for v in moisture_data])
            response += f"Moisture Content: {moisture:.1f}%\n"
        
        return response
    
    def _analyze_settlement(self, data: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Analyze settlement data"""
        recommendations = []
        response = "**Settlement Analysis:**\n\n"
        
        settlement_data = data.get('settlement', [])
        if settlement_data:
            values = [v['value'] if isinstance(v, dict) else v.value for v in settlement_data]
            total_settlement = sum(values)
            
            response += f"- Total estimated settlement: {total_settlement:.1f} mm\n"
            
            if total_settlement > 50:
                recommendations.append("High settlement expected - consider ground improvement")
            elif total_settlement > 25:
                recommendations.append("Moderate settlement - monitor during construction")
            else:
                recommendations.append("Settlement within acceptable limits")
        
        # Estimate from SPT if no direct settlement data
        elif 'spt_values' in data:
            response += "Estimating settlement potential from SPT data...\n"
            spt_values = [v['value'] if isinstance(v, dict) else v.value for v in data['spt_values']]
            avg_n = np.mean(spt_values)
            
            if avg_n < 10:
                response += "- High settlement potential due to low SPT values\n"
                recommendations.append("Detailed settlement analysis recommended")
            elif avg_n < 20:
                response += "- Moderate settlement potential\n"
            else:
                response += "- Low settlement potential\n"
        
        return response, recommendations
    
    def _general_soil_analysis(self, data: Dict[str, Any]) -> str:
        """Provide general soil analysis"""
        response = "**General Soil Analysis:**\n\n"
        
        if not data:
            return "No soil data available for analysis. Please upload geotechnical documents containing soil investigation results, SPT data, laboratory test results, or engineering reports."
        
        # Summarize available data
        response += "Available soil parameters:\n"
        for param_type, values in data.items():
            if values:
                count = len(values)
                response += f"- {param_type.replace('_', ' ').title()}: {count} measurements\n"
        
        # Provide general recommendations
        response += "\n**General Recommendations:**\n"
        response += "- Ensure all critical soil parameters are tested\n"
        response += "- Consider seasonal variations in soil properties\n"
        response += "- Verify field test results with laboratory testing\n"
        
        return response
    
    def _comprehensive_document_analysis(self, query: str, documents: Dict[str, Any]) -> str:
        """Provide comprehensive analysis from all document content"""
        response = "**Document Analysis Results:**\n\n"
        
        for doc_name, doc_data in documents.items():
            if doc_data.get('document_type') == 'image' and 'content' in doc_data:
                ai_response = doc_data['content'].get('response', '')
                if ai_response:
                    response += f"**From {doc_name}:**\n"
                    # Include full AI response, not just a snippet
                    response += ai_response + "\n\n"
        
        if response == "**Document Analysis Results:**\n\n":
            response = "No detailed analysis available from documents. Please ensure documents contain geotechnical information."
        
        return response

class TunnelSupportAgent(BaseGeotechnicalAgent):
    """Agent specialized in tunnel engineering and support systems"""
    
    def __init__(self):
        super().__init__("Tunnel Engineering Expert")
        self.expertise_areas = [
            "tunnel", "rock", "rqd", "support", "lining", "excavation",
            "shotcrete", "rockbolt", "tbm", "natm", "convergence"
        ]
        
    def analyze(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        """Analyze tunnel-related queries"""
        from datetime import datetime
        
        response_text = ""
        recommendations = []
        warnings = []
        data_used = {}
        
        # First, search through document content for relevant information
        if context.get('processed_documents'):
            relevant_sections = self.search_document_content(query, context['processed_documents'])
            if relevant_sections:
                response_text = "Based on the uploaded documents:\n\n"
                for section in relevant_sections[:3]:
                    response_text += f"**From {section['source']}:**\n{section['content']}\n\n"
        
        # Extract relevant data
        numerical_data = self._extract_numerical_data(context)
        
        # Analyze based on query type
        if "support" in query.lower() or "lining" in query.lower():
            analysis, recs = self._analyze_tunnel_support(numerical_data)
            if analysis:
                response_text += "\n" + analysis
            recommendations.extend(recs)
            
        elif "rqd" in query.lower() or "rock quality" in query.lower():
            analysis, recs, warns = self._analyze_rock_quality(numerical_data)
            if analysis:
                response_text += "\n" + analysis
            recommendations.extend(recs)
            warnings.extend(warns)
            
        elif "excavation" in query.lower():
            analysis, recs = self._analyze_excavation_method(numerical_data)
            if analysis:
                response_text += "\n" + analysis
            recommendations.extend(recs)
        
        # If no specific analysis was triggered, provide comprehensive document insights
        if not response_text and context.get('processed_documents'):
            response_text = self._comprehensive_document_analysis(query, context['processed_documents'])
        
        # If still no response, provide general analysis
        if not response_text:
            response_text = self._general_tunnel_analysis(numerical_data)
        
        return AgentResponse(
            agent_type=self.name,
            query=query,
            response=response_text or "No specific tunnel engineering data found.",
            confidence=0.85 if response_text else 0.5,
            recommendations=recommendations,
            warnings=warnings,
            data_used=numerical_data,
            document_based=bool(context.get('processed_documents')),
            timestamp=datetime.now().isoformat()
        )
    
    def _extract_numerical_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract numerical data relevant to tunneling"""
        data = {}
        
        for doc_name, doc_data in context.get('processed_documents', {}).items():
            if 'numerical_data' in doc_data:
                for param_type, values in doc_data['numerical_data'].items():
                    # Include all rock mechanics parameters
                    if param_type not in data:
                        data[param_type] = []
                    data[param_type].extend(values)
        
        return data
    
    def _analyze_tunnel_support(self, data: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Analyze tunnel support requirements"""
        recommendations = []
        response = "**Tunnel Support System Analysis:**\n\n"
        
        # Check rock quality
        rqd_data = data.get('rqd', [])
        ucs_data = data.get('ucs', [])
        
        if rqd_data:
            rqd_values = [v['value'] if isinstance(v, dict) else v.value for v in rqd_data]
            avg_rqd = np.mean(rqd_values)
            
            response += f"Rock Quality Designation (RQD): {avg_rqd:.1f}%\n"
            
            # Support recommendations based on RQD
            if avg_rqd < 25:
                response += "- Very poor rock quality\n"
                recommendations.append("Heavy support required: Steel sets with lagging")
                recommendations.append("Consider forepoling or spiling")
            elif avg_rqd < 50:
                response += "- Poor rock quality\n"
                recommendations.append("Systematic bolting with shotcrete")
                recommendations.append("Steel fiber reinforced shotcrete recommended")
            elif avg_rqd < 75:
                response += "- Fair rock quality\n"
                recommendations.append("Pattern bolting with mesh and shotcrete")
            elif avg_rqd < 90:
                response += "- Good rock quality\n"
                recommendations.append("Spot bolting where required")
            else:
                response += "- Excellent rock quality\n"
                recommendations.append("Minimal support required")
        
        if ucs_data:
            ucs_values = [v['value'] if isinstance(v, dict) else v.value for v in ucs_data]
            avg_ucs = np.mean(ucs_values)
            
            response += f"\nUnconfined Compressive Strength: {avg_ucs:.1f} MPa\n"
            
            if avg_ucs < 25:
                recommendations.append("Low rock strength - careful excavation required")
            elif avg_ucs > 100:
                recommendations.append("High rock strength - suitable for most tunneling methods")
        
        return response, recommendations
    
    def _analyze_rock_quality(self, data: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
        """Analyze rock quality parameters"""
        recommendations = []
        warnings = []
        response = "**Rock Mass Quality Assessment:**\n\n"
        
        # RQD Analysis
        rqd_data = data.get('rqd', [])
        if rqd_data:
            rqd_values = [v['value'] if isinstance(v, dict) else v.value for v in rqd_data]
            
            response += "RQD Classification:\n"
            for rqd in rqd_values:
                if rqd < 25:
                    quality = "Very Poor"
                    warnings.append(f"Very poor rock quality (RQD={rqd}%) detected")
                elif rqd < 50:
                    quality = "Poor"
                elif rqd < 75:
                    quality = "Fair"
                elif rqd < 90:
                    quality = "Good"
                else:
                    quality = "Excellent"
                
                response += f"- RQD {rqd}%: {quality}\n"
        
        # Q-System estimation if data available
        if rqd_data and data.get('cohesion'):
            response += "\n**Estimated Q-System Rating:**\n"
            avg_rqd = np.mean(rqd_values)
            q_estimate = (avg_rqd / 100) * 10  # Simplified estimation
            
            response += f"Q ≈ {q_estimate:.1f}\n"
            
            if q_estimate < 1:
                recommendations.append("Exceptionally poor ground conditions")
            elif q_estimate < 10:
                recommendations.append("Poor to fair ground conditions")
            else:
                recommendations.append("Fair to good ground conditions")
        
        return response, recommendations, warnings
    
    def _analyze_excavation_method(self, data: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Recommend excavation methods"""
        recommendations = []
        response = "**Excavation Method Analysis:**\n\n"
        
        # Analyze ground conditions
        rqd_avg = 50  # Default
        ucs_avg = 50  # Default
        
        if 'rqd' in data:
            rqd_values = [v['value'] if isinstance(v, dict) else v.value for v in data['rqd']]
            rqd_avg = np.mean(rqd_values)
        
        if 'ucs' in data:
            ucs_values = [v['value'] if isinstance(v, dict) else v.value for v in data['ucs']]
            ucs_avg = np.mean(ucs_values)
        
        # Method selection
        if rqd_avg < 30 or ucs_avg < 25:
            response += "Recommended method: NATM with careful sequencing\n"
            recommendations.append("Use multiple drifts or pilot tunnel")
            recommendations.append("Immediate support installation required")
        elif rqd_avg < 60:
            response += "Recommended method: Conventional drill and blast or roadheader\n"
            recommendations.append("Controlled blasting to minimize disturbance")
        else:
            response += "Recommended method: TBM or drill and blast\n"
            recommendations.append("TBM suitable for long tunnels")
            recommendations.append("Good advance rates expected")
        
        response += f"\nBased on: RQD={rqd_avg:.0f}%, UCS={ucs_avg:.0f} MPa\n"
        
        return response, recommendations
    
    def _general_tunnel_analysis(self, data: Dict[str, Any]) -> str:
        """General tunnel engineering analysis"""
        response = "**General Tunnel Engineering Assessment:**\n\n"
        
        if not data:
            response += "Limited geotechnical data available.\n"
            response += "\nRecommended investigations:\n"
            response += "- Rock Quality Designation (RQD) testing\n"
            response += "- Unconfined Compressive Strength (UCS) tests\n"
            response += "- Joint survey and orientation analysis\n"
            response += "- In-situ stress measurements\n"
        else:
            response += "Available tunnel-related parameters:\n"
            for param, values in data.items():
                if values:
                    response += f"- {param.upper()}: {len(values)} measurements\n"
            
            response += "\n**General Guidelines:**\n"
            response += "- Perform detailed geological mapping\n"
            response += "- Monitor convergence during excavation\n"
            response += "- Design support based on observational method\n"
        
        return response
    
    def _comprehensive_document_analysis(self, query: str, documents: Dict[str, Any]) -> str:
        """Provide comprehensive analysis from all document content"""
        response = "**Document Analysis Results:**\n\n"
        
        for doc_name, doc_data in documents.items():
            if doc_data.get('document_type') == 'image' and 'content' in doc_data:
                ai_response = doc_data['content'].get('response', '')
                if 'rock' in ai_response.lower() or 'tunnel' in ai_response.lower() or 'rqd' in ai_response.lower():
                    response += f"**From {doc_name}:**\n"
                    response += ai_response + "\n\n"
        
        if response == "**Document Analysis Results:**\n\n":
            response = "No tunnel engineering data found in documents. Please upload documents containing rock mechanics data, RQD values, or tunnel design information."
        
        return response

class SafetyChecklistAgent(BaseGeotechnicalAgent):
    """Agent for generating safety checklists and risk assessments"""
    
    def __init__(self):
        super().__init__("Safety & Risk Assessment Expert")
        self.expertise_areas = [
            "safety", "risk", "hazard", "checklist", "assessment",
            "stability", "failure", "factor of safety", "monitoring"
        ]
        
    def analyze(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        """Generate safety analysis and checklists"""
        from datetime import datetime
        
        response_text = ""
        recommendations = []
        warnings = []
        data_used = {}
        
        # Extract data
        numerical_data = self._extract_numerical_data(context)
        
        # Generate appropriate safety analysis
        if "checklist" in query.lower():
            response_text = self._generate_safety_checklist(numerical_data)
        elif "risk" in query.lower():
            response_text, warns = self._perform_risk_assessment(numerical_data)
            warnings.extend(warns)
        elif "stability" in query.lower():
            response_text, recs, warns = self._analyze_stability(numerical_data)
            recommendations.extend(recs)
            warnings.extend(warns)
        else:
            response_text = self._general_safety_analysis(numerical_data)
        
        # Always add general safety recommendations
        recommendations.extend([
            "Implement continuous monitoring during construction",
            "Regular safety audits and inspections",
            "Emergency response procedures in place"
        ])
        
        return AgentResponse(
            agent_type=self.name,
            query=query,
            response=response_text,
            confidence=0.90,
            recommendations=recommendations,
            warnings=warnings,
            data_used=numerical_data,
            document_based=bool(context.get('processed_documents')),
            timestamp=datetime.now().isoformat()
        )
    
    def _extract_numerical_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all numerical data for safety assessment"""
        data = {}
        
        for doc_name, doc_data in context.get('processed_documents', {}).items():
            if 'numerical_data' in doc_data:
                for param_type, values in doc_data['numerical_data'].items():
                    if param_type not in data:
                        data[param_type] = []
                    data[param_type].extend(values)
        
        return data
    
    def _generate_safety_checklist(self, data: Dict[str, Any]) -> str:
        """Generate comprehensive safety checklist"""
        response = "**Geotechnical Safety Checklist:**\n\n"
        
        # Site Investigation
        response += "**1. Site Investigation & Testing:**\n"
        response += "☐ Comprehensive soil/rock testing completed\n"
        response += "☐ All boreholes properly logged and documented\n"
        response += "☐ Laboratory test results verified\n"
        response += "☐ Groundwater conditions assessed\n\n"
        
        # Design Verification
        response += "**2. Design Verification:**\n"
        response += "☐ Design calculations independently checked\n"
        response += "☐ Safety factors meet code requirements\n"
        response += "☐ Settlement analysis completed\n"
        response += "☐ Stability analysis performed\n\n"
        
        # Construction Safety
        response += "**3. Construction Safety:**\n"
        response += "☐ Excavation support systems designed\n"
        response += "☐ Slope stability during construction verified\n"
        response += "☐ Dewatering plan in place\n"
        response += "☐ Temporary works designed and checked\n\n"
        
        # Monitoring
        response += "**4. Monitoring Requirements:**\n"
        response += "☐ Settlement monitoring points installed\n"
        response += "☐ Piezometers for groundwater monitoring\n"
        response += "☐ Inclinometers for lateral movement\n"
        response += "☐ Regular inspection schedule established\n\n"
        
        # Emergency Procedures
        response += "**5. Emergency Procedures:**\n"
        response += "☐ Emergency response plan documented\n"
        response += "☐ Evacuation procedures established\n"
        response += "☐ Communication protocols defined\n"
        response += "☐ Emergency equipment available on-site\n"
        
        # Data-specific warnings
        if data.get('spt_values'):
            spt_values = [v['value'] if isinstance(v, dict) else v.value for v in data['spt_values']]
            if any(n < 10 for n in spt_values):
                response += "\n**⚠️ CRITICAL: Low SPT values detected - Extra caution required**\n"
        
        return response
    
    def _perform_risk_assessment(self, data: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Perform risk assessment based on available data"""
        warnings = []
        response = "**Geotechnical Risk Assessment:**\n\n"
        
        risks = []
        
        # Assess SPT-based risks
        if 'spt_values' in data:
            spt_values = [v['value'] if isinstance(v, dict) else v.value for v in data['spt_values']]
            min_spt = min(spt_values)
            
            if min_spt < 5:
                risks.append(("VERY HIGH", "Liquefaction potential in loose soils"))
                warnings.append("Critical: Very low SPT values indicate high liquefaction risk")
            elif min_spt < 10:
                risks.append(("HIGH", "Settlement risk in soft soils"))
                warnings.append("High settlement risk due to low SPT values")
            elif min_spt < 20:
                risks.append(("MODERATE", "Moderate bearing capacity concerns"))
        
        # Assess bearing capacity risks
        if 'bearing_capacity' in data:
            bc_values = [v['value'] if isinstance(v, dict) else v.value for v in data['bearing_capacity']]
            min_bc = min(bc_values)
            
            if min_bc < 100:  # kPa
                risks.append(("HIGH", "Low bearing capacity"))
                warnings.append("Low bearing capacity may require special foundation design")
            elif min_bc < 200:
                risks.append(("MODERATE", "Limited bearing capacity"))
        
        # Assess groundwater risks
        if 'moisture_content' in data:
            moisture_values = [v['value'] if isinstance(v, dict) else v.value for v in data['moisture_content']]
            if any(m > 30 for m in moisture_values):
                risks.append(("HIGH", "High groundwater or saturated conditions"))
                warnings.append("High moisture content indicates potential groundwater issues")
        
        # Present risk matrix
        response += "**Risk Matrix:**\n"
        for level, description in risks:
            response += f"- [{level}] {description}\n"
        
        if not risks:
            response += "- No critical risks identified based on available data\n"
        
        # Mitigation measures
        response += "\n**Risk Mitigation Measures:**\n"
        if any(level in ["VERY HIGH", "HIGH"] for level, _ in risks):
            response += "- Implement comprehensive monitoring program\n"
            response += "- Consider ground improvement techniques\n"
            response += "- Develop contingency plans\n"
            response += "- Increase safety factors in design\n"
        else:
            response += "- Standard monitoring procedures\n"
            response += "- Regular inspections\n"
            response += "- Follow standard safety protocols\n"
        
        return response, warnings
    
    def _analyze_stability(self, data: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
        """Analyze stability concerns"""
        recommendations = []
        warnings = []
        response = "**Stability Analysis:**\n\n"
        
        # Check for stability parameters
        cohesion = None
        friction_angle = None
        
        if 'cohesion' in data:
            cohesion_values = [v['value'] if isinstance(v, dict) else v.value for v in data['cohesion']]
            cohesion = np.mean(cohesion_values)
            
        if 'friction_angle' in data:
            friction_values = [v['value'] if isinstance(v, dict) else v.value for v in data['friction_angle']]
            friction_angle = np.mean(friction_values)
        
        if cohesion is not None and friction_angle is not None:
            response += f"Shear strength parameters:\n"
            response += f"- Cohesion (c): {cohesion:.1f} kPa\n"
            response += f"- Friction angle (φ): {friction_angle:.1f}°\n\n"
            
            # Simple stability assessment
            if cohesion < 10 and friction_angle < 25:
                response += "**Stability Concern: LOW shear strength parameters**\n"
                warnings.append("Low shear strength - stability analysis critical")
                recommendations.append("Detailed slope stability analysis required")
                recommendations.append("Consider soil reinforcement")
            elif cohesion < 20 or friction_angle < 30:
                response += "**Stability Status: Moderate strength parameters**\n"
                recommendations.append("Standard stability analysis recommended")
            else:
                response += "**Stability Status: Good strength parameters**\n"
                recommendations.append("Standard design procedures appropriate")
        
        # Factor of safety recommendations
        response += "\n**Recommended Safety Factors:**\n"
        response += "- Permanent slopes: 1.5 minimum\n"
        response += "- Temporary excavations: 1.3 minimum\n"
        response += "- Critical structures: 2.0 minimum\n"
        
        return response, recommendations, warnings
    
    def _general_safety_analysis(self, data: Dict[str, Any]) -> str:
        """General safety analysis and recommendations"""
        response = "**General Safety Assessment:**\n\n"
        
        response += "**Key Safety Considerations:**\n"
        response += "1. **Ground Conditions**: Verify all assumptions with field data\n"
        response += "2. **Construction Sequence**: Plan and review each stage\n"
        response += "3. **Monitoring**: Implement comprehensive monitoring program\n"
        response += "4. **Quality Control**: Regular testing and inspection\n"
        response += "5. **Documentation**: Maintain complete records\n\n"
        
        response += "**Best Practices:**\n"
        response += "- Never exceed design assumptions\n"
        response += "- Stop work if unexpected conditions encountered\n"
        response += "- Regular safety meetings and training\n"
        response += "- Clear communication channels\n"
        response += "- Emergency procedures well-documented\n"
        
        if data:
            response += "\n**Data-Specific Observations:**\n"
            param_count = sum(len(v) for v in data.values() if v)
            response += f"- {param_count} test measurements available\n"
            response += "- Ensure sufficient data coverage across site\n"
            response += "- Verify critical parameters with additional testing if needed\n"
        
        return response

class GeotechnicalAgentOrchestrator:
    """Orchestrates multiple agents for comprehensive analysis"""
    
    def __init__(self):
        self.agents = {
            'soil': SoilAnalysisAgent(),
            'tunnel': TunnelSupportAgent(),
            'safety': SafetyChecklistAgent()
        }
        
    def route_query(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        """Route query to most appropriate agent"""
        # First, check if query is asking for general information from documents
        query_lower = query.lower()
        
        # If it's a general query, use the soil agent as default for document analysis
        if any(word in query_lower for word in ['what', 'tell me', 'explain', 'describe', 'show me', 'information']):
            # Use soil agent for general queries
            selected_agent = self.agents['soil']
            response = selected_agent.analyze(query, context)
            response.confidence = 0.9
            return response
        
        # Otherwise, calculate confidence scores for each agent
        scores = {}
        for agent_type, agent in self.agents.items():
            scores[agent_type] = agent.can_handle(query)
        
        # Select agent with highest confidence
        best_agent_type = max(scores, key=scores.get)
        best_score = scores[best_agent_type]
        
        # If no agent has good confidence, use soil agent as default
        if best_score < 0.3:
            best_agent_type = 'soil'
        
        # Get response from selected agent
        selected_agent = self.agents[best_agent_type]
        response = selected_agent.analyze(query, context)
        
        # Add routing information
        response.confidence = best_score
        
        return response
    
    def get_comprehensive_analysis(self, context: Dict[str, Any]) -> Dict[str, AgentResponse]:
        """Get analysis from all agents"""
        results = {}
        
        for agent_type, agent in self.agents.items():
            query = f"Provide {agent_type} analysis for the uploaded documents"
            results[agent_type] = agent.analyze(query, context)
        
        return results
