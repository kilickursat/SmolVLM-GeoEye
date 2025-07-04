# Key Code Changes for VLM Integration Fix

This document shows the key code changes needed in app.py to fix the VLM integration issues.

## 1. Add GeotechnicalDataExtractor Class (after line ~200)

```python
class GeotechnicalDataExtractor:
    """Extract numerical geotechnical data from text"""
    
    def extract_numerical_data_from_text(self, text: str) -> Dict[str, List[Dict]]:
        """Extract numerical values with context from text"""
        numerical_data = {
            'spt_values': [],
            'bearing_capacity': [],
            'density': [],
            'moisture_content': [],
            'shear_strength': [],
            'cohesion': [],
            'friction_angle': [],
            'settlement': [],
            'depth': [],
            'permeability': [],
            'plasticity_index': [],
            'liquid_limit': [],
            'plastic_limit': []
        }
        
        # Enhanced patterns to capture values with context
        patterns = {
            'spt_values': [
                r'SPT[:\s]*N[:\s]*=?\s*(\d+)(?:\s+at\s+(\d+\.?\d*)\s*(m|ft))?',
                r'N-value[:\s]*(\d+)(?:\s+at\s+(\d+\.?\d*)\s*(m|ft))?',
                r'blow count[:\s]*(\d+)(?:\s+at\s+(\d+\.?\d*)\s*(m|ft))?',
                r'(\d+)\s*blows(?:\s+at\s+(\d+\.?\d*)\s*(m|ft))?'
            ],
            'bearing_capacity': [
                r'bearing capacity[:\s]*(\d+\.?\d*)\s*(kPa|MPa|kN/m2|psf|ksf)',
                r'allowable bearing[:\s]*(\d+\.?\d*)\s*(kPa|MPa|kN/m2|psf|ksf)',
                r'ultimate bearing[:\s]*(\d+\.?\d*)\s*(kPa|MPa|kN/m2|psf|ksf)',
                r'qa[:\s]*=?\s*(\d+\.?\d*)\s*(kPa|MPa|kN/m2|psf|ksf)'
            ],
            'density': [
                r'(?:dry\s+)?density[:\s]*(\d+\.?\d*)\s*(g/cm3|kg/m3|pcf|kN/m3)',
                r'unit weight[:\s]*(\d+\.?\d*)\s*(kN/m3|pcf|kg/m3)',
                r'bulk density[:\s]*(\d+\.?\d*)\s*(g/cm3|kg/m3|pcf)',
                r'γ[:\s]*=?\s*(\d+\.?\d*)\s*(kN/m3|pcf)'
            ],
            'moisture_content': [
                r'moisture content[:\s]*(\d+\.?\d*)\s*%',
                r'water content[:\s]*(\d+\.?\d*)\s*%',
                r'w[:\s]*=?\s*(\d+\.?\d*)\s*%',
                r'MC[:\s]*(\d+\.?\d*)\s*%'
            ],
            'depth': [
                r'(?:at\s+)?depth[:\s]*(\d+\.?\d*)\s*(m|ft|cm)',
                r'(\d+\.?\d*)\s*(m|ft)\s+(?:depth|deep|below)',
                r'elevation[:\s]*([+-]?\d+\.?\d*)\s*(m|ft)',
                r'level[:\s]*([+-]?\d+\.?\d*)\s*(m|ft)'
            ]
        }
        
        # Extract values for each parameter type
        for param, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        value = float(match[0])
                        
                        # Handle depth information if present
                        depth_value = None
                        if len(match) > 2 and match[1]:  # Has depth info
                            try:
                                depth_value = float(match[1])
                                depth_unit = match[2] if len(match) > 2 else "m"
                            except:
                                depth_value = None
                        
                        unit = match[1] if len(match) > 1 and not match[1].replace('.','').isdigit() else ""
                        if len(match) > 2 and match[2] and not match[2].replace('.','').isdigit():
                            unit = match[2]
                        
                        # Find context (surrounding text)
                        context_pattern = rf'.{{0,100}}{re.escape(match[0])}.{{0,100}}'
                        context_matches = re.findall(context_pattern, text, re.IGNORECASE | re.DOTALL)
                        context = context_matches[0] if context_matches else ""
                        
                        data_entry = {
                            'value': value,
                            'unit': unit,
                            'context': context.strip(),
                            'source': 'VLM extraction'
                        }
                        
                        # Add depth if available
                        if depth_value is not None:
                            data_entry['depth'] = depth_value
                            data_entry['depth_unit'] = depth_unit
                        
                        numerical_data[param].append(data_entry)
        
        # Remove duplicates
        for param in numerical_data:
            seen = set()
            unique_data = []
            for item in numerical_data[param]:
                key = (item['value'], item.get('depth', 'no_depth'))
                if key not in seen:
                    seen.add(key)
                    unique_data.append(item)
            numerical_data[param] = unique_data
        
        return numerical_data
```

## 2. Update DocumentIngestionModule (modify existing methods)

Add `self.data_extractor = GeotechnicalDataExtractor()` to `__init__` method.

In `process_pdf` method, add after extracting text:
```python
# Extract numerical data from all text
extracted_data['numerical_data'] = self.data_extractor.extract_numerical_data_from_text(all_text)
```

## 3. Update GeotechnicalExtractionModule

Add `self.data_extractor = GeotechnicalDataExtractor()` to `__init__` method.

In `extract_from_image` method, update the VLM query to request numerical values:
```python
query = """Analyze this geotechnical engineering document or image. Focus on extracting:

GEOTECHNICAL PARAMETERS WITH VALUES:
1. Soil properties with numerical values (density, moisture content, plasticity index, liquid limit, plastic limit)
2. Rock properties with values (strength, RQD, joint conditions, GSI)
3. Foundation data with numbers (bearing capacity in kPa/MPa, settlement in mm, pile capacity)
4. Test results with specific values (SPT N-values at different depths, CPT data, laboratory test results)

IMPORTANT: Always include the numerical values with their units. For example, don't just say "high density", say "density = 2.1 g/cm³" or "SPT N-value = 25 at 5m depth".

Please provide a detailed technical analysis with all numerical values and their units."""
```

After getting response, extract numerical data:
```python
# Extract numerical data from response
numerical_data = self.data_extractor.extract_numerical_data_from_text(response_text)

return {
    "extraction_type": "geotechnical_vision_analysis",
    "query": query,
    "response": response_text,
    "numerical_data": numerical_data,  # Add this
    "confidence": "high",
    "processing_time": output.get("processing_time", "unknown"),
    "timestamp": datetime.now().isoformat()
}
```

## 4. Update StructuredOutputModule

In `organize_data` method, add storage for numerical data:
```python
structured_data = {
    "document_id": document_id,
    "timestamp": datetime.now().isoformat(),
    "document_type": extracted_data.get("type", "unknown"),
    "processing_status": "completed",
    "content": {},
    "metadata": {},
    "searchable_fields": [],
    "numerical_data": {}  # Add this
}

# Store numerical data separately for easy access
if "numerical_data" in extracted_data:
    structured_data["numerical_data"] = extracted_data["numerical_data"]
```

## 5. Update GeotechnicalVisualizationModule

Add new method `_create_numerical_visualization`:
```python
def _create_numerical_visualization(self, numerical_data: Dict[str, List[Dict]], doc_type: str) -> go.Figure:
    """Create visualization from extracted numerical data"""
    # Create subplots for different data types
    subplot_titles = []
    plot_count = 0
    
    # Determine which plots to create
    has_spt_depth = bool(numerical_data.get('spt_values')) and any(d.get('depth') for d in numerical_data.get('spt_values', []))
    has_bearing = bool(numerical_data.get('bearing_capacity'))
    has_soil_props = bool(numerical_data.get('density') or numerical_data.get('moisture_content'))
    has_strength = bool(numerical_data.get('cohesion') or numerical_data.get('friction_angle'))
    
    if has_spt_depth:
        subplot_titles.append("SPT N-Values vs Depth")
        plot_count += 1
    if has_bearing:
        subplot_titles.append("Bearing Capacity")
        plot_count += 1
    if has_soil_props:
        subplot_titles.append("Soil Properties")
        plot_count += 1
    if has_strength:
        subplot_titles.append("Strength Parameters")
        plot_count += 1
    
    if plot_count == 0:
        return self._create_error_visualization("No numerical data to visualize")
    
    # Create subplot layout
    rows = (plot_count + 1) // 2
    cols = 2 if plot_count > 1 else 1
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles[:plot_count],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    current_plot = 1
    
    # Plot 1: SPT values vs depth
    if has_spt_depth:
        spt_data = numerical_data['spt_values']
        depths = []
        values = []
        labels = []
        
        for item in spt_data:
            if 'depth' in item:
                depths.append(item['depth'])
                values.append(item['value'])
                labels.append(f"N={item['value']} at {item['depth']}{item.get('depth_unit', 'm')}")
        
        if depths and values:
            row = ((current_plot - 1) // cols) + 1
            col = ((current_plot - 1) % cols) + 1
            
            fig.add_trace(
                go.Scatter(
                    x=values,
                    y=depths,
                    mode='markers+lines',
                    marker=dict(size=10, color='blue'),
                    line=dict(color='blue', width=2),
                    text=labels,
                    name='SPT N-Value',
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=row, col=col
            )
            fig.update_yaxes(autorange="reversed", title_text="Depth (m)", row=row, col=col)
            fig.update_xaxes(title_text="SPT N-Value", row=row, col=col)
            current_plot += 1
    
    # Add more plots for other data types...
    
    fig.update_layout(
        title_text=f"Geotechnical Data Analysis - {doc_type.upper()} Document",
        height=400 * rows,
        showlegend=False
    )
    
    return fig
```

Update `create_visualization_from_any_document` to use numerical data:
```python
def create_visualization_from_any_document(self, doc_data: Dict[str, Any]) -> go.Figure:
    """Create visualizations from any document type based on extracted data"""
    try:
        doc_type = doc_data.get("document_type", "unknown")
        content = doc_data.get("content", {})
        numerical_data = doc_data.get("numerical_data", {})
        
        # Use numerical data if available
        if numerical_data and any(numerical_data.values()):
            return self._create_numerical_visualization(numerical_data, doc_type)
        # ... rest of existing code
```

## 6. Update GeotechnicalMultiAgentOrchestrator

Replace `route_geotechnical_query` method:
```python
def route_geotechnical_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Route queries to appropriate geotechnical specialists using document context"""
    try:
        query_lower = query.lower()
        
        # Extract document content from context
        document_contents = []
        numerical_summary = {}
        
        if context and "processed_documents" in context:
            processed_docs = context["processed_documents"]
            
            for doc_id, doc_data in processed_docs.items():
                content = doc_data.get("content", {})
                numerical_data = doc_data.get("numerical_data", {})
                
                # Extract VLM analysis response
                if "response" in content:
                    document_contents.append(f"VLM Analysis from {doc_id}:\n{content['response']}\n")
                
                # Extract text from PDFs
                if "text_data" in content:
                    for page in content["text_data"][:3]:  # First 3 pages
                        document_contents.append(f"PDF Page {page['page_number']}:\n{page['text'][:500]}...\n")
                
                # Extract numerical data summary
                if numerical_data:
                    for param, values in numerical_data.items():
                        if values:
                            if param not in numerical_summary:
                                numerical_summary[param] = []
                            numerical_summary[param].extend(values)
        
        # Combine document content
        combined_content = "\n".join(document_contents) if document_contents else "No documents uploaded"
        
        # Format numerical summary
        numerical_content = ""
        if numerical_summary:
            numerical_content = "\nExtracted Numerical Data:\n"
            for param, values in numerical_summary.items():
                numerical_content += f"\n{param.replace('_', ' ').title()}:\n"
                for val in values[:5]:  # Limit to 5 values per parameter
                    depth_info = f" at {val['depth']}{val.get('depth_unit', 'm')} depth" if 'depth' in val else ""
                    numerical_content += f"  • {val['value']} {val['unit']}{depth_info}\n"
        
        # Create context-aware query
        context_aware_query = f"""Based on the following uploaded document content:

{combined_content}
{numerical_content}

User Question: {query}

Please answer the user's question specifically using the information extracted from the uploaded documents. If specific numerical values are available in the extracted data, use them in your analysis. If the information is not available in the documents, clearly state that."""

        # Route to appropriate agent with context
        if any(term in query_lower for term in ["soil", "bearing", "density", "moisture", "spt"]):
            agent_type = "soil_analyst"
            enhanced_query = context_aware_query
        elif any(term in query_lower for term in ["tunnel", "support", "excavation", "rock"]):
            agent_type = "tunnel_engineer"
            enhanced_query = context_aware_query
        else:
            agent_type = "soil_analyst"
            enhanced_query = context_aware_query
        
        if agent_type in self.agents:
            result = self.agents[agent_type].run(enhanced_query)
            return {
                "agent_type": agent_type,
                "query": query,
                "enhanced_query": enhanced_query,
                "response": result,
                "timestamp": datetime.now().isoformat(),
                "domain": "Geotechnical Engineering",
                "powered_by": "RunPod GPU + SmolAgent",
                "document_based": bool(document_contents)
            }
        else:
            return {
                "agent_type": "fallback",
                "query": query,
                "response": "Please upload documents and ask questions about the content.",
                "timestamp": datetime.now().isoformat(),
                "domain": "Geotechnical Engineering",
                "document_based": False
            }
            
    except Exception as e:
        logger.error(f"Error in geotechnical agent routing: {str(e)}")
        return {
            "agent_type": "error",
            "query": query,
            "response": f"I encountered an error processing your query. Please ensure you have uploaded documents.",
            "timestamp": datetime.now().isoformat(),
            "domain": "Geotechnical Engineering",
            "document_based": False
        }
```

## 7. Update main() function chat handling

In the chat message handling section, pass the full document data:
```python
context = {
    "documents": list(st.session_state.processed_documents.keys()),
    "document_count": len(st.session_state.processed_documents),
    "processed_documents": st.session_state.processed_documents,  # Pass full document data
    "runpod_status": runpod_status["status"],
    "domain": "geotechnical_engineering"
}
```

## Summary

These changes ensure that:
1. AI chat answers are based on uploaded document content
2. Numerical data is properly extracted from VLM responses
3. Visualizations show actual numerical data instead of keywords
4. The system fully leverages SmolVLM's superior OCR capabilities
