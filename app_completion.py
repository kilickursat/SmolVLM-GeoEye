                    # Show analysis details
                    with st.expander("🔍 Analysis Details"):
                        st.json({
                            "agent_type": agent_response.get("agent_type"),
                            "document_based": agent_response.get("document_based"),
                            "timestamp": agent_response.get("timestamp")
                        })
            
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        
        # Enhanced example queries for geotechnical engineering
        st.subheader("💡 Example Questions for Uploaded Documents")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📄 Document Analysis Questions:**")
            example_queries_1 = [
                "What are the SPT values in the uploaded document?",
                "Summarize the soil properties from the test results",
                "What is the bearing capacity mentioned in the report?",
                "Extract all density values with their depths"
            ]
            
            for i, example in enumerate(example_queries_1):
                if st.button(example, key=f"example1_{i}"):
                    st.session_state.messages.append({"role": "user", "content": example})
                    context = {
                        "documents": list(st.session_state.processed_documents.keys()),
                        "processed_documents": st.session_state.processed_documents,
                        "domain": "geotechnical_engineering"
                    }
                    agent_response = system["orchestrator"].route_geotechnical_query(example, context)
                    response_text = agent_response.get("response", "Could not process query.")
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    st.rerun()
        
        with col2:
            st.markdown("**🔬 Technical Analysis Questions:**")
            example_queries_2 = [
                "Analyze the soil test data and provide recommendations",
                "Is the bearing capacity sufficient for a 5-story building?",
                "What are the safety concerns based on the SPT values?",
                "Calculate settlement based on the soil parameters"
            ]
            
            for i, example in enumerate(example_queries_2):
                if st.button(example, key=f"example2_{i}"):
                    st.session_state.messages.append({"role": "user", "content": example})
                    context = {
                        "documents": list(st.session_state.processed_documents.keys()),
                        "processed_documents": st.session_state.processed_documents,
                        "domain": "geotechnical_engineering"
                    }
                    agent_response = system["orchestrator"].route_geotechnical_query(example, context)
                    response_text = agent_response.get("response", "Could not process query.")
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="geotechnical-container">', unsafe_allow_html=True)
        st.subheader("📊 Geotechnical Data Analysis")
        
        if st.session_state.processed_documents:
            doc_options = list(st.session_state.processed_documents.keys())
            selected_doc = st.selectbox("Select document for analysis:", doc_options)
            
            if selected_doc:
                doc_data = st.session_state.processed_documents[selected_doc]
                
                # Enhanced metrics display
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown('<div class="engineering-metric">Document Type<br/>' + 
                               doc_data.get("document_type", "Unknown") + '</div>', 
                               unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="soil-analysis">Analysis Status<br/>' + 
                               doc_data.get("processing_status", "Unknown") + '</div>', 
                               unsafe_allow_html=True)
                with col3:
                    num_params = sum(len(v) for v in doc_data.get("numerical_data", {}).values())
                    st.markdown('<div class="tunnel-info">Extracted Values<br/>' + 
                               f"{num_params} parameters</div>", 
                               unsafe_allow_html=True)
                with col4:
                    timestamp = doc_data.get("timestamp", "Unknown")[:10] if doc_data.get("timestamp") else "Unknown"
                    st.markdown('<div class="engineering-metric">Processed<br/>' + 
                               timestamp + '</div>', 
                               unsafe_allow_html=True)
                
                st.divider()
                
                # Display extracted numerical data
                numerical_data = doc_data.get("numerical_data", {})
                if numerical_data and any(numerical_data.values()):
                    st.subheader("📐 Extracted Numerical Data")
                    
                    for param_type, values in numerical_data.items():
                        if values:
                            with st.expander(f"📊 {param_type.replace('_', ' ').title()} ({len(values)} values)"):
                                # Create a DataFrame for better display
                                df_data = []
                                for val in values:
                                    row = {
                                        'Value': val['value'],
                                        'Unit': val['unit'],
                                        'Context': val.get('context', '')[:100] + '...' if val.get('context') else ''
                                    }
                                    if 'depth' in val:
                                        row['Depth'] = f"{val['depth']} {val.get('depth_unit', 'm')}"
                                    df_data.append(row)
                                
                                df = pd.DataFrame(df_data)
                                st.dataframe(df, use_container_width=True)
                                
                                # Statistical summary for numerical values
                                if len(values) > 1:
                                    st.write("**Statistical Summary:**")
                                    vals = [v['value'] for v in values]
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Min", f"{min(vals):.2f}")
                                    with col2:
                                        st.metric("Max", f"{max(vals):.2f}")
                                    with col3:
                                        st.metric("Mean", f"{np.mean(vals):.2f}")
                                    with col4:
                                        st.metric("Std Dev", f"{np.std(vals):.2f}")
                
                # Display content analysis
                content = doc_data.get("content", {})
                doc_type = doc_data.get("document_type")
                
                if doc_type == "image" and "response" in content:
                    st.subheader("👁️ AI Vision Analysis")
                    st.write("**🔍 Analysis Query:**")
                    st.info(content.get('query', 'N/A'))
                    st.write("**📋 AI Analysis:**")
                    st.write(content.get('response', 'N/A'))
                    
                    processing_time = content.get('processing_time', 'unknown')
                    if processing_time != 'unknown':
                        st.success(f"⚡ GPU Processing Time: {processing_time}")
                
                # Raw content viewer
                with st.expander("🔍 Raw Document Data"):
                    st.json(doc_data)
        else:
            st.info("📥 Upload and analyze geotechnical documents to see detailed analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="geotechnical-container">', unsafe_allow_html=True)
        st.subheader("📈 Geotechnical Data Visualizations")
        st.caption("Visualizations based on extracted numerical data")
        
        if st.session_state.processed_documents:
            # Allow visualization of ANY document type
            doc_options = list(st.session_state.processed_documents.keys())
            selected_doc = st.selectbox("Select document for visualization:", doc_options, key="viz_doc")
            
            if selected_doc:
                doc_data = st.session_state.processed_documents[selected_doc]
                
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("🎨 Generate Visualization", type="primary"):
                        with st.spinner("Creating geotechnical visualization..."):
                            # Use enhanced visualization module
                            fig = system["visualization"].create_visualization_from_any_document(doc_data)
                            st.plotly_chart(fig, use_container_width=True)
                
                with col1:
                    doc_type = doc_data.get("document_type", "Unknown")
                    num_params = sum(len(v) for v in doc_data.get("numerical_data", {}).values())
                    st.info(f"📊 Document Type: {doc_type} | Extracted Values: {num_params}")
                
                # Auto-generate visualization
                if doc_data.get("numerical_data") and any(doc_data["numerical_data"].values()):
                    with st.spinner("Preparing visualization..."):
                        fig = system["visualization"].create_visualization_from_any_document(doc_data)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ No numerical data extracted yet. Upload documents with geotechnical data.")
        else:
            st.info("📥 Upload geotechnical documents to create intelligent visualizations")
            st.write("**Supported visualizations:**")
            st.write("• 📊 CSV/Excel: Statistical analysis, correlation matrices, depth profiles")
            st.write("• 🖼️ Images: Parameter extraction charts, measurement displays")  
            st.write("• 📄 PDFs: Numerical data visualization, parameter distributions")
            st.write("• 📋 JSON: Data structure visualization")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="geotechnical-container">', unsafe_allow_html=True)
        st.subheader("🚀 System Status & Performance")
        
        # System status overview
        config = system["config"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**🔧 Configuration Status:**")
            config_status = "✅ Configured" if config.api_key and config.endpoint_id else "❌ Not Configured"
            st.write(f"Status: {config_status}")
            st.write(f"API Key: {'***' + config.api_key[-4:] if config.api_key else 'Not set'}")
            st.write(f"Endpoint: {config.endpoint_id if config.endpoint_id else 'Not set'}")
        
        with col2:
            st.write("**🏗️ Geotechnical Capabilities:**")
            if st.button("🔄 Test AI System"):
                with st.spinner("Testing geotechnical AI capabilities..."):
                    health = system["runpod_client"].health_check()
                    if health["status"] == "healthy":
                        st.success("✅ AI vision analysis ready!")
                        st.success("✅ Soil analysis agent active")
                        st.success("✅ Tunnel engineering agent active")
                        st.success("✅ Safety checklist generator ready")
                    else:
                        st.error(f"❌ System issue: {health.get('error', 'Unknown error')}")
        
        st.divider()
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if runpod_status["status"] == "healthy":
                st.markdown('<div class="engineering-metric">🚀 SmolVLM<br/>Ready</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="engineering-metric" style="background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);">❌ SmolVLM<br/>Error</div>', unsafe_allow_html=True)
        
        with col2:
            if system["orchestrator"].agents:
                st.markdown('<div class="soil-analysis">✅ Agents<br/>Active</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="soil-analysis" style="background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);">❌ Agents<br/>Error</div>', unsafe_allow_html=True)
        
        with col3:
            doc_count = len(st.session_state.processed_documents)
            st.markdown(f'<div class="tunnel-info">📄 Documents<br/>{doc_count}</div>', unsafe_allow_html=True)
        
        with col4:
            processing_mode = "GPU Accelerated" if config.api_key else "Local"
            st.markdown(f'<div class="engineering-metric">🌐 Mode<br/>{processing_mode}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown('<div class="geotechnical-container">', unsafe_allow_html=True)
        st.subheader("🔧 Application Settings")
        
        st.write("**🏗️ Geotechnical Engineering Workflow Configuration**")
        
        # Data management
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.success("Chat history cleared!")
                st.rerun()
        
        with col2:
            if st.button("📂 Clear Documents"):
                st.session_state.processed_documents = {}
                st.success("Documents cleared!")
                st.rerun()
        
        with col3:
            if st.button("⏳ Clear Async Jobs"):
                st.session_state.async_jobs = {}
                st.success("Async jobs cleared!")
                st.rerun()
        
        st.divider()
        
        # System information
        st.write("**📋 System Information:**")
        st.write("• **Domain**: Geotechnical & Tunnel Engineering")
        st.write("• **AI Models**: SmolVLM-Instruct (Vision), SmolAgent (Reasoning)")
        st.write("• **Infrastructure**: RunPod Serverless GPU")
        st.write("• **Specializations**: Soil mechanics, Foundation design, Tunnel engineering, Slope stability")
        st.write("• **Key Features**: VLM-based extraction (superior to OCR), Numerical data analysis, Document-based Q&A")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
