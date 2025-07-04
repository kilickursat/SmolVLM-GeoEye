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
