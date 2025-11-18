"""
FarmAI Analytics Platform - Main Streamlit Application
Complete production-ready agricultural AI platform
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Import configuration
try:
    from config import (
        PAGE_ICON, PAGE_TITLE, PAGE_LAYOUT, SUPPORTED_CROPS,
        SUPPORTED_LANGUAGES, DATABASE_PATH, MODEL_PATH, GOOGLE_API_KEY
    )
    from src.database_manager import FarmAIDatabaseManager
    from src.crop_classifier import CropDiseaseClassifier
    from src.chatbot_agent import FarmAIChatbot
    from src.analytics_engine import AnalyticsEngine
except ImportError as e:
    st.error(f"⚠️ Import Error: {str(e)}")
    st.stop()

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== INITIALIZE COMPONENTS ====================
@st.cache_resource
def initialize_components():
    """Initialize all platform components"""
    try:
        db = FarmAIDatabaseManager(DATABASE_PATH)
        
        # Check if model exists
        if not Path(MODEL_PATH).exists():
            st.warning(f"⚠️ Model file not found at {MODEL_PATH}. Using demo mode.")
            classifier = None
        else:
            classifier = CropDiseaseClassifier(MODEL_PATH)
        
        # Check if API key exists
        if not GOOGLE_API_KEY:
            st.warning("⚠️ GOOGLE_API_KEY not found. Chatbot will be limited.")
            chatbot = None
        else:
            chatbot = FarmAIChatbot(GOOGLE_API_KEY)
        
        analytics = AnalyticsEngine(db)
        
        return db, classifier, chatbot, analytics
    
    except Exception as e:
        st.error(f"❌ Initialization Error: {str(e)}")
        return None, None, None, None

# Initialize components
db, classifier, chatbot, analytics = initialize_components()

# ==================== SESSION STATE ====================
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'farmer_id' not in st.session_state:
    st.session_state.farmer_id = f"farmer_{int(time.time())}"

if 'current_disease' not in st.session_state:
    st.session_state.current_disease = None

if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# ==================== HELPER FUNCTIONS ====================
def format_disease_name(disease_name: str) -> str:
    """Format disease name for display"""
    return disease_name.replace('___', ': ').replace('_', ' ').title()

def get_severity_color(severity: str) -> str:
    """Get color code for severity level"""
    colors = {
        'High': '#dc3545',
        'Medium': '#ffc107',
        'Low': '#28a745'
    }
    return colors.get(severity, '#6c757d')

# ==================== MAIN UI ====================
st.markdown(f'<h1 class="main-header">{PAGE_ICON} {PAGE_TITLE}</h1>', unsafe_allow_html=True)
st.markdown("**🌾 Advanced AI-Powered Agricultural Assistant with Real-Time Analytics**")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("🔧 Navigation")
    
    page = st.radio(
        "Select Feature",
        ["🏠 Dashboard", "🤖 AI Chatbot", "📸 Disease Detection", 
         "📊 Analytics", "👨‍🌾 Farmer Profile", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Farmer info
    st.subheader("👨‍🌾 Current User")
    st.caption(f"ID: {st.session_state.farmer_id[:12]}...")
    
    # Quick stats
    if db:
        try:
            summary = db.get_analytics_summary()
            st.metric("Your Queries", summary.get('total_queries', 0))
            st.metric("Platform Users", summary.get('total_farmers', 0))
        except:
            pass
    
    st.divider()
    st.caption("Made with ❤️ for Farmers")

# ==================== PAGE 1: DASHBOARD ====================
if page == "🏠 Dashboard":
    st.header("📊 Platform Dashboard")
    
    if not db:
        st.error("Database not initialized")
        st.stop()
    
    # Get metrics
    try:
        metrics = analytics.get_dashboard_metrics()
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Queries", 
                f"{metrics['total_queries']:,}",
                delta="+12 today",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                "Active Farmers", 
                f"{metrics['total_farmers']:,}",
                delta="+3 this week"
            )
        
        with col3:
            st.metric(
                "Avg Response Time", 
                f"{metrics['avg_response_time']:.2f}s",
                delta="-0.3s",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                "Model Accuracy", 
                f"{metrics['model_accuracy']:.1f}%",
                delta="+2.1%"
            )
        
        st.divider()
        
        # Trends Section
        st.subheader("📈 30-Day Trends")
        
        trends = analytics.get_trend_analysis(days=30)
        
        if trends and trends.get('total_queries_trend'):
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.line(
                    x=trends['dates'],
                    y=trends['total_queries_trend'],
                    title="Daily Queries",
                    labels={'x': 'Date', 'y': 'Number of Queries'}
                )
                fig.update_traces(line_color='#2E7D32', line_width=3)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.line(
                    x=trends['dates'],
                    y=trends['avg_response_time_trend'],
                    title="Average Response Time",
                    labels={'x': 'Date', 'y': 'Response Time (seconds)'}
                )
                fig.update_traces(line_color='#1976D2', line_width=3)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 No trend data available yet. Start using the platform to see analytics!")
        
        st.divider()
        
        # Disease Distribution
        st.subheader("🦠 Top Detected Diseases")
        
        disease_dist = analytics.get_disease_distribution()
        
        if not disease_dist.empty:
            fig = px.bar(
                disease_dist,
                x='disease_name',
                y='count',
                title="Disease Detection Frequency",
                labels={'disease_name': 'Disease', 'count': 'Detections'},
                color='count',
                color_continuous_scale='Greens'
            )
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Table view
            with st.expander("📋 View Detailed Statistics"):
                st.dataframe(disease_dist, use_container_width=True)
        else:
            st.info("🔍 No disease data available yet. Upload images to start detecting diseases!")
        
    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")

# ==================== PAGE 2: AI CHATBOT ====================
elif page == "🤖 AI Chatbot":
    st.header("💬 AI Agricultural Assistant")
    
    if not chatbot:
        st.error("⚠️ Chatbot not available. Please configure GOOGLE_API_KEY in environment.")
        st.stop()
    
    # Configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        crop_type = st.selectbox("🌱 Select Your Crop", SUPPORTED_CROPS)
    
    with col2:
        language = st.selectbox("🗣️ Language", SUPPORTED_LANGUAGES)
    
    with col3:
        if st.session_state.current_disease:
            st.info(f"🔬 Context: {format_disease_name(st.session_state.current_disease)}")
    
    st.divider()
    
    # Chat Interface
    user_input = st.text_area(
        "💬 Ask Your Question:",
        height=120,
        placeholder="Example: Why are my tomato leaves turning yellow? What should I do?"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        ask_button = st.button("🚀 Get AI Answer", use_container_width=True, type="primary")
    
    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()
    
    if ask_button and user_input:
        with st.spinner("🤖 AI is thinking..."):
            try:
                response, response_time = chatbot.generate_response(
                    user_input,
                    crop_type,
                    disease_name=st.session_state.current_disease,
                    language=language
                )
                
                # Display response
                st.success(f"✅ Response generated in {response_time:.2f}s")
                
                with st.container():
                    st.markdown("### 💡 AI Response:")
                    st.markdown(response)
                
                # Save to database
                if db:
                    db.save_query(
                        st.session_state.farmer_id,
                        user_input,
                        response,
                        response_time,
                        language=language
                    )
                    
                    db.log_chatbot_interaction(
                        st.session_state.farmer_id,
                        user_input,
                        response,
                        language,
                        response_time
                    )
                
                # Add to session history
                st.session_state.chat_history.append({
                    'query': user_input,
                    'response': response,
                    'time': response_time,
                    'crop': crop_type,
                    'language': language,
                    'timestamp': datetime.now()
                })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Chat History
    if st.session_state.chat_history:
        st.divider()
        st.subheader("📜 Recent Conversations")
        
        for i, msg in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.expander(
                f"🗨️ Q{len(st.session_state.chat_history)-i}: {msg['query'][:60]}...",
                expanded=(i == 0)
            ):
                st.markdown(f"**🌱 Crop:** {msg['crop']}")
                st.markdown(f"**📅 Time:** {msg['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                st.markdown(f"**⏱️ Response Time:** {msg['time']:.2f}s")
                st.markdown("---")
                st.markdown(f"**💬 Question:**\n{msg['query']}")
                st.markdown(f"**💡 Answer:**\n{msg['response']}")

# ==================== PAGE 3: DISEASE DETECTION ====================
elif page == "📸 Disease Detection":
    st.header("🔍 Crop Disease Detection")
    
    if not classifier:
        st.warning("⚠️ Model not loaded. Please ensure crop_disease_model.h5 exists in models/ folder")
        st.info("You can still test the UI - predictions will be simulated")
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        crop_type = st.selectbox("🌱 Select Crop Type", SUPPORTED_CROPS)
    
    with col2:
        uploaded_file = st.file_uploader(
            "📁 Upload Crop Leaf Image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear photo of the affected crop leaf"
        )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            st.markdown("### 📋 Image Information")
            st.write(f"**Format:** {image.format}")
            st.write(f"**Size:** {image.size}")
            st.write(f"**Mode:** {image.mode}")
        
        with col2:
            if st.button("🔬 Analyze Disease", use_container_width=True, type="primary"):
                with st.spinner("🔍 Analyzing image..."):
                    try:
                        if classifier:
                            prediction = classifier.predict(image)
                        else:
                            # Simulation mode
                            import random
                            prediction = {
                                'disease': random.choice(['Tomato Early Blight', 'Potato Late Blight', 'Healthy']),
                                'confidence': random.uniform(0.75, 0.98),
                                'confidence_percentage': random.uniform(75, 98),
                                'treatment': 'Apply fungicide treatment',
                                'severity': random.choice(['High', 'Medium', 'Low']),
                                'prediction_time': random.uniform(0.5, 2.0),
                                'model_version': 'Demo v1.0'
                            }
                        
                        if 'error' not in prediction:
                            st.success("✅ Analysis Complete!")
                            
                            # Store in session
                            st.session_state.prediction_result = prediction
                            st.session_state.current_disease = prediction['disease']
                            
                            # Display results
                            st.markdown("### 🎯 Detection Results")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Disease", format_disease_name(prediction['disease']))
                            
                            with col2:
                                confidence_pct = prediction.get('confidence_percentage', prediction['confidence'] * 100)
                                st.metric("Confidence", f"{confidence_pct:.1f}%")
                            
                            with col3:
                                severity = prediction.get('severity', 'Medium')
                                severity_color = get_severity_color(severity)
                                st.markdown(f"**Severity:** <span style='color:{severity_color};font-weight:bold;font-size:1.2em'>{severity}</span>", unsafe_allow_html=True)
                            
                            # Treatment recommendations
                            st.markdown("### 💊 Treatment Recommendations")
                            st.info(prediction.get('treatment', 'Consult agricultural expert'))
                            
                            # Save to database
                            if db:
                                db.save_prediction(
                                    st.session_state.farmer_id,
                                    predicted_disease_id=1,  # Would map disease name to ID
                                    confidence=prediction['confidence'],
                                    model_version=prediction.get('model_version', '1.0'),
                                    prediction_time=prediction.get('prediction_time', 0),
                                    image_file=uploaded_file.name
                                )
                            
                            # Quick action button
                            if st.button("💬 Ask AI About This Disease", use_container_width=True):
                                st.session_state.page = "🤖 AI Chatbot"
                                st.rerun()
                        
                        else:
                            st.error(f"❌ Error: {prediction['error']}")
                    
                    except Exception as e:
                        st.error(f"❌ Prediction Error: {str(e)}")

# ==================== PAGE 4: ANALYTICS ====================
elif page == "📊 Analytics":
    st.header("📈 Advanced Analytics Dashboard")
    
    if not db:
        st.error("Database not available")
        st.stop()
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Trends", "🦠 Diseases", "💾 Export"])
    
    with tab1:
        st.subheader("Key Performance Indicators")
        
        metrics = analytics.get_dashboard_metrics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Usage Metrics")
            data = {
                'Metric': ['Total Queries', 'Active Farmers', 'Total Predictions', 'Avg Confidence'],
                'Value': [
                    metrics['total_queries'],
                    metrics['total_farmers'],
                    metrics['total_predictions'],
                    f"{metrics.get('avg_confidence', 0):.2%}"
                ]
            }
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### ⚡ Performance Metrics")
            data = {
                'Metric': ['Avg Response Time', 'Model Accuracy', 'Top Disease', 'Platform Status'],
                'Value': [
                    f"{metrics['avg_response_time']}s",
                    f"{metrics['model_accuracy']}%",
                    metrics['top_disease'],
                    '🟢 Active'
                ]
            }
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("📈 Time Series Analysis")
        
        days = st.slider("Select Time Period (days)", 7, 90, 30)
        trends = analytics.get_trend_analysis(days)
        
        if trends and trends.get('total_queries_trend'):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trends['dates'],
                y=trends['total_queries_trend'],
                mode='lines+markers',
                name='Queries',
                line=dict(color='#2E7D32', width=3)
            ))
            fig.update_layout(
                title=f"Query Trends - Last {days} Days",
                xaxis_title="Date",
                yaxis_title="Number of Queries",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trend data available yet")
    
    with tab3:
        st.subheader("🦠 Disease Analytics")
        
        disease_dist = analytics.get_disease_distribution(limit=15)
        
        if not disease_dist.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    disease_dist,
                    values='count',
                    names='disease_name',
                    title="Disease Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    disease_dist.head(10),
                    x='count',
                    y='disease_name',
                    orientation='h',
                    title="Top 10 Diseases",
                    color='avg_confidence',
                    color_continuous_scale='Greens'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No disease data available")
    
    with tab4:
        st.subheader("💾 Export Data for Power BI / Tableau")
        
        st.markdown("""
        Export your analytics data for advanced visualization in:
        - **Power BI Desktop**
        - **Tableau Public**
        - **Excel Analysis**
        """)
        
        if st.button("📥 Export Analytics Data", type="primary", use_container_width=True):
            try:
                filepath = db.export_analytics_data()
                st.success(f"✅ Data exported successfully to: {filepath}")
                
                with open(filepath, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download CSV File",
                        data=f,
                        file_name="farmai_analytics_export.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Export failed: {str(e)}")

# ==================== PAGE 5: FARMER PROFILE ====================
elif page == "👨‍🌾 Farmer Profile":
    st.header("👨‍🌾 Farmer Profile & History")
    
    farmer_id = st.session_state.farmer_id
    
    # Farmer info form
    with st.form("farmer_info"):
        st.subheader("📝 Update Your Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name")
            phone = st.text_input("Phone Number")
            village = st.text_input("Village")
        
        with col2:
            district = st.text_input("District")
            state = st.text_input("State")
            farm_size = st.number_input("Farm Size (acres)", min_value=0.0, step=0.5)
        
        crops = st.multiselect("Crops You Grow", SUPPORTED_CROPS)
        
        if st.form_submit_button("💾 Save Profile", use_container_width=True):
            if name and db:
                success = db.add_farmer(
                    farmer_id, name, phone, village, district, 
                    state, ', '.join(crops), farm_size
                )
                if success:
                    st.success("✅ Profile updated successfully!")
            else:
                st.warning("Please enter your name")
    
    st.divider()
    
    # Query history
    if db:
        st.subheader("📜 Your Query History")
        history = db.get_farmer_history(farmer_id, limit=20)
        
        if not history.empty:
            st.dataframe(history, use_container_width=True)
        else:
            st.info("No query history yet. Start using the platform!")

# ==================== PAGE 6: SETTINGS ====================
elif page == "⚙️ Settings":
    st.header("⚙️ Platform Settings")
    
    tab1, tab2, tab3 = st.tabs(["🔧 General", "ℹ️ System Info", "🗄️ Database"])
    
    with tab1:
        st.subheader("General Settings")
        
        st.text_input("Farmer ID (Read-only)", value=st.session_state.farmer_id, disabled=True)
        
        if st.button("🔄 Generate New Farmer ID"):
            st.session_state.farmer_id = f"farmer_{int(time.time())}"
            st.success("New Farmer ID generated!")
            st.rerun()
        
        st.divider()
        
        if st.button("🗑️ Clear All Chat History", type="secondary"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
        
        if st.button("🔄 Reset All Session Data", type="secondary"):
            st.session_state.clear()
            st.success("Session reset! Refresh the page.")
    
    with tab2:
        st.subheader("System Information")
        
        info = {
            "Platform Version": "1.0.0",
            "Model Version": "MobileNetV2 v1.0",
            "Database": DATABASE_PATH,
            "Model Path": MODEL_PATH,
            "Chatbot Engine": "Google Gemini Pro",
            "Status": "🟢 Operational"
        }
        
        for key, value in info.items():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**{key}:**")
            with col2:
                st.write(value)
    
    with tab3:
        st.subheader("Database Management")
        
        if db:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Backup Database", use_container_width=True):
                    if db.backup_database():
                        st.success("✅ Database backed up successfully!")
            
            with col2:
                if st.button("📥 Export Analytics", use_container_width=True):
                    filepath = db.export_analytics_data()
                    st.success(f"✅ Exported to {filepath}")

# ==================== FOOTER ====================
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>🌾 FarmAI Analytics Platform</strong> | Version 1.0.0</p>
    <p>Built with ❤️ for Indian Farmers | 
    <a href="https://github.com/AshishRathodDev/FarmAI-Analytics-Platform">GitHub</a> | 
    <a href="#">Documentation</a></p>
    <p style="font-size: 12px; margin-top: 10px;">
    ⚠️ For educational and demonstration purposes. Always consult agricultural experts for critical decisions.
    </p>
</div>
""", unsafe_allow_html=True)
