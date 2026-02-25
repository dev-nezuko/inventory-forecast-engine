"""
Streamlit Demo Dashboard for Forecast MVP
Run with: streamlit run streamlit_demo.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from pathlib import Path
import json
from datetime import datetime, timedelta

from src.config import LANDING_DIR, FEATURE_DIR

# Page config
st.set_page_config(
    page_title="SKU Forecast MVP Demo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load processed data from pipeline outputs"""
    data = {}
    
    # Raw data
    try:
        data['credit'] = pd.read_parquet(LANDING_DIR / "credit_txn.parquet")
        data['panel'] = pd.read_parquet(LANDING_DIR / "panel.parquet")
        data['linked'] = pd.read_parquet(LANDING_DIR / "linked_panel_credit.parquet")
    except Exception as e:
        st.error(f"Error loading raw data: {e}")
        return None
    
    # Features
    try:
        data['features'] = pd.read_parquet(FEATURE_DIR / "features.parquet")
    except Exception as e:
        st.warning(f"Features not available: {e}")
    
    # Forecasts
    try:
        data['forecast_ensemble'] = pd.read_parquet(FEATURE_DIR / "forecast_ensemble.parquet")
    except Exception as e:
        st.warning(f"Ensemble forecasts not available: {e}")
    
    try:
        data['forecast_deepar'] = pd.read_parquet(FEATURE_DIR / "forecast_deepar.parquet")
    except Exception as e:
        st.warning(f"DeepAR forecasts not available: {e}")
    
    return data

def load_api_index():
    """Load available SKU/Region pairs from forecast files used by the API.
    Prefer ensemble, then LightGBM, then DeepAR as last resort for options.
    """
    try:
        for fname in ["forecast_ensemble.parquet", "forecast_lgb.parquet", "forecast_deepar.parquet"]:
            path = FEATURE_DIR / fname
            if path.exists():
                df = pd.read_parquet(path)
                # Keep only identifiers for options
                cols = [c for c in ["sku", "region"] if c in df.columns]
                if len(cols) == 2:
                    return df[cols].drop_duplicates().reset_index(drop=True)
        return None
    except Exception as e:
        # Non-fatal for the UI; just show a warning and continue.
        st.warning(f"Could not load API options from features: {e}")
        return None

def call_forecast_api(sku, region, horizon_weeks, api_url="http://localhost:8000", index_df=None):
    """Call the forecast API"""
    try:
        response = requests.post(
            f"{api_url}/forecast",
            json={"sku": sku, "region": region, "horizon_weeks": horizon_weeks},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            # Provide friendlier guidance when the selected pair isn't available
            try:
                payload = response.json()
            except Exception:
                payload = {"detail": response.text}

            detail = payload.get("detail", "")
            if response.status_code == 404 and isinstance(detail, str) and "No forecast rows" in detail and index_df is not None:
                if sku in set(index_df["sku"].unique()):
                    valid_regions = sorted(index_df[index_df["sku"] == sku]["region"].unique().tolist())
                    st.error(
                        f"API Error 404: No forecast rows for {sku}/{region}. Valid regions for {sku}: {', '.join(valid_regions)}"
                    )
                else:
                    valid_skus = sorted(index_df["sku"].unique().tolist())
                    st.error(
                        f"API Error 404: SKU '{sku}' not available. Try one of: {', '.join(valid_skus)}"
                    )
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"API Connection Error: {e}")
        return None

def main():
    # Header
    st.markdown('<div class="main-header">📈 SKU Forecast MVP Demo</div>', unsafe_allow_html=True)
    st.markdown("**SKU × region data pipeline and precomputed artifact explorer**")
    
    st.info("Training is not implemented: LightGBM, DeepAR, and PyMC entry points are stubs. Charts show supplied artifacts, not a reproduced model evaluation.")

    # Load data
    with st.spinner("Loading pipeline data..."):
        data = load_data()
    
    if data is None:
        st.error("❌ Failed to load data. Please run the pipeline first.")
        return
    
    # Sidebar
    st.sidebar.header("🎛️ Demo Controls")
    
    # Pipeline Status
    st.sidebar.subheader("Pipeline Status")
    pipeline_steps = [
        ("Data Ingestion", "credit" in data and "panel" in data),
        ("Data Linking", "linked" in data),
        ("Feature Engineering", "features" in data),
        ("Forecast artifact available", "forecast_ensemble" in data or "forecast_deepar" in data),
    ]
    
    for step, status in pipeline_steps:
        icon = "✅" if status else "❌"
        st.sidebar.write(f"{icon} {step}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🔗 Data Linking", 
        "⚙️ Feature Engineering", 
        "🔮 Forecasting", 
        "🚀 Live API Demo"
    ])
    
    # Tab 1: Data Overview
    with tab1:
        st.header("📊 Data Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Credit Card Transactions")
            if 'credit' in data:
                credit_df = data['credit']
                st.write(f"**Records:** {len(credit_df):,}")
                st.write(f"**Date Range:** {credit_df['txn_date'].min()} to {credit_df['txn_date'].max()}")
                st.write(f"**Unique SKUs:** {credit_df['sku'].nunique()}")
                st.write(f"**Unique Customers:** {credit_df['customer_id'].nunique()}")
                
                # Sample data
                st.write("**Sample Records:**")
                st.dataframe(credit_df.head(), use_container_width=True)
                
                # Transaction volume chart
                credit_df['txn_date'] = pd.to_datetime(credit_df['txn_date'])
                daily_volume = credit_df.groupby('txn_date')['quantity'].sum().reset_index()
                fig = px.line(daily_volume, x='txn_date', y='quantity', 
                             title='Daily Transaction Volume')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Consumer Panel")
            if 'panel' in data:
                panel_df = data['panel']
                st.write(f"**Panel Members:** {len(panel_df):,}")
                st.write(f"**Unique Customers:** {panel_df['customer_id'].nunique()}")
                
                # Sample data
                st.write("**Sample Records:**")
                st.dataframe(panel_df.head(), use_container_width=True)
                
                # Demographics charts
                col_a, col_b = st.columns(2)
                with col_a:
                    fig = px.pie(panel_df, names='age_group', title='Age Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_b:
                    fig = px.pie(panel_df, names='region', title='Regional Distribution')
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Data Linking
    with tab2:
        st.header("🔗 Data Linking & Privacy")
        
        if 'linked' in data:
            linked_df = data['linked']
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Linked Records", f"{len(linked_df):,}")
            with col2:
                linkage_rate = len(linked_df) / len(data['credit']) * 100
                st.metric("Linkage Rate", f"{linkage_rate:.1f}%")
            with col3:
                st.metric("Unique Customers", linked_df['cust_hash'].nunique())
            
            st.markdown("""
            <div class="success-box">
            <strong>✅ Privacy Protection:</strong> Customer IDs are hashed using HMAC-SHA256 
            with a secret salt, ensuring customer privacy while enabling data linkage.
            </div>
            """, unsafe_allow_html=True)
            
            # Show linked data sample
            st.subheader("Linked Dataset Sample")
            display_cols = ['sku', 'region', 'txn_date', 'quantity', 'amount', 'age_group', 'income_bin']
            available_cols = [col for col in display_cols if col in linked_df.columns]
            st.dataframe(linked_df[available_cols].head(10), use_container_width=True)
            
            # Linkage quality chart
            st.subheader("Linkage Quality by SKU")
            sku_linkage = linked_df.groupby('sku').size().reset_index(name='linked_records')
            fig = px.bar(sku_linkage, x='sku', y='linked_records', 
                        title='Linked Records by SKU')
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Feature Engineering
    with tab3:
        st.header("⚙️ Feature Engineering")
        
        if 'features' in data:
            features_df = data['features']
            
            # Feature summary
            st.subheader("Feature Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Time Series", f"{len(features_df):,} weeks")
            with col2:
                st.metric("SKU × Region Combinations", 
                         features_df[['sku', 'region']].drop_duplicates().shape[0])
            with col3:
                date_range = (pd.to_datetime(features_df['week'].max()) - 
                             pd.to_datetime(features_df['week'].min())).days
                st.metric("Date Range", f"{date_range} days")
            
            # Feature descriptions
            st.subheader("Engineered Features")
            feature_descriptions = {
                'units_sold': 'Target variable - weekly units sold',
                'sales_dollars': 'Weekly revenue in dollars',
                'n_transactions': 'Number of transactions per week',
                'avg_price': 'Average price per unit',
                'lag_1_units': 'Previous week units sold (lag-1)',
                'rolling_4w_mean': '4-week rolling average of units sold',
                'weekofyear': 'Week of year (seasonality)',
                'month': 'Month (seasonality)'
            }
            
            for feature, description in feature_descriptions.items():
                if feature in features_df.columns:
                    st.write(f"**{feature}:** {description}")
            
            # Time series visualization
            st.subheader("Time Series by SKU × Region")
            
            # SKU selector
            available_skus = features_df['sku'].unique()
            selected_sku = st.selectbox("Select SKU:", available_skus)
            
            sku_data = features_df[features_df['sku'] == selected_sku].copy()
            sku_data['week'] = pd.to_datetime(sku_data['week'])
            
            fig = make_subplots(rows=2, cols=1, 
                               subplot_titles=['Units Sold', 'Features'],
                               vertical_spacing=0.1)
            
            # Plot by region
            for region in sku_data['region'].unique():
                region_data = sku_data[sku_data['region'] == region]
                
                # Units sold
                fig.add_trace(
                    go.Scatter(x=region_data['week'], y=region_data['units_sold'],
                              name=f'{region} - Units', line=dict(width=3)),
                    row=1, col=1
                )
                
                # Rolling mean
                fig.add_trace(
                    go.Scatter(x=region_data['week'], y=region_data['rolling_4w_mean'],
                              name=f'{region} - 4W Avg', line=dict(dash='dash')),
                    row=2, col=1
                )
            
            fig.update_layout(height=600, title=f"Time Series for {selected_sku}")
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Forecasting
    with tab4:
        st.header("🔮 Forecasting Results")
        
        forecast_data = None
        if 'forecast_ensemble' in data:
            forecast_data = data['forecast_ensemble']
            st.success("✅ Using Ensemble Forecasts")
        elif 'forecast_deepar' in data:
            forecast_data = data['forecast_deepar']
            st.info("ℹ️ Using DeepAR/Naive Forecasts")
        
        if forecast_data is not None:
            # Forecast summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Forecast Horizon", f"{len(forecast_data) // forecast_data[['sku', 'region']].drop_duplicates().shape[0]} weeks")
            with col2:
                st.metric("SKU × Region Series", forecast_data[['sku', 'region']].drop_duplicates().shape[0])
            with col3:
                median_forecast = forecast_data['q0.5'].mean()
                st.metric("Avg Weekly Forecast", f"{median_forecast:.1f} units")
            
            # Forecast visualization
            st.subheader("Forecast Visualization")
            
            # SKU and region selectors
            col1, col2 = st.columns(2)
            with col1:
                forecast_skus = forecast_data['sku'].unique()
                selected_forecast_sku = st.selectbox("Select SKU:", forecast_skus, key="forecast_sku")
            with col2:
                available_regions = forecast_data[forecast_data['sku'] == selected_forecast_sku]['region'].unique()
                selected_region = st.selectbox("Select Region:", available_regions)
            
            # Filter and plot
            plot_data = forecast_data[
                (forecast_data['sku'] == selected_forecast_sku) & 
                (forecast_data['region'] == selected_region)
            ].copy()
            plot_data['date'] = pd.to_datetime(plot_data['date'])
            plot_data = plot_data.sort_values('date')
            
            fig = go.Figure()
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=plot_data['date'], y=plot_data['q0.9'],
                fill=None, mode='lines', line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=plot_data['date'], y=plot_data['q0.1'],
                fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
                name='80% Confidence Interval', fillcolor='rgba(31,119,180,0.2)'
            ))
            
            # Median forecast
            fig.add_trace(go.Scatter(
                x=plot_data['date'], y=plot_data['q0.5'],
                mode='lines+markers', name='Median Forecast',
                line=dict(color='#1f77b4', width=3)
            ))
            
            fig.update_layout(
                title=f"Forecast: {selected_forecast_sku} in {selected_region}",
                xaxis_title="Date",
                yaxis_title="Units Sold",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast table
            st.subheader("Forecast Table")
            display_forecast = plot_data[['date', 'q0.1', 'q0.5', 'q0.9']].copy()
            display_forecast['date'] = display_forecast['date'].dt.strftime('%Y-%m-%d')
            display_forecast = display_forecast.rename(columns={
                'q0.1': '10th Percentile',
                'q0.5': 'Median',
                'q0.9': '90th Percentile'
            })
            st.dataframe(display_forecast, use_container_width=True)
    
    # Tab 5: Live API Demo
    with tab5:
        st.header("🚀 Live API Demo")
        
        st.markdown("""
        This section demonstrates the live forecasting API that can be integrated 
        into production systems for real-time forecasting.
        """)
        
        # API configuration
        col1, col2 = st.columns([2, 1])
        with col1:
            api_url = st.text_input("API URL:", "http://localhost:8000")
        with col2:
            st.write("")
            st.write("")
            # Test API connection
            try:
                health_response = requests.get(f"{api_url}/health", timeout=5)
                if health_response.status_code == 200:
                    st.success("✅ API Connected")
                else:
                    st.error("❌ API Error")
            except:
                st.error("❌ API Offline")
        
        # API request form
        st.subheader("Forecast Request")
        col1, col2, col3 = st.columns(3)

        # Dynamically populate valid SKU/Region pairs from forecast files
        api_index = load_api_index()
        available_skus = sorted(api_index["sku"].unique()) if api_index is not None else ["SKU123"]
        available_regions = sorted(api_index["region"].unique()) if api_index is not None else ["NE"]

        with col1:
            api_sku = st.selectbox("SKU:", available_skus, key="api_sku")
        with col2:
            if api_index is not None:
                regions_for_sku = sorted(api_index[api_index["sku"] == api_sku]["region"].unique())
                # Fallback in unlikely case of no regions for selected sku
                regions_for_sku = regions_for_sku or available_regions
            else:
                regions_for_sku = available_regions
            api_region = st.selectbox("Region:", regions_for_sku, key="api_region")
        with col3:
            api_horizon = st.slider("Forecast Weeks:", 1, 12, 8)
        
        if st.button("🔮 Get Forecast", type="primary"):
            with st.spinner("Calling forecast API..."):
                forecast_result = call_forecast_api(api_sku, api_region, api_horizon, api_url, index_df=api_index)
            
            if forecast_result:
                st.success(f"✅ Retrieved {len(forecast_result)} forecast points")
                
                # Convert to DataFrame for visualization
                forecast_df = pd.DataFrame(forecast_result)
                forecast_df['date'] = pd.to_datetime(forecast_df['date'])
                
                # Plot API results
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=forecast_df['date'], y=forecast_df['q0.9'],
                    fill=None, mode='lines', line_color='rgba(0,0,0,0)',
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_df['date'], y=forecast_df['q0.1'],
                    fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
                    name='80% Confidence Interval', fillcolor='rgba(31,119,180,0.2)'
                ))
                
                # Median forecast
                fig.add_trace(go.Scatter(
                    x=forecast_df['date'], y=forecast_df['q0.5'],
                    mode='lines+markers', name='Median Forecast',
                    line=dict(color='#1f77b4', width=3)
                ))
                
                fig.update_layout(
                    title=f"API Forecast: {api_sku} in {api_region}",
                    xaxis_title="Date",
                    yaxis_title="Units Sold",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show raw JSON response
                with st.expander("📋 Raw API Response"):
                    st.json(forecast_result)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <strong>SKU Forecast MVP</strong> 
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
