import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Demand Intelligence", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Times New Roman', Times, serif;
    }
    
    /* Blue rectangular border around all pages (Increased Width) */
    .block-container {
        border: 6px solid #1D4ED8;
        border-radius: 12px;
        padding: 3rem !important;
        margin-top: 5rem !important;
        margin-bottom: 2rem !important;
    }
    
    /* Metric Card Styling */
    div[data-testid="metric-container"] {
        background-color: rgba(128, 128, 128, 0.1);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 5% 5% 5% 10%;
        border-radius: 12px;
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        transform: scale(1.02);
    }
    
    /* Primary Button Styling (Matches Dark Mode Success Box) */
    .stButton>button {
        background-color: rgba(25, 60, 38, 0.8) !important;
        color: #5AE07D !important;
        border-radius: 8px;
        border: 1px solid rgba(25, 60, 38, 1) !important;
        font-weight: bold !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: rgba(35, 80, 50, 0.9) !important;
        box-shadow: 0 4px 12px rgba(25, 60, 38, 0.4);
        transform: translateY(-2px);
    }
    .stButton>button p {
        color: #5AE07D !important;
    }
    
    /* Headings */
    h1 {
        background: -webkit-linear-gradient(45deg, #1D4ED8, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold !important;
    }
    h2, h3 {
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to bold plotly layouts
def apply_bold_layout(fig, title_text):
    fig.update_layout(
        title=f"<b>{title_text}</b>",
        font=dict(family="Times New Roman", size=14),
        hoverlabel=dict(font=dict(family="Times New Roman", size=14))
    )
    
    # Wrap existing axis titles in bold tags if they exist
    if fig.layout.xaxis.title.text:
        # Avoid double bolding
        if not fig.layout.xaxis.title.text.startswith("<b>"):
            fig.layout.xaxis.title.text = f"<b>{fig.layout.xaxis.title.text}</b>"
    if fig.layout.yaxis.title.text:
        if not fig.layout.yaxis.title.text.startswith("<b>"):
            fig.layout.yaxis.title.text = f"<b>{fig.layout.yaxis.title.text}</b>"
            
    fig.update_xaxes(title_font=dict(family="Times New Roman", size=15), tickfont=dict(family="Times New Roman", size=12))
    fig.update_yaxes(title_font=dict(family="Times New Roman", size=15), tickfont=dict(family="Times New Roman", size=12))
    return fig

# --- SESSION STATE INITIALIZATION ---
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

def navigate_to(page_name):
    st.session_state.page = page_name

# --- DATA LOADING ---
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df = df.sort_values('Order Date')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset not found. Please ensure 'train.csv' is in the same directory.")
    st.stop()

# ==========================================
# PAGE 0: HOME / NAVIGATION DASHBOARD
# ==========================================
if st.session_state.page == 'Home':
    st.title("Demand Intelligence System")
    st.markdown("""
    Welcome to the **End-to-End Sales Forecasting & Demand Intelligence System**. 
    This dashboard is designed to help supply chain managers, analysts, and business owners make data-driven decisions about inventory and sales strategies.
    
    Click on any of the features below to start exploring!
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 1. Sales Overview")
        st.write("Understand historical sales, revenue growth, and category performance at a glance. Perfect for getting a baseline understanding of how the business is doing.")
        if st.button("Explore Sales Overview", key="btn_sales", use_container_width=True):
            navigate_to('Sales Overview')
            st.rerun()
            
        st.markdown("<br><br>", unsafe_allow_html=True)
            
        st.markdown("### 3. Anomaly Report")
        st.write("Automatically detect unusual spikes or drops in sales volume using AI. Essential for identifying missed opportunities or sudden supply chain disruptions.")
        if st.button("View Anomaly Report", key="btn_anomaly", use_container_width=True):
            navigate_to('Anomaly Report')
            st.rerun()
            
    with col2:
        st.markdown("### 2. Forecast Explorer")
        st.write("Predict future demand for specific regions and categories using Machine Learning (Facebook Prophet). Use this to prevent overstocking or stockouts.")
        if st.button("Generate Forecasts", key="btn_forecast", use_container_width=True):
            navigate_to('Forecast Explorer')
            st.rerun()
            
        st.markdown("<br><br>", unsafe_allow_html=True)
            
        st.markdown("### 4. Product Segments")
        st.write("Group products into strategic clusters (like 'Stable High Volume' or 'High Volatility') to tailor your inventory stocking strategies effectively.")
        if st.button("Analyze Product Segments", key="btn_segments", use_container_width=True):
            navigate_to('Product Segments')
            st.rerun()

# ==========================================
# PAGE 1: SALES OVERVIEW
# ==========================================
elif st.session_state.page == 'Sales Overview':
    if st.button("Back to Home"):
        navigate_to('Home')
        st.rerun()
        
    st.title("Sales Overview Dashboard")
    
    # Top KPI Metrics
    total_revenue = df['Sales'].sum()
    total_orders = df.shape[0]
    avg_order_val = total_revenue / total_orders
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    col_kpi1.metric("Total Revenue (All Time)", f"${total_revenue:,.0f}")
    col_kpi2.metric("Total Orders", f"{total_orders:,}")
    col_kpi3.metric("Average Order Value", f"${avg_order_val:,.2f}")
    
    st.info("**What this means:** These Key Performance Indicators (KPIs) give you a quick snapshot of the business's overall health. A high total revenue paired with a healthy average order value indicates strong consumer spending.")
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Revenue by Year")
        yearly_sales = df.groupby('Year')['Sales'].sum().reset_index()
        fig_year = px.bar(yearly_sales, x='Year', y='Sales', text_auto='.2s', 
                          color='Sales', color_continuous_scale='Greens')
        fig_year = apply_bold_layout(fig_year, "Annual Revenue Growth")
        fig_year.update_layout(margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
        fig_year.update_traces(textfont=dict(family="Times New Roman", weight="bold"))
        st.plotly_chart(fig_year, use_container_width=True)
        st.success("**Observation:** Notice how the revenue grows consistently year over year. The business is expanding, meaning we need to plan for higher inventory levels each subsequent year.")
        
    with col2:
        st.subheader("Monthly Revenue Trend")
        monthly_sales = df.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
        fig_trend = px.line(monthly_sales, x='Order Date', y='Sales', line_shape='spline')
        fig_trend.update_traces(line_color='#10B981', line_width=3, fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.1)')
        fig_trend = apply_bold_layout(fig_trend, "Monthly Revenue Timeline")
        fig_trend.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_trend, use_container_width=True)
        st.success("**Observation:** This chart reveals strong 'Seasonality'. Look at how sales spike massively at the end of every year (November/December). This means we must heavily stock up in October to prepare for the holiday rush.")

# ==========================================
# PAGE 2: FORECAST EXPLORER
# ==========================================
elif st.session_state.page == 'Forecast Explorer':
    if st.button("Back to Home"):
        navigate_to('Home')
        st.rerun()
        
    st.title("Demand Forecast Explorer")
    st.markdown("Leverage Machine Learning to predict exactly how much inventory we will need.")
    
    st.info("**How to use this:** Select a specific Category (like Furniture) or Region (like West). The AI will look at past sales and predict the future demand for the next few months.")
    
    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])
        segment_type = col1.selectbox("Forecast Level", ["Category", "Region"])
        
        if segment_type == "Category":
            segment_val = col2.selectbox("Select Target", df['Category'].unique())
            segment_col = 'Category'
        else:
            segment_val = col2.selectbox("Select Target", df['Region'].unique())
            segment_col = 'Region'
            
        horizon = col3.slider("Forecast Horizon (Months)", min_value=1, max_value=6, value=3)
        
        generate_btn = st.button("Generate AI Forecast", use_container_width=True)
        
    if generate_btn:
        with st.spinner(f"Training specialized AI model for {segment_val}..."):
            segment_df = df[df[segment_col] == segment_val]
            monthly_seg = segment_df.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
            prophet_df = monthly_seg.rename(columns={'Order Date': 'ds', 'Sales': 'y'})
            
            m_full = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m_full.fit(prophet_df)
            future = m_full.make_future_dataframe(periods=horizon, freq='ME')
            forecast = m_full.predict(future)
            
            st.success("Forecast generation complete!")
            
            # Interactive Plotly Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='<b>Historical Sales (Actual)</b>', line=dict(color='#1D4ED8', width=2)))
            fig.add_trace(go.Scatter(x=forecast['ds'].iloc[-horizon:], y=forecast['yhat'].iloc[-horizon:], name='<b>Future Forecast</b>', line=dict(color='#F59E0B', width=3, dash='dot')))
            fig.add_trace(go.Scatter(x=forecast['ds'].iloc[-horizon:].tolist() + forecast['ds'].iloc[-horizon:][::-1].tolist(),
                                     y=forecast['yhat_upper'].iloc[-horizon:].tolist() + forecast['yhat_lower'].iloc[-horizon:][::-1].tolist(),
                                     fill='toself', fillcolor='rgba(245, 158, 11, 0.2)', line=dict(color='rgba(255,255,255,0)'),
                                     hoverinfo="skip", showlegend=True, name='<b>Confidence Interval</b>'))
            
            fig.update_layout(xaxis_title="Date", yaxis_title="Sales")
            fig = apply_bold_layout(fig, f"Demand Projection: {segment_val}")
            fig.update_layout(hovermode="x unified", legend=dict(font=dict(family="Times New Roman", size=12)))
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("**Observation:** The blue line shows what already happened. The dotted orange line shows what the AI expects to happen next. The shaded orange area is the 'Confidence Interval'—meaning the AI is fairly certain sales will fall somewhere inside that shaded zone. If the line drops, order less stock; if it rises, order more!")

# ==========================================
# PAGE 3: ANOMALY REPORT
# ==========================================
elif st.session_state.page == 'Anomaly Report':
    if st.button("Back to Home"):
        navigate_to('Home')
        st.rerun()
        
    st.title("Supply Chain Anomaly Detection")
    st.markdown("Automated detection of irregular sales volumes.")
    
    st.info("**What is an Anomaly?** It's a week where sales were either suspiciously high (a massive spike) or suspiciously low (a sudden drop). Finding these helps us investigate if our marketing worked too well, or if we had a supply chain failure (ran out of stock).")
    
    # Process anomalies
    weekly_sales = df.groupby(pd.Grouper(key='Order Date', freq='W-MON'))['Sales'].sum().reset_index()
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    weekly_sales['Anomaly'] = iso_forest.fit_predict(weekly_sales[['Sales']])
    anomalies = weekly_sales[weekly_sales['Anomaly'] == -1]
    
    # Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly_sales['Order Date'], y=weekly_sales['Sales'], name='<b>Normal Sales</b>', line=dict(color='#9CA3AF', width=1)))
    fig.add_trace(go.Scatter(x=anomalies['Order Date'], y=anomalies['Sales'], mode='markers', name='<b>Flagged Anomaly (Outlier)</b>', 
                             marker=dict(color='#EF4444', size=10, symbol='x')))
    
    fig.update_layout(xaxis_title="Date", yaxis_title="Weekly Sales")
    fig = apply_bold_layout(fig, "Weekly Sales with Outliers Highlighted")
    fig.update_layout(hovermode="x unified", legend=dict(font=dict(family="Times New Roman", size=12)))
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("**Observation:** The red 'X' marks highlight weeks that broke the normal pattern. Most of our positive anomalies occur around November due to Black Friday/Holiday sales. Any negative anomalies (drops) mid-year should be investigated by the warehouse team for potential stockouts.")
    
    st.subheader("Critical Anomaly Log")
    log_df = anomalies[['Order Date', 'Sales']].sort_values('Sales', ascending=False)
    log_df['Status'] = ["Surge (Spike)" if s > weekly_sales['Sales'].mean() else "Drop" for s in log_df['Sales']]
    log_df['Order Date'] = log_df['Order Date'].dt.strftime('%Y-%m-%d')
    st.dataframe(log_df.style.format({'Sales': '${:,.2f}'}), use_container_width=True)

# ==========================================
# PAGE 4: PRODUCT SEGMENTS
# ==========================================
elif st.session_state.page == 'Product Segments':
    if st.button("Back to Home"):
        navigate_to('Home')
        st.rerun()
        
    st.title("Strategic Product Segmentation")
    st.markdown("We use **AI Clustering** to group similar product categories together based on their sales behavior.")
    
    st.info("**Why do this?** You shouldn't order pens the same way you order expensive machinery. Grouping products allows us to assign tailored supply chain rules to each group.")
    
    st.markdown("""
    <div style="background-color: rgba(16, 185, 129, 0.1); border: 1px solid #10B981; padding: 15px; border-radius: 8px; color: #047857;">
    <b>Glossary of Terms (How we define the labels):</b><br><br>
    
    <b>Volume:</b> The total quantity of items sold.<br>
    - <i>Low Volume:</i> Items that rarely sell or only sell in small batches.<br>
    - <i>High Volume:</i> Items that sell constantly and in massive quantities.<br><br>
    
    <b>Volatility:</b> How unpredictable the sales pattern is.<br>
    - <i>Stable:</i> Demand is highly predictable and consistent every single month.<br>
    - <i>High Volatility:</i> Demand swings wildly (e.g., zero sales for weeks, then a massive sudden spike). Very hard to predict.<br><br>
    
    <b>Value:</b> The average price or revenue per order.<br>
    - <i>High Value:</i> Very expensive items (like Machinery or Copiers) where a single sale brings in huge revenue.<br><br>
    
    <b>Growth / Trend:</b> The year-over-year sales trajectory.<br>
    - <i>Fast Growth:</i> Sales are increasing rapidly compared to previous years.<br>
    - <i>Declining:</i> Sales are dropping over time.
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Clustering product profiles..."):
        subcat_stats = df.groupby('Sub-Category').agg(Total_Sales=('Sales', 'sum'), AOV=('Sales', 'mean')).reset_index()
        subcat_monthly = df.groupby(['Sub-Category', pd.Grouper(key='Order Date', freq='ME')])['Sales'].sum().reset_index()
        volatility = subcat_monthly.groupby('Sub-Category')['Sales'].std().reset_index().rename(columns={'Sales': 'Volatility'})
        
        subcat_year = df.groupby(['Sub-Category', 'Year'])['Sales'].sum().unstack()
        subcat_year['Growth_Rate'] = (subcat_year[2018] - subcat_year[2017]) / subcat_year[2017]
        growth = subcat_year[['Growth_Rate']].reset_index()
        
        cluster_df = pd.merge(subcat_stats, volatility, on='Sub-Category')
        cluster_df = pd.merge(cluster_df, growth, on='Sub-Category').fillna(0)
        
        features = ['Total_Sales', 'AOV', 'Volatility', 'Growth_Rate']
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_df[features])
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        cluster_df['Cluster'] = kmeans.fit_predict(scaled_data)
        
        pca = PCA(n_components=2)
        cluster_df[['PCA1', 'PCA2']] = pca.fit_transform(scaled_data)
        
        cluster_labels = {
            0: "Fast Growth, Low Volume",
            1: "Stable, High Volume",
            2: "High Volatility, High Value",
            3: "Low Volume / Declining"
        }
        cluster_df['Profile'] = cluster_df['Cluster'].map(cluster_labels)
        
        fig = px.scatter(cluster_df, x='PCA1', y='PCA2', color='Profile', text='Sub-Category', size='Total_Sales',
                         color_discrete_sequence=px.colors.qualitative.Dark2)
        fig.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='Black')), textfont=dict(family="Times New Roman", weight="bold", size=11))
        fig = apply_bold_layout(fig, "Product Demand Matrix (How Products Relate to Each Other)")
        fig.update_layout(height=600, legend=dict(font=dict(family="Times New Roman", size=12)))
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("**Observation:** Products in the same colored group behave similarly. For example, 'Stable, High Volume' items (like Phones and Chairs) need consistent bulk re-ordering. Items in 'High Volatility, High Value' (like Copiers and Machines) sell rarely but for huge amounts, so we should keep flexible safety stock for them.")
        
        st.subheader("Recommended Stocking Strategy")
        st.dataframe(cluster_df[['Sub-Category', 'Profile', 'Total_Sales', 'Volatility', 'Growth_Rate']].sort_values('Profile'), use_container_width=True)
