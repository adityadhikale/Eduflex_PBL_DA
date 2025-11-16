import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, silhouette_score, davies_bouldin_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')
import io
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Learning Platform - Data Analysis Suite",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        font-weight: bold;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_data(uploaded_file):
    """Load data from uploaded file"""
    try:
        df = pd.read_csv(uploaded_file)
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def generate_synthetic_data(n_samples=1000):
    """Generate synthetic survey data"""
    np.random.seed(42)
    
    # WTP mapping
    wtp_categories = ['$0 - $10', '$11 - $25', '$26 - $50', '$51 - $75', 
                      '$76 - $100', '$101 - $150', 'Above $150']
    wtp_weights = [0.08, 0.15, 0.25, 0.22, 0.15, 0.10, 0.05]
    
    data = {
        'Q1_Age': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], 
                                  n_samples, p=[0.15, 0.35, 0.25, 0.15, 0.10]),
        'Q2_Gender': np.random.choice(['Male', 'Female', 'Non-binary', 'Prefer not to say'], 
                                      n_samples, p=[0.48, 0.48, 0.02, 0.02]),
        'Q8_Income': np.random.choice(['Less than $25,000', '$25,000 - $50,000', '$50,001 - $75,000',
                                     '$75,001 - $100,000', '$100,001 - $150,000', 'Above $150,000'],
                                    n_samples, p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05]),
        'Q22_Current_Spending': np.random.choice(['$0 (Only free resources)', '$1 - $20', '$21 - $50',
                                                  '$51 - $100', '$101 - $200', 'Above $200'],
                                                 n_samples, p=[0.30, 0.30, 0.20, 0.12, 0.05, 0.03]),
        'Q23_Willingness_To_Pay': np.random.choice(wtp_categories, n_samples, p=wtp_weights),
        'Q31_Interest_Level': np.random.choice(['Definitely would subscribe', 'Very likely to subscribe',
                                                'Somewhat interested', 'Might consider', 'Not interested'],
                                               n_samples, p=[0.15, 0.25, 0.30, 0.20, 0.10]),
        'Q9_Learning_Hours': np.random.choice(['0-2 hours', '3-5 hours', '6-10 hours', 
                                               '11-15 hours', '16-20 hours', 'More than 20 hours'],
                                              n_samples, p=[0.20, 0.30, 0.25, 0.15, 0.07, 0.03])
    }
    
    return pd.DataFrame(data)

def convert_wtp_to_numeric(df, wtp_column):
    """Convert WTP to numeric values"""
    wtp_mapping = {
        '$0 - $10': 5,
        '$11 - $25': 18,
        '$26 - $50': 38,
        '$51 - $75': 63,
        '$76 - $100': 88,
        '$101 - $150': 125,
        'Above $150': 175
    }
    
    if wtp_column in df.columns:
        df['WTP_Numeric'] = df[wtp_column].map(wtp_mapping)
    
    return df

def create_features(df):
    """Create engineered features"""
    df_processed = df.copy()
    
    # Age mapping
    age_mapping = {'18-24': 21, '25-34': 29.5, '35-44': 39.5, '45-54': 49.5, '55+': 60}
    if 'Q1_Age' in df.columns:
        df_processed['Age_Numeric'] = df['Q1_Age'].map(age_mapping)
    
    # Income mapping
    income_mapping = {
        'Less than $25,000': 20000,
        '$25,000 - $50,000': 37500,
        '$50,001 - $75,000': 62500,
        '$75,001 - $100,000': 87500,
        '$100,001 - $150,000': 125000,
        'Above $150,000': 175000
    }
    if 'Q8_Income' in df.columns:
        df_processed['Income_Numeric'] = df['Q8_Income'].map(income_mapping)
    
    # Current spending mapping
    spending_mapping = {
        '$0 (Only free resources)': 0,
        '$1 - $20': 10,
        '$21 - $50': 35,
        '$51 - $100': 75,
        '$101 - $200': 150,
        'Above $200': 250
    }
    if 'Q22_Current_Spending' in df.columns:
        df_processed['Current_Spending'] = df['Q22_Current_Spending'].map(spending_mapping)
    
    # Interest level mapping
    interest_mapping = {
        'Definitely would subscribe': 5,
        'Very likely to subscribe': 4,
        'Somewhat interested': 3,
        'Might consider': 2,
        'Not interested': 1
    }
    if 'Q31_Interest_Level' in df.columns:
        df_processed['Interest_Level'] = df['Q31_Interest_Level'].map(interest_mapping)
    
    # Learning hours mapping
    hours_mapping = {
        '0-2 hours': 1,
        '3-5 hours': 4,
        '6-10 hours': 8,
        '11-15 hours': 13,
        '16-20 hours': 18,
        'More than 20 hours': 25
    }
    if 'Q9_Learning_Hours' in df.columns:
        df_processed['Learning_Hours'] = df['Q9_Learning_Hours'].map(hours_mapping)
    
    # Derived features
    if 'Income_Numeric' in df_processed.columns and 'Current_Spending' in df_processed.columns:
        df_processed['Income_Spending_Ratio'] = df_processed['Income_Numeric'] / (df_processed['Current_Spending'] + 1)
    
    if 'Income_Numeric' in df_processed.columns:
        df_processed['High_Earner'] = (df_processed['Income_Numeric'] >= 100000).astype(int)
    
    if 'Interest_Level' in df_processed.columns:
        df_processed['Highly_Interested'] = (df_processed['Interest_Level'] >= 4).astype(int)
    
    return df_processed

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Title
    st.markdown('<h1 class="main-header">🎯 AI Learning Platform - Data Analysis Suite</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 10px; margin-bottom: 2rem;'>
        <h3>Complete Data-Driven Decision Making Platform</h3>
        <p>Analyze customer data, predict willingness to pay, discover patterns, and segment customers</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=100)
        st.title("📊 Navigation")
        
        page = st.radio(
            "Select Analysis Module:",
            ["🏠 Home & Data Upload",
             "📊 Exploratory Data Analysis",
             "🔍 Association Rule Mining",
             "👥 Customer Clustering",
             "💰 Regression Analysis (WTP)",
             "📈 Business Dashboard",
             "📥 Export Reports"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 🎯 Quick Stats")
        
        if 'df' in st.session_state and st.session_state.df is not None:
            st.metric("Total Records", len(st.session_state.df))
            st.metric("Total Features", len(st.session_state.df.columns))
        else:
            st.info("Load data to see stats")
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'df_processed' not in st.session_state:
        st.session_state.df_processed = None
    
    # Route to selected page
    if page == "🏠 Home & Data Upload":
        page_home()
    elif page == "📊 Exploratory Data Analysis":
        page_eda()
    elif page == "🔍 Association Rule Mining":
        page_association_rules()
    elif page == "👥 Customer Clustering":
        page_clustering()
    elif page == "💰 Regression Analysis (WTP)":
        page_regression()
    elif page == "📈 Business Dashboard":
        page_dashboard()
    elif page == "📥 Export Reports":
        page_export()

# ============================================================================
# PAGE 1: HOME & DATA UPLOAD (FIXED - SINGLE VERSION)
# ============================================================================

def page_home():
    st.markdown('<h2 class="sub-header">🏠 Welcome to the Analysis Suite</h2>', 
                unsafe_allow_html=True)
    
    # STATUS INDICATOR
    st.markdown("### 📊 Data Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.df is not None:
            st.success("✅ Data Loaded")
        else:
            st.error("❌ No Data")
    
    with col2:
        if st.session_state.df_processed is not None:
            st.success("✅ Data Processed")
        else:
            st.warning("⚠️ Data Not Processed")
    
    with col3:
        if st.session_state.df is not None:
            st.info(f"📊 {len(st.session_state.df)} rows")
        else:
            st.info("📊 0 rows")
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 What This Application Does
        
        This comprehensive data analysis suite provides:
        
        1. **📊 Exploratory Data Analysis (EDA)**
           - Comprehensive statistical summaries
           - Distribution analysis
           - Correlation heatmaps
           - Interactive visualizations
        
        2. **🔍 Association Rule Mining**
           - Apriori algorithm implementation
           - Market basket analysis
           - Customer behavior patterns
           - Actionable recommendations
        
        3. **👥 Customer Clustering**
           - K-Means, Hierarchical, DBSCAN
           - Customer segmentation
           - Cluster profiling
           - Business personas
        
        4. **💰 Regression Analysis**
           - Willingness to Pay prediction
           - Multiple ML models comparison
           - Feature importance analysis
           - Pricing recommendations
        
        5. **📈 Business Dashboard**
           - Executive summary
           - KPI tracking
           - Visual insights
           - Strategic recommendations
        """)
    
    with col2:
        st.markdown("### 📊 Quick Start")
        st.info("""
        **Step 1:** Upload or generate data ⬇️
        
        **Step 2:** Click "Process Data" button ✅
        
        **Step 3:** Run analyses from sidebar 📊
        
        **Step 4:** Export reports 📥
        """)
    
    st.markdown("---")
    
    # Data Upload Section
    st.markdown("### 📁 STEP 1: Load Data")
    
    tab1, tab2 = st.tabs(["📤 Upload CSV", "🎲 Generate Synthetic Data"])
    
    with tab1:
        st.markdown("**Upload your survey data CSV file**")
        # THIS IS THE WIDGET THAT HAD THE DUPLICATE KEY
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key="csv_upload_main")
        
        if uploaded_file is not None:
            with st.spinner("Loading data..."):
                df, error = load_data(uploaded_file)
                
                if error:
                    st.error(f"Error loading data: {error}")
                else:
                    st.session_state.df = df
                    st.session_state.df_processed = None # Reset processed data on new upload
                    st.success(f"✅ Data loaded successfully! {len(df)} rows, {len(df.columns)} columns")
                    
                    # Preview
                    with st.expander("👀 Preview Data (Click to expand)"):
                        st.dataframe(df.head(10), use_container_width=True)
    
    with tab2:
        st.markdown("**Generate synthetic survey data for testing**")
        
        n_samples = st.slider("Number of samples to generate", 100, 5000, 1000, 100)
        
        if st.button("🎲 Generate Synthetic Data", type="primary"):
            with st.spinner("Generating synthetic data..."):
                df = generate_synthetic_data(n_samples)
                st.session_state.df = df
                st.session_state.df_processed = None # Reset processed data
                
                st.success(f"✅ Generated {n_samples} synthetic records!")
                
                # Preview
                with st.expander("👀 Preview Data (Click to expand)"):
                    st.dataframe(df.head(10), use_container_width=True)
    
    # STEP 2: PROCESS DATA
    st.markdown("---")
    st.markdown("### 🔧 STEP 2: Process Data (REQUIRED for Analysis)")
    
    if st.session_state.df is not None:
        st.info("⚠️ Click the button below to process your data before running any analysis")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🔧 PROCESS DATA NOW", type="primary", use_container_width=True):
                with st.spinner("Processing data and creating features..."):
                    try:
                        # Convert WTP to numeric
                        df_processed = convert_wtp_to_numeric(st.session_state.df.copy(), 'Q23_Willingness_To_Pay')
                        
                        # Create features
                        df_processed = create_features(df_processed)
                        
                        # Store in session state
                        st.session_state.df_processed = df_processed
                        
                        st.success("✅ Data processed successfully!")
                        st.balloons()
                        
                        # Show what was created
                        st.markdown("#### ✅ Created Features:")
                        new_cols = [col for col in df_processed.columns if col not in st.session_state.df.columns]
                        st.write(f"Added {len(new_cols)} new features:")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            for col in new_cols[:(len(new_cols) + 1)//2]: # Handle odd numbers
                                st.write(f"• {col}")
                        with col_b:
                            for col in new_cols[(len(new_cols) + 1)//2:]:
                                st.write(f"• {col}")
                        
                        st.info("🎉 You can now use all analysis features from the sidebar!")
                        
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")
                        st.error("Please check your data format and try again")
        
        # Show current status
        if st.session_state.df_processed is not None:
            st.success("✅ Data is already processed and ready for analysis!")
            
            with st.expander("📊 View Processed Data Summary"):
                st.write(f"**Total Features:** {len(st.session_state.df_processed.columns)}")
                st.write(f"**Numeric Features:** {len(st.session_state.df_processed.select_dtypes(include=[np.number]).columns)}")
                
                # Show sample
                st.dataframe(st.session_state.df_processed.head(5), use_container_width=True)
    else:
        st.warning("⚠️ Please load data first (Step 1 above)")
    
    # Data Info
    if st.session_state.df is not None:
        st.markdown("---")
        st.markdown("### 📋 Dataset Information")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(st.session_state.df))
        with col2:
            st.metric("Total Features", len(st.session_state.df.columns))
        with col3:
            st.metric("Missing Values", st.session_state.df.isnull().sum().sum())
        with col4:
            if st.session_state.df_processed is not None:
                st.metric("Processed Features", len(st.session_state.df_processed.columns))
            else:
                st.metric("Processed Features", "Not yet")


# ============================================================================
# PAGE 2: EXPLORATORY DATA ANALYSIS
# ============================================================================

def page_eda():
    st.markdown('<h2 class="sub-header">📊 Exploratory Data Analysis</h2>', 
                unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload or generate data first!")
        return
    
    df = st.session_state.df
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Statistical Summary", "📊 Distributions", 
                                     "🔗 Correlations", "🎯 Target Analysis"])
    
    # Tab 1: Statistical Summary
    with tab1:
        st.markdown("### 📈 Statistical Summary")
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
        
        # Categorical columns
        st.markdown("### 📋 Categorical Variables")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_cols) > 0:
            selected_cat = st.selectbox("Select categorical variable", categorical_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### Value Counts: {selected_cat}")
                value_counts = df[selected_cat].value_counts()
                st.dataframe(value_counts, use_container_width=True)
            
            with col2:
                st.markdown("#### Distribution")
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                             labels={'x': selected_cat, 'y': 'Count'},
                             title=f'Distribution of {selected_cat}')
                fig.update_traces(marker_color='#667eea')
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Distributions
    with tab2:
        st.markdown("### 📊 Variable Distributions")
        
        # Select variable
        all_cols = df.columns.tolist()
        selected_var = st.selectbox("Select variable to visualize", all_cols, key="dist_select")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            if df[selected_var].dtype in ['int64', 'float64']:
                fig = px.histogram(df, x=selected_var, title=f'Distribution of {selected_var}',
                                   marginal='box', color_discrete_sequence=['#667eea'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                value_counts = df[selected_var].value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                             title=f'Distribution of {selected_var}')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot or count plot
            if df[selected_var].dtype in ['int64', 'float64']:
                fig = px.box(df, y=selected_var, title=f'Box Plot of {selected_var}',
                             color_discrete_sequence=['#764ba2'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                value_counts = df[selected_var].value_counts().head(10)
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                             title=f'Top 10 Categories in {selected_var}',
                             labels={'x': selected_var, 'y': 'Count'},
                             color=value_counts.values,
                             color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Correlations
    with tab3:
        st.markdown("### 🔗 Correlation Analysis")
        
        if st.session_state.df_processed is not None:
            df_proc = st.session_state.df_processed
            numeric_cols = df_proc.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 1:
                corr_matrix = df_proc[numeric_cols].corr()
                
                # Heatmap
                fig = px.imshow(corr_matrix, 
                                labels=dict(color="Correlation"),
                                x=corr_matrix.columns,
                                y=corr_matrix.columns,
                                color_continuous_scale='RdBu_r',
                                aspect="auto",
                                title="Correlation Matrix")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Top correlations with target
                if 'WTP_Numeric' in numeric_cols:
                    st.markdown("### 🎯 Top Features Correlated with WTP")
                    correlations = df_proc[numeric_cols].corrwith(df_proc['WTP_Numeric']).sort_values(ascending=False)
                    
                    fig = px.bar(x=correlations.values, y=correlations.index,
                                 orientation='h',
                                 labels={'x': 'Correlation', 'y': 'Feature'},
                                 title='Feature Correlations with Willingness to Pay',
                                 color=correlations.values,
                                 color_continuous_scale='RdYlGn')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Process data first to see correlations")
    
    # Tab 4: Target Analysis
    with tab4:
        st.markdown("### 🎯 Willingness to Pay Analysis")
        
        wtp_col = 'Q23_Willingness_To_Pay'
        
        if wtp_col in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution
                value_counts = df[wtp_col].value_counts().sort_index()
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                             labels={'x': 'Willingness to Pay', 'y': 'Count'},
                             title='WTP Distribution',
                             color=value_counts.values,
                             color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Pie chart
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                             title='WTP Percentage Distribution',
                             color_discrete_sequence=px.colors.sequential.Viridis)
                st.plotly_chart(fig, use_container_width=True)
            
            # Numeric analysis
            if st.session_state.df_processed is not None and 'WTP_Numeric' in st.session_state.df_processed.columns:
                st.markdown("### 📊 Numeric WTP Statistics")
                
                wtp_numeric = st.session_state.df_processed['WTP_Numeric']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"${wtp_numeric.mean():.2f}")
                with col2:
                    st.metric("Median", f"${wtp_numeric.median():.2f}")
                with col3:
                    st.metric("Std Dev", f"${wtp_numeric.std():.2f}")
                with col4:
                    st.metric("Range", f"${wtp_numeric.min():.0f}-${wtp_numeric.max():.0f}")

# ============================================================================
# PAGE 3: ASSOCIATION RULE MINING
# ============================================================================

def page_association_rules():
    st.markdown('<h2 class="sub-header">🔍 Association Rule Mining</h2>', 
                unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload or generate data first!")
        return
    
    df = st.session_state.df
    
    st.markdown("""
    ### 📋 About Association Rule Mining
    Discover hidden patterns and relationships in customer behavior using the Apriori algorithm.
    """)
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_support = st.slider("Minimum Support (%)", 5, 50, 10, 5) / 100
    with col2:
        min_confidence = st.slider("Minimum Confidence (%)", 30, 90, 50, 5) / 100
    with col3:
        max_rules = st.slider("Max Rules to Display", 5, 50, 10, 5)
    
    if st.button("🔍 Run Association Rule Mining", type="primary"):
        with st.spinner("Mining association rules..."):
            try:
                # Prepare transaction data (simplified for demo)
                # In real app, you'd use all multi-select columns
                transactions = []
                
                for _, row in df.iterrows():
                    transaction = []
                    
                    # Add categorical values as items
                    for col in ['Q1_Age', 'Q8_Income', 'Q22_Current_Spending', 
                                'Q23_Willingness_To_Pay', 'Q31_Interest_Level']:
                        if col in df.columns and pd.notna(row[col]):
                            transaction.append(f"{col}: {row[col]}")
                    
                    if len(transaction) > 0:
                        transactions.append(transaction)
                
                # Encode transactions
                te = TransactionEncoder()
                te_array = te.fit(transactions).transform(transactions)
                df_encoded = pd.DataFrame(te_array, columns=te.columns_)
                
                # Apply Apriori
                frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
                
                if len(frequent_itemsets) > 0:
                    # Generate rules
                    rules = association_rules(frequent_itemsets, metric="confidence", 
                                              min_threshold=min_confidence)
                    
                    if len(rules) > 0:
                        rules = rules.sort_values('lift', ascending=False).head(max_rules)
                        
                        st.success(f"✅ Found {len(rules)} association rules!")
                        
                        # Display rules
                        st.markdown("### 🎯 Top Association Rules")
                        
                        for idx, rule in rules.iterrows():
                            with st.expander(f"Rule #{idx+1} (Lift: {rule['lift']:.2f})"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**IF (Antecedent):**")
                                    for item in list(rule['antecedents']):
                                        st.write(f"• {item}")
                                
                                with col2:
                                    st.markdown("**THEN (Consequent):**")
                                    for item in list(rule['consequents']):
                                        st.write(f"• {item}")
                                
                                st.markdown("**Metrics:**")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Support", f"{rule['support']:.2%}")
                                with col2:
                                    st.metric("Confidence", f"{rule['confidence']:.2%}")
                                with col3:
                                    st.metric("Lift", f"{rule['lift']:.2f}")
                        
                        # Visualization
                        st.markdown("### 📊 Rules Visualization")
                        
                        fig = px.scatter(rules, x='support', y='confidence', 
                                         size='lift', color='lift',
                                         hover_data=['support', 'confidence', 'lift'],
                                         title='Association Rules: Support vs Confidence',
                                         labels={'support': 'Support', 'confidence': 'Confidence'},
                                         color_continuous_scale='Viridis')
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.warning(f"No rules found with confidence >= {min_confidence*100}%. Try lowering the threshold.")
                else:
                    st.warning(f"No frequent itemsets found with support >= {min_support*100}%. Try lowering the threshold.")
                    
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

# ============================================================================
# PAGE 4: CUSTOMER CLUSTERING
# ============================================================================

def page_clustering():
    st.markdown('<h2 class="sub-header">👥 Customer Clustering Analysis</h2>', 
                unsafe_allow_html=True)
    
    if st.session_state.df_processed is None:
        st.warning("⚠️ Please process data first!")
        return
    
    df = st.session_state.df_processed
    
    st.markdown("""
    ### 📋 About Customer Clustering
    Segment customers into distinct groups based on their characteristics and behavior.
    """)
    
    # Select features for clustering
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric features for clustering")
        return
    
    # Remove target variable from features
    if 'WTP_Numeric' in numeric_cols:
        numeric_cols.remove('WTP_Numeric')
    
    selected_features = st.multiselect(
        "Select features for clustering",
        numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))]
    )
    
    if len(selected_features) < 2:
        st.warning("Please select at least 2 features")
        return
    
    # Clustering parameters
    col1, col2 = st.columns(2)
    
    with col1:
        algorithm = st.selectbox("Clustering Algorithm", 
                                 ["K-Means", "Hierarchical", "DBSCAN"])
    
    with col2:
        if algorithm == "K-Means" or algorithm == "Hierarchical":
            n_clusters = st.slider("Number of Clusters", 2, 10, 4)
    
    if st.button("🎯 Run Clustering Analysis", type="primary"):
        with st.spinner(f"Running {algorithm} clustering..."):
            try:
                # Prepare data
                X = df[selected_features].copy()
                X = X.fillna(X.median())
                
                # Standardize
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Apply clustering
                if algorithm == "K-Means":
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = model.fit_predict(X_scaled)
                    
                elif algorithm == "Hierarchical":
                    model = AgglomerativeClustering(n_clusters=n_clusters)
                    labels = model.fit_predict(X_scaled)
                    
                elif algorithm == "DBSCAN":
                    model = DBSCAN(eps=0.5, min_samples=5)
                    labels = model.fit_predict(X_scaled)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                # Calculate metrics
                if len(set(labels)) > 1:
                    silhouette = silhouette_score(X_scaled, labels)
                    davies_bouldin = davies_bouldin_score(X_scaled, labels)
                    
                    # Display metrics
                    st.markdown("### 📊 Clustering Metrics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Number of Clusters", n_clusters)
                    with col2:
                        st.metric("Silhouette Score", f"{silhouette:.3f}")
                    with col3:
                        st.metric("Davies-Bouldin Index", f"{davies_bouldin:.3f}")
                    
                    # PCA for visualization
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    # Create visualization dataframe
                    viz_df = pd.DataFrame({
                        'PC1': X_pca[:, 0],
                        'PC2': X_pca[:, 1],
                        'Cluster': labels.astype(str)
                    })
                    
                    # Plot
                    st.markdown("### 📊 Cluster Visualization (PCA)")
                    
                    fig = px.scatter(viz_df, x='PC1', y='PC2', color='Cluster',
                                     title=f'{algorithm} Clustering Results',
                                     labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                                             'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'},
                                     color_discrete_sequence=px.colors.qualitative.Set2)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster profiles
                    st.markdown("### 👥 Cluster Profiles")
                    
                    df['Cluster'] = labels
                    
                    for cluster_id in sorted(set(labels)):
                        if cluster_id != -1: # Skip noise points in DBSCAN
                            with st.expander(f"Cluster {cluster_id} ({sum(labels==cluster_id)} members)"):
                                cluster_data = df[df['Cluster'] == cluster_id][selected_features]
                                st.dataframe(cluster_data.describe().T, use_container_width=True)
                
                else:
                    st.error("Clustering failed to identify multiple clusters")
                    
            except Exception as e:
                st.error(f"Error during clustering: {str(e)}")

# ============================================================================
# PAGE 5: REGRESSION ANALYSIS
# ============================================================================

def page_regression():
    st.markdown('<h2 class="sub-header">💰 Willingness to Pay - Regression Analysis</h2>', 
                unsafe_allow_html=True)
    
    if st.session_state.df_processed is None:
        st.warning("⚠️ Please process data first!")
        return
    
    df = st.session_state.df_processed
    
    if 'WTP_Numeric' not in df.columns:
        st.error("WTP_Numeric column not found. Please ensure data is processed correctly.")
        return
    
    st.markdown("""
    ### 📋 About Regression Analysis
    Predict customer willingness to pay using machine learning algorithms.
    """)
    
    # Feature selection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target from features
    if 'WTP_Numeric' in numeric_cols:
        numeric_cols.remove('WTP_Numeric')
    
    selected_features = st.multiselect(
        "Select features for prediction",
        numeric_cols,
        default=numeric_cols[:min(10, len(numeric_cols))]
    )
    
    if len(selected_features) < 1:
        st.warning("Please select at least 1 feature")
        return
    
    # Model selection
    models_to_run = st.multiselect(
        "Select models to compare",
        ["Linear Regression", "Ridge Regression", "Lasso Regression", 
         "Random Forest", "Gradient Boosting"],
        default=["Linear Regression", "Random Forest", "Gradient Boosting"]
    )
    
    test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5) / 100
    
    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models..."):
            try:
                # Prepare data
                X = df[selected_features].copy()
                y = df['WTP_Numeric'].copy()
                
                # Handle missing values
                X = X.fillna(X.median())
                y = y.fillna(y.median())
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train models
                results = {}
                
                for model_name in models_to_run:
                    if model_name == "Linear Regression":
                        model = LinearRegression()
                    elif model_name == "Ridge Regression":
                        model = Ridge(alpha=1.0, random_state=42)
                    elif model_name == "Lasso Regression":
                        model = Lasso(alpha=0.1, random_state=42)
                    elif model_name == "Random Forest":
                        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    elif model_name == "Gradient Boosting":
                        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                    
                    # Train
                    model.fit(X_train_scaled, y_train)
                    
                    # Predict
                    y_pred = model.predict(X_test_scaled)
                    
                    # Metrics
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    results[model_name] = {
                        'model': model,
                        'r2': r2,
                        'rmse': rmse,
                        'mae': mae,
                        'y_pred': y_pred
                    }
                
                # Display results
                st.markdown("### 📊 Model Performance Comparison")
                
                comparison_df = pd.DataFrame({
                    'Model': list(results.keys()),
                    'R² Score': [r['r2'] for r in results.values()],
                    'RMSE': [r['rmse'] for r in results.values()],
                    'MAE': [r['mae'] for r in results.values()]
                })
                
                st.dataframe(comparison_df.style.highlight_max(subset=['R² Score'], color='lightgreen')
                             .highlight_min(subset=['RMSE', 'MAE'], color='lightgreen'),
                             use_container_width=True)
                
                # Best model
                best_model_name = comparison_df.loc[comparison_df['R² Score'].idxmax(), 'Model']
                best_result = results[best_model_name]
                
                st.success(f"🏆 Best Model: {best_model_name} (R² = {best_result['r2']:.4f})")
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Model comparison
                    fig = px.bar(comparison_df, x='Model', y='R² Score',
                                 title='Model Comparison (R² Score)',
                                 color='R² Score',
                                 color_continuous_scale='Viridis')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Actual vs Predicted
                    fig = px.scatter(x=y_test, y=best_result['y_pred'],
                                     labels={'x': 'Actual WTP', 'y': 'Predicted WTP'},
                                     title=f'Actual vs Predicted ({best_model_name})',
                                     trendline='ols')
                    fig.add_shape(type='line', x0=y_test.min(), y0=y_test.min(),
                                  x1=y_test.max(), y1=y_test.max(),
                                  line=dict(color='red', dash='dash'))
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance (if available)
                if hasattr(best_result['model'], 'feature_importances_'):
                    st.markdown("### 🎯 Feature Importance")
                    
                    importance_df = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': best_result['model'].feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(importance_df.head(10), x='Importance', y='Feature',
                                 orientation='h',
                                 title='Top 10 Most Important Features',
                                 color='Importance',
                                 color_continuous_scale='Viridis')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Store results in session state
                st.session_state.regression_results = results
                st.session_state.best_model_name = best_model_name
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")

# ============================================================================
# PAGE 6: BUSINESS DASHBOARD
# ============================================================================

def page_dashboard():
    st.markdown('<h2 class="sub-header">📈 Executive Business Dashboard</h2>', 
                unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload or generate data first!")
        return
    
    df = st.session_state.df
    
    # KPIs
    st.markdown("### 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Respondents", len(df))
    
    with col2:
        if 'Q31_Interest_Level' in df.columns:
            high_interest = len(df[df['Q31_Interest_Level'].isin(['Definitely would subscribe', 'Very likely to subscribe'])])
            st.metric("High Interest", f"{high_interest} ({high_interest/len(df)*100:.1f}%)")
    
    with col3:
        if st.session_state.df_processed is not None and 'WTP_Numeric' in st.session_state.df_processed.columns:
            avg_wtp = st.session_state.df_processed['WTP_Numeric'].mean()
            st.metric("Avg WTP", f"${avg_wtp:.2f}")
    
    with col4:
        if 'Q22_Current_Spending' in df.columns:
            current_spenders = len(df[df['Q22_Current_Spending'] != '$0 (Only free resources)'])
            st.metric("Current Spenders", f"{current_spenders} ({current_spenders/len(df)*100:.1f}%)")
    
    # Charts
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Q31_Interest_Level' in df.columns:
            st.markdown("### 📊 Interest Level Distribution")
            interest_counts = df['Q31_Interest_Level'].value_counts()
            fig = px.pie(values=interest_counts.values, names=interest_counts.index,
                         title='Customer Interest Distribution',
                         color_discrete_sequence=px.colors.sequential.Viridis)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Q23_Willingness_To_Pay' in df.columns:
            st.markdown("### 💰 Willingness to Pay Distribution")
            wtp_counts = df['Q23_Willingness_To_Pay'].value_counts().sort_index()
            fig = px.bar(x=wtp_counts.index, y=wtp_counts.values,
                         labels={'x': 'WTP Range', 'y': 'Count'},
                         title='WTP Distribution',
                         color=wtp_counts.values,
                         color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
    
    # Business Insights
    st.markdown("---")
    st.markdown("### 💡 Key Business Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='insight-box'>
        <h4>🎯 Target Customer Profile</h4>
        <ul>
            <li>Age: 25-44 years (highest interest)</li>
            <li>Income: $50K-$150K (optimal pricing power)</li>
            <li>Interest: High engagement with tech topics</li>
            <li>Current behavior: Active learners, 6-10 hours/week</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='insight-box'>
        <h4>💰 Pricing Strategy</h4>
        <ul>
            <li>Basic Tier: $25-38/month (budget segment)</li>
            <li>Standard Tier: $63-88/month (value seekers)</li>
            <li>Premium Tier: $125+/month (high earners)</li>
            <li>Freemium model for customer acquisition</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='insight-box'>
        <h4>📈 Growth Opportunities</h4>
        <ul>
            <li>40% show high interest - focus on conversion</li>
            <li>Tech professionals willing to pay premium</li>
            <li>Strong demand for personalization features</li>
            <li>Certification value drives higher WTP</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class'insight-box'>
        <h4>🎓 Content Priorities</h4>
        <ul>
            <li>Data Science & ML (highest demand)</li>
            <li>Python Programming (career advancement)</li>
            <li>Business Analytics (corporate market)</li>
            <li>Cloud Computing (enterprise focus)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE 7: EXPORT REPORTS
# ============================================================================

def page_export():
    st.markdown('<h2 class="sub-header">📥 Export Reports & Data</h2>', 
                unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload or generate data first!")
        return
    
    st.markdown("### 📊 Available Exports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📄 Raw Data")
        
        # Export raw data
        csv = st.session_state.df.to_csv(index=False)
        st.download_button(
            label="📥 Download Raw Data (CSV)",
            data=csv,
            file_name=f"survey_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Export processed data
        if st.session_state.df_processed is not None:
            csv_processed = st.session_state.df_processed.to_csv(index=False)
            st.download_button(
                label="📥 Download Processed Data (CSV)",
                data=csv_processed,
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        st.markdown("#### 📊 Analysis Results")
        
        # Export regression results
        if 'regression_results' in st.session_state:
            results_text = "REGRESSION ANALYSIS RESULTS\n\n"
            for model_name, result in st.session_state.regression_results.items():
                results_text += f"{model_name}:\n"
                results_text += f"  R² Score: {result['r2']:.4f}\n"
                results_text += f"  RMSE: ${result['rmse']:.2f}\n"
                results_text += f"  MAE: ${result['mae']:.2f}\n\n"
            
            st.download_button(
                label="📥 Download Regression Results (TXT)",
                data=results_text,
                file_name=f"regression_results_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
    
    st.markdown("---")
    
    # Summary report
    st.markdown("### 📋 Generate Summary Report")
    
    if st.button("📄 Generate Executive Summary"):
        with st.spinner("Generating report..."):
            report = f"""
            EXECUTIVE SUMMARY REPORT
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            ============================================================
            DATASET OVERVIEW
            ============================================================
            Total Records: {len(st.session_state.df)}
            Total Features: {len(st.session_state.df.columns)}
            
            ============================================================
            KEY FINDINGS
            ============================================================
            """
            
            if 'Q31_Interest_Level' in st.session_state.df.columns:
                high_interest = len(st.session_state.df[st.session_state.df['Q31_Interest_Level'].isin(['Definitely would subscribe', 'Very likely to subscribe'])])
                report += f"\n• High Interest Customers: {high_interest} ({high_interest/len(st.session_state.df)*100:.1f}%)"
            
            if st.session_state.df_processed is not None and 'WTP_Numeric' in st.session_state.df_processed.columns:
                avg_wtp = st.session_state.df_processed['WTP_Numeric'].mean()
                report += f"\n• Average Willingness to Pay: ${avg_wtp:.2f}"
            
            report += """
            
            ============================================================
            RECOMMENDATIONS
            ============================================================
            1. Implement tiered pricing strategy ($25, $63, $125 tiers)
            2. Focus marketing on 25-44 age group with $50K+ income
            3. Prioritize Data Science and ML content development
            4. Emphasize certification value in messaging
            5. Offer freemium model for customer acquisition
            
            ============================================================
            NEXT STEPS
            ============================================================
            1. Validate findings with A/B testing
            2. Develop MVP with core features
            3. Launch beta program with early adopters
            4. Monitor KPIs and adjust strategy
            5. Scale based on validated learnings
            """
            
            st.download_button(
                label="📥 Download Executive Summary",
                data=report,
                file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
            
            st.success("✅ Report generated successfully!")

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
