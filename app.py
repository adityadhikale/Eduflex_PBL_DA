"""
AI-Powered Learning Platform - Analytics Dashboard
Multi-tab Streamlit application for market analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score
)
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import xgboost as xgb
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AI Learning Platform Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS - DATA GENERATION
# ============================================================================

@st.cache_data
def generate_synthetic_data(n_samples=1000):
    """Generate synthetic survey dataset"""
    np.random.seed(42)
    
    # Demographics
    ages = np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], n_samples, 
                           p=[0.15, 0.35, 0.25, 0.15, 0.10])
    incomes = np.random.choice(['<25k', '25-50k', '50-75k', '75-100k', '100-150k', '>150k'], n_samples,
                              p=[0.15, 0.25, 0.25, 0.15, 0.12, 0.08])
    fields = np.random.choice(['Tech/IT', 'Data Science', 'Business', 'Marketing', 'Finance', 'Other'], 
                             n_samples, p=[0.25, 0.20, 0.20, 0.15, 0.12, 0.08])
    
    # Learning behavior
    learning_hours = np.random.choice(['0-2', '3-5', '6-10', '11-15', '16-20', '>20'], n_samples,
                                     p=[0.20, 0.35, 0.25, 0.12, 0.05, 0.03])
    
    # Create interest level with correlation to income and hours
    interest_probs = np.random.random(n_samples)
    income_map = {'<25k': 0.2, '25-50k': 0.3, '50-75k': 0.5, '75-100k': 0.6, '100-150k': 0.7, '>150k': 0.8}
    income_boost = np.array([income_map[inc] for inc in incomes])
    hours_map = {'0-2': 0.2, '3-5': 0.4, '6-10': 0.6, '11-15': 0.8, '16-20': 0.9, '>20': 0.95}
    hours_boost = np.array([hours_map[h] for h in learning_hours])
    
    interest_score = (interest_probs + income_boost + hours_boost) / 3
    interest_levels = np.where(interest_score > 0.7, 'Definitely Subscribe',
                    np.where(interest_score > 0.5, 'Very Likely',
                    np.where(interest_score > 0.35, 'Somewhat Interested',
                    np.where(interest_score > 0.2, 'Might Consider', 'Not Interested'))))
    
    # Willingness to Pay (correlated with income and interest)
    wtp_base = np.random.randint(10, 150, n_samples)
    income_multiplier = np.array([0.5, 0.7, 1.0, 1.3, 1.6, 2.0])[[list({'<25k': 0, '25-50k': 1, '50-75k': 2, '75-100k': 3, '100-150k': 4, '>150k': 5}.keys()).index(inc) for inc in incomes]]
    wtp = (wtp_base * income_multiplier * (1 + interest_score)).astype(int)
    wtp = np.clip(wtp, 10, 200)
    
    # Spending behavior
    current_spending = np.where(wtp < 30, '<20',
                       np.where(wtp < 60, '20-50',
                       np.where(wtp < 90, '50-80',
                       np.where(wtp < 120, '80-120', '>120'))))
    
    # Topics (multi-select simulation)
    all_topics = ['Data Science', 'Python', 'Web Dev', 'Business Analytics', 
                 'AI/ML', 'Cloud Computing', 'Marketing', 'Finance']
    topics_binary = np.random.randint(0, 2, (n_samples, len(all_topics)))
    
    # Platforms (multi-select)
    all_platforms = ['Coursera', 'Udemy', 'LinkedIn Learning', 'YouTube', 'DataCamp', 'edX']
    platforms_binary = np.random.randint(0, 2, (n_samples, len(all_platforms)))
    
    # Formats (multi-select)
    all_formats = ['Video Short', 'Video Long', 'Interactive Coding', 'Live Sessions', 'Reading', 'Projects']
    formats_binary = np.random.randint(0, 2, (n_samples, len(all_formats)))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Age': ages,
        'Income': incomes,
        'Professional_Field': fields,
        'Learning_Hours': learning_hours,
        'Interest_Level': interest_levels,
        'Willingness_To_Pay': wtp,
        'Current_Spending': current_spending,
        'Engagement_Frequency': np.random.choice(['Daily', '3-5/week', '1-2/week', 'Few/month', 'Rarely'], 
                                                 n_samples, p=[0.15, 0.30, 0.30, 0.15, 0.10])
    })
    
    # Add topic columns
    for i, topic in enumerate(all_topics):
        df[f'Topic_{topic.replace(" ", "_").replace("/", "_")}'] = topics_binary[:, i]
    
    # Add platform columns
    for i, platform in enumerate(all_platforms):
        df[f'Platform_{platform.replace(" ", "_")}'] = platforms_binary[:, i]
    
    # Add format columns
    for i, fmt in enumerate(all_formats):
        df[f'Format_{fmt.replace(" ", "_")}'] = formats_binary[:, i]
    
    return df

@st.cache_data
def load_sample_data():
    """Load or generate sample data"""
    try:
        # Try to load from URL (replace with your GitHub URL)
        # df = pd.read_csv('YOUR_GITHUB_RAW_URL_HERE')
        df = generate_synthetic_data(1000)
        return df
    except:
        return generate_synthetic_data(1000)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def download_csv(df, filename):
    """Generate download link for dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def to_excel(df):
    """Convert dataframe to Excel"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# ============================================================================
# ASSOCIATION RULES FUNCTIONS
# ============================================================================

def prepare_transaction_data(df, selected_cols):
    """Prepare data for association rule mining"""
    transactions = []
    for _, row in df.iterrows():
        transaction = []
        for col in selected_cols:
            if col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    if row[col] == 1:
                        transaction.append(col.replace('_', ' '))
                else:
                    transaction.append(f"{col}: {row[col]}")
        transactions.append(transaction)
    return transactions

def run_apriori(df, selected_cols, min_support, min_confidence, min_lift):
    """Run Apriori algorithm"""
    transactions = prepare_transaction_data(df, selected_cols)
    
    # One-hot encode
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)
    
    # Find frequent itemsets
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    
    if len(frequent_itemsets) == 0:
        return None, None
    
    # Generate rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    # Filter by lift
    rules = rules[rules['lift'] >= min_lift]
    
    return frequent_itemsets, rules

# ============================================================================
# CLASSIFICATION FUNCTIONS
# ============================================================================

def prepare_classification_data(df, target_col, binary=False):
    """Prepare data for classification"""
    # Encode categorical features
    df_encoded = df.copy()
    le_dict = {}
    
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object' and col != target_col:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            le_dict[col] = le
    
    # Prepare target
    le_target = LabelEncoder()
    y = le_target.fit_transform(df_encoded[target_col])
    
    if binary:
        # Convert to binary (high interest vs low interest)
        y = np.where(y <= 1, 1, 0)
    
    # Features
    X = df_encoded.drop([target_col], axis=1)
    
    # Handle any remaining object columns
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    return X, y, le_target

def train_classification_models(X_train, X_test, y_train, y_test, models_list):
    """Train multiple classification models"""
    results = {}
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for model_name in models_list:
        if model_name == 'Logistic Regression':
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_name == 'Random Forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_name == 'XGBoost':
            model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        elif model_name == 'SVM':
            model = SVC(probability=True, random_state=42)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
        
        results[model_name] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
    
    return results, scaler

# ============================================================================
# CLUSTERING FUNCTIONS
# ============================================================================

def prepare_clustering_data(df, feature_cols):
    """Prepare data for clustering"""
    df_cluster = df[feature_cols].copy()
    
    # Encode categorical
    for col in df_cluster.columns:
        if df_cluster[col].dtype == 'object':
            df_cluster[col] = LabelEncoder().fit_transform(df_cluster[col].astype(str))
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)
    
    return X_scaled, df_cluster, scaler

def run_clustering(X, algorithm, n_clusters=3, eps=0.5, min_samples=5):
    """Run clustering algorithm"""
    if algorithm == 'K-Means':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif algorithm == 'Hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif algorithm == 'DBSCAN':
        model = DBSCAN(eps=eps, min_samples=min_samples)
    
    labels = model.fit_predict(X)
    
    # Calculate metrics
    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
    else:
        silhouette = -1
        davies_bouldin = -1
    
    return labels, silhouette, davies_bouldin, model

# ============================================================================
# REGRESSION FUNCTIONS
# ============================================================================

def prepare_regression_data(df, target_col):
    """Prepare data for regression"""
    df_encoded = df.copy()
    
    # Encode categorical
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object' and col != target_col:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))
    
    y = df_encoded[target_col]
    X = df_encoded.drop([target_col], axis=1)
    
    # Handle any remaining object columns
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    return X, y

def train_regression_models(X_train, X_test, y_train, y_test, models_list):
    """Train multiple regression models"""
    results = {}
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for model_name in models_list:
        if model_name == 'Linear Regression':
            model = LinearRegression()
        elif model_name == 'Ridge':
            model = Ridge(alpha=1.0, random_state=42)
        elif model_name == 'Lasso':
            model = Lasso(alpha=1.0, random_state=42)
        elif model_name == 'Random Forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_name == 'XGBoost':
            model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        results[model_name] = {
            'model': model,
            'predictions': y_pred,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    return results, scaler

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Title
    st.title("🎓 AI Learning Platform - Market Analytics Dashboard")
    st.markdown("### Comprehensive Analysis Suite: Association Rules | Classification | Clustering | Regression | Dynamic Pricing")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📂 Data Upload")
        
        # Data source selection
        data_source = st.radio(
            "Select Data Source:",
            ["Use Sample Data", "Upload CSV", "Load from URL"]
        )
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} rows")
            else:
                df = load_sample_data()
                st.info("Using sample data")
        elif data_source == "Load from URL":
            url = st.text_input("Enter CSV URL (GitHub raw URL recommended)")
            if url:
                try:
                    df = pd.read_csv(url)
                    st.success(f"✅ Loaded {len(df)} rows from URL")
                except:
                    st.error("Failed to load from URL. Using sample data.")
                    df = load_sample_data()
            else:
                df = load_sample_data()
                st.info("Using sample data")
        else:
            df = load_sample_data()
            st.success(f"✅ Sample data loaded: {len(df)} rows")
        
        st.markdown("---")
        st.markdown("### 📊 Data Preview")
        st.dataframe(df.head(3), use_container_width=True)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("This dashboard provides comprehensive market analysis tools for the AI Learning Platform.")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔗 Association Rules", 
        "🎯 Classification", 
        "👥 Clustering", 
        "📈 Regression",
        "💰 Dynamic Pricing"
    ])
    
    # ========================================================================
    # TAB 1: ASSOCIATION RULES
    # ========================================================================
    
    with tab1:
        st.header("🔗 Association Rule Mining")
        st.markdown("Discover patterns and relationships in customer behavior")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Parameters")
            
            min_support = st.slider("Minimum Support", 0.01, 0.5, 0.1, 0.01)
            min_confidence = st.slider("Minimum Confidence", 0.1, 0.9, 0.5, 0.05)
            min_lift = st.slider("Minimum Lift", 1.0, 5.0, 1.5, 0.1)
            
            # Select columns for analysis
            binary_cols = [col for col in df.columns if df[col].nunique() == 2 or col.startswith(('Topic_', 'Platform_', 'Format_'))]
            categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
            
            available_cols = binary_cols + categorical_cols
            selected_cols = st.multiselect(
                "Select Features for Analysis",
                available_cols,
                default=binary_cols[:10] if len(binary_cols) > 10 else binary_cols
            )
            
            run_apriori_btn = st.button("🚀 Run Association Rule Mining", type="primary")
        
        with col2:
            if run_apriori_btn and len(selected_cols) > 0:
                with st.spinner("Running Apriori algorithm..."):
                    frequent_itemsets, rules = run_apriori(df, selected_cols, min_support, min_confidence, min_lift)
                    
                    if rules is not None and len(rules) > 0:
                        st.success(f"✅ Found {len(rules)} association rules")
                        
                        # Sort by lift
                        rules = rules.sort_values('lift', ascending=False)
                        
                        # Display top rules
                        st.subheader(f"📊 Top 10 Rules by Lift")
                        
                        top_rules = rules.head(10).copy()
                        top_rules['antecedents'] = top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                        top_rules['consequents'] = top_rules['consequents'].apply(lambda x: ', '.join(list(x)))
                        
                        display_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
                        st.dataframe(
                            top_rules[display_cols].style.format({
                                'support': '{:.3f}',
                                'confidence': '{:.3f}',
                                'lift': '{:.3f}'
                            }),
                            use_container_width=True
                        )
                        
                        # Visualization
                        st.subheader("📈 Visualizations")
                        
                        # Scatter plot
                        fig = px.scatter(
                            rules,
                            x='support',
                            y='confidence',
                            size='lift',
                            color='lift',
                            hover_data=['support', 'confidence', 'lift'],
                            title='Association Rules: Support vs Confidence (sized by Lift)',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Metrics
                        col_m1, col_m2, col_m3 = st.columns(3)
                        col_m1.metric("Total Rules", len(rules))
                        col_m2.metric("Avg Confidence", f"{rules['confidence'].mean():.3f}")
                        col_m3.metric("Avg Lift", f"{rules['lift'].mean():.3f}")
                        
                        # Download
                        st.subheader("💾 Download Results")
                        rules_export = rules.copy()
                        rules_export['antecedents'] = rules_export['antecedents'].apply(lambda x: ', '.join(list(x)))
                        rules_export['consequents'] = rules_export['consequents'].apply(lambda x: ', '.join(list(x)))
                        
                        csv = rules_export.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Rules as CSV",
                            data=csv,
                            file_name="association_rules.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("⚠️ No rules found with current parameters. Try lowering the thresholds.")
            else:
                st.info("👈 Configure parameters and click 'Run' to generate association rules")
    
    # ========================================================================
    # TAB 2: CLASSIFICATION
    # ========================================================================
    
    with tab2:
        st.header("🎯 Classification Analysis")
        st.markdown("Predict customer interest levels using machine learning")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Configuration")
            
            # Target selection
            target_col = st.selectbox(
                "Target Variable",
                ['Interest_Level', 'Willingness_To_Pay'] if 'Interest_Level' in df.columns else df.select_dtypes(include='object').columns
            )
            
            # Binary or multi-class
            is_binary = st.checkbox("Convert to Binary Classification", value=False)
            if is_binary:
                st.info("Will convert to: High Interest (1) vs Low Interest (0)")
            
            # Model selection
            available_models = ['Logistic Regression', 'Random Forest', 'XGBoost', 'SVM']
            selected_models = st.multiselect(
                "Select Models to Compare",
                available_models,
                default=['Logistic Regression', 'Random Forest']
            )
            
            # Train-test split
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
            
            run_classification = st.button("🚀 Train Models", type="primary")
        
        with col2:
            if run_classification and len(selected_models) > 0:
                with st.spinner("Training models..."):
                    # Prepare data
                    X, y, le_target = prepare_classification_data(df, target_col, binary=is_binary)
                    
                    # Split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                    
                    # Train models
                    results, scaler = train_classification_models(
                        X_train, X_test, y_train, y_test, selected_models
                    )
                    
                    st.success("✅ Training complete!")
                    
                    # Model comparison
                    st.subheader("📊 Model Comparison")
                    
                    comparison_df = pd.DataFrame({
                        'Model': list(results.keys()),
                        'Accuracy': [r['accuracy'] for r in results.values()],
                        'Precision': [r['precision'] for r in results.values()],
                        'Recall': [r['recall'] for r in results.values()],
                        'F1-Score': [r['f1'] for r in results.values()]
                    })
                    
                    st.dataframe(
                        comparison_df.style.format({
                            'Accuracy': '{:.4f}',
                            'Precision': '{:.4f}',
                            'Recall': '{:.4f}',
                            'F1-Score': '{:.4f}'
                        }).highlight_max(axis=0, color='lightgreen'),
                        use_container_width=True
                    )
                    
                    # Best model
                    best_model_name = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
                    st.success(f"🏆 Best Model: **{best_model_name}**")
                    
                    # Visualizations
                    st.subheader("📈 Visualizations")
                    
                    # Metrics comparison
                    fig = px.bar(
                        comparison_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                        x='Model',
                        y='Score',
                        color='Metric',
                        barmode='group',
                        title='Model Performance Comparison'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Confusion matrix for best model
                    st.subheader(f"🎯 Confusion Matrix - {best_model_name}")
                    
                    best_result = results[best_model_name]
                    cm = confusion_matrix(y_test, best_result['predictions'])
                    
                    fig_cm = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=[f"Class {i}" for i in range(cm.shape[1])],
                        y=[f"Class {i}" for i in range(cm.shape[0])],
                        color_continuous_scale='Blues',
                        text_auto=True
                    )
                    fig_cm.update_layout(height=400)
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                    # Classification report
                    with st.expander("📋 Detailed Classification Report"):
                        report = classification_report(y_test, best_result['predictions'], output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
                    
                    # Feature importance (if applicable)
                    if best_model_name in ['Random Forest', 'XGBoost']:
                        st.subheader("🎯 Feature Importance")
                        
                        model = best_result['model']
                        if hasattr(model, 'feature_importances_'):
                            feat_imp = pd.DataFrame({
                                'Feature': X.columns,
                                'Importance': model.feature_importances_
                            }).sort_values('Importance', ascending=False).head(15)
                            
                            fig_imp = px.bar(
                                feat_imp,
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title='Top 15 Most Important Features'
                            )
                            fig_imp.update_layout(height=500)
                            st.plotly_chart(fig_imp, use_container_width=True)
                    
                    # Download predictions
                    st.subheader("💾 Download Results")
                    predictions_df = pd.DataFrame({
                        'Actual': y_test,
                        'Predicted': best_result['predictions']
                    })
                    
                    csv = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="classification_predictions.csv",
                        mime="text/csv"
                    )
            else:
                st.info("👈 Select models and click 'Train Models' to start analysis")
    
    # ========================================================================
    # TAB 3: CLUSTERING
    # ========================================================================
    
    with tab3:
        st.header("👥 Customer Segmentation - Clustering")
        st.markdown("Identify customer segments and create personas")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Configuration")
            
            # Feature selection
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include='object').columns.tolist()
            
            clustering_features = st.multiselect(
                "Select Features for Clustering",
                numeric_cols + categorical_cols,
                default=['Age', 'Income', 'Learning_Hours', 'Willingness_To_Pay'][:min(4, len(numeric_cols + categorical_cols))]
            )
            
            # Algorithm
            algorithm = st.selectbox("Clustering Algorithm", ['K-Means', 'Hierarchical', 'DBSCAN'])
            
            if algorithm in ['K-Means', 'Hierarchical']:
                n_clusters = st.slider("Number of Clusters", 2, 10, 4)
            else:
                eps = st.slider("DBSCAN: Epsilon", 0.1, 2.0, 0.5, 0.1)
                min_samples = st.slider("DBSCAN: Min Samples", 2, 20, 5)
                n_clusters = None
            
            run_clustering_btn = st.button("🚀 Run Clustering", type="primary")
        
        with col2:
            if run_clustering_btn and len(clustering_features) > 0:
                with st.spinner("Running clustering..."):
                    # Prepare data
                    X_scaled, df_cluster, scaler = prepare_clustering_data(df, clustering_features)
                    
                    # Run clustering
                    if algorithm == 'DBSCAN':
                        labels, silhouette, davies_bouldin, model = run_clustering(
                            X_scaled, algorithm, eps=eps, min_samples=min_samples
                        )
                    else:
                        labels, silhouette, davies_bouldin, model = run_clustering(
                            X_scaled, algorithm, n_clusters=n_clusters
                        )
                    
                    df_result = df.copy()
                    df_result['Cluster'] = labels
                    
                    st.success(f"✅ Clustering complete! Found {len(np.unique(labels))} clusters")
                    
                    # Metrics
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Clusters Found", len(np.unique(labels)))
                    col_m2.metric("Silhouette Score", f"{silhouette:.3f}" if silhouette > -1 else "N/A")
                    col_m3.metric("Davies-Bouldin Index", f"{davies_bouldin:.3f}" if davies_bouldin > -1 else "N/A")
                    
                    # Cluster profiles
                    st.subheader("📊 Cluster Profiles (Personas)")
                    
                    profile_features = [f for f in clustering_features if f in df.columns]
                    cluster_profiles = df_result.groupby('Cluster')[profile_features].agg(['mean', 'count'])
                    
                    st.dataframe(cluster_profiles.style.format("{:.2f}"), use_container_width=True)
                    
                    # PCA Visualization
                    st.subheader("📈 2D Visualization (PCA)")
                    
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    pca_df = pd.DataFrame({
                        'PC1': X_pca[:, 0],
                        'PC2': X_pca[:, 1],
                        'Cluster': labels
                    })
                    
                    fig_pca = px.scatter(
                        pca_df,
                        x='PC1',
                        y='PC2',
                        color='Cluster',
                        title=f'Cluster Visualization (PCA) - Explained Variance: {sum(pca.explained_variance_ratio_):.2%}',
                        labels={'Cluster': 'Cluster'},
                        color_continuous_scale='viridis' if algorithm == 'DBSCAN' else None
                    )
                    fig_pca.update_layout(height=500)
                    st.plotly_chart(fig_pca, use_container_width=True)
                    
                    # Elbow curve (K-Means only)
                    if algorithm == 'K-Means':
                        st.subheader("📉 Elbow Curve")
                        
                        inertias = []
                        K_range = range(2, min(11, len(X_scaled)))
                        
                        for k in K_range:
                            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                            kmeans.fit(X_scaled)
                            inertias.append(kmeans.inertia_)
                        
                        fig_elbow = px.line(
                            x=list(K_range),
                            y=inertias,
                            markers=True,
                            labels={'x': 'Number of Clusters', 'y': 'Inertia'},
                            title='Elbow Curve for Optimal K'
                        )
                        fig_elbow.update_layout(height=400)
                        st.plotly_chart(fig_elbow, use_container_width=True)
                    
                    # Cluster distribution
                    st.subheader("📊 Cluster Distribution")
                    
                    cluster_counts = pd.DataFrame(df_result['Cluster'].value_counts()).reset_index()
                    cluster_counts.columns = ['Cluster', 'Count']
                    
                    fig_dist = px.bar(
                        cluster_counts,
                        x='Cluster',
                        y='Count',
                        title='Number of Customers per Cluster',
                        color='Count',
                        color_continuous_scale='Blues'
                    )
                    fig_dist.update_layout(height=400)
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Download
                    st.subheader("💾 Download Results")
                    
                    csv = df_result.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Cluster Assignments",
                        data=csv,
                        file_name="clustering_results.csv",
                        mime="text/csv"
                    )
            else:
                st.info("👈 Select features and click 'Run Clustering' to start analysis")
    
    # ========================================================================
    # TAB 4: REGRESSION
    # ========================================================================
    
    with tab4:
        st.header("📈 Regression Analysis")
        st.markdown("Predict continuous values (e.g., Willingness to Pay)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Configuration")
            
            # Target selection
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            target_col_reg = st.selectbox(
                "Target Variable",
                numeric_cols,
                index=numeric_cols.index('Willingness_To_Pay') if 'Willingness_To_Pay' in numeric_cols else 0
            )
            
            # Model selection
            available_models_reg = ['Linear Regression', 'Ridge', 'Lasso', 'Random Forest', 'XGBoost']
            selected_models_reg = st.multiselect(
                "Select Models to Compare",
                available_models_reg,
                default=['Linear Regression', 'Random Forest', 'XGBoost']
            )
            
            # Train-test split
            test_size_reg = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05, key='reg_test_size')
            
            run_regression = st.button("🚀 Train Models", type="primary", key='reg_train')
        
        with col2:
            if run_regression and len(selected_models_reg) > 0:
                with st.spinner("Training regression models..."):
                    # Prepare data
                    X_reg, y_reg = prepare_regression_data(df, target_col_reg)
                    
                    # Split
                    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
                        X_reg, y_reg, test_size=test_size_reg, random_state=42
                    )
                    
                    # Train models
                    results_reg, scaler_reg = train_regression_models(
                        X_train_reg, X_test_reg, y_train_reg, y_test_reg, selected_models_reg
                    )
                    
                    st.success("✅ Training complete!")
                    
                    # Model comparison
                    st.subheader("📊 Model Comparison")
                    
                    comparison_df_reg = pd.DataFrame({
                        'Model': list(results_reg.keys()),
                        'R²': [r['r2'] for r in results_reg.values()],
                        'RMSE': [r['rmse'] for r in results_reg.values()],
                        'MAE': [r['mae'] for r in results_reg.values()],
                        'MAPE (%)': [r['mape'] for r in results_reg.values()]
                    })
                    
                    st.dataframe(
                        comparison_df_reg.style.format({
                            'R²': '{:.4f}',
                            'RMSE': '{:.2f}',
                            'MAE': '{:.2f}',
                            'MAPE (%)': '{:.2f}'
                        }).highlight_max(subset=['R²'], color='lightgreen')
                        .highlight_min(subset=['RMSE', 'MAE', 'MAPE (%)'], color='lightgreen'),
                        use_container_width=True
                    )
                    
                    # Best model
                    best_model_name_reg = comparison_df_reg.loc[comparison_df_reg['R²'].idxmax(), 'Model']
                    st.success(f"🏆 Best Model: **{best_model_name_reg}** (Highest R²)")
                    
                    # Metrics visualization
                    st.subheader("📈 Model Performance Metrics")
                    
                    fig_metrics = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('R² Score', 'RMSE', 'MAE', 'MAPE'),
                        specs=[[{'type': 'bar'}, {'type': 'bar'}],
                               [{'type': 'bar'}, {'type': 'bar'}]]
                    )
                    
                    models = comparison_df_reg['Model'].tolist()
                    
                    fig_metrics.add_trace(
                        go.Bar(x=models, y=comparison_df_reg['R²'], name='R²'),
                        row=1, col=1
                    )
                    fig_metrics.add_trace(
                        go.Bar(x=models, y=comparison_df_reg['RMSE'], name='RMSE'),
                        row=1, col=2
                    )
                    fig_metrics.add_trace(
                        go.Bar(x=models, y=comparison_df_reg['MAE'], name='MAE'),
                        row=2, col=1
                    )
                    fig_metrics.add_trace(
                        go.Bar(x=models, y=comparison_df_reg['MAPE (%)'], name='MAPE'),
                        row=2, col=2
                    )
                    
                    fig_metrics.update_layout(height=600, showlegend=False)
                    st.plotly_chart(fig_metrics, use_container_width=True)
                    
                    # Actual vs Predicted
                    st.subheader(f"🎯 Actual vs Predicted - {best_model_name_reg}")
                    
                    best_result_reg = results_reg[best_model_name_reg]
                    
                    actual_pred_df = pd.DataFrame({
                        'Actual': y_test_reg,
                        'Predicted': best_result_reg['predictions']
                    })
                    
                    fig_scatter = px.scatter(
                        actual_pred_df,
                        x='Actual',
                        y='Predicted',
                        title=f'Actual vs Predicted Values - {best_model_name_reg}',
                        labels={'Actual': f'Actual {target_col_reg}', 'Predicted': f'Predicted {target_col_reg}'},
                        trendline='ols'
                    )
                    
                    # Add perfect prediction line
                    max_val = max(actual_pred_df['Actual'].max(), actual_pred_df['Predicted'].max())
                    min_val = min(actual_pred_df['Actual'].min(), actual_pred_df['Predicted'].min())
                    fig_scatter.add_trace(
                        go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(color='red', dash='dash')
                        )
                    )
                    
                    fig_scatter.update_layout(height=500)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # Residual plot
                    st.subheader("📉 Residual Analysis")
                    
                    residuals = y_test_reg - best_result_reg['predictions']
                    residual_df = pd.DataFrame({
                        'Predicted': best_result_reg['predictions'],
                        'Residuals': residuals
                    })
                    
                    fig_residual = px.scatter(
                        residual_df,
                        x='Predicted',
                        y='Residuals',
                        title='Residual Plot',
                        labels={'Predicted': f'Predicted {target_col_reg}', 'Residuals': 'Residuals'}
                    )
                    fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
                    fig_residual.update_layout(height=400)
                    st.plotly_chart(fig_residual, use_container_width=True)
                    
                    # Feature importance
                    if best_model_name_reg in ['Random Forest', 'XGBoost']:
                        st.subheader("🎯 Feature Importance")
                        
                        model_reg = best_result_reg['model']
                        if hasattr(model_reg, 'feature_importances_'):
                            feat_imp_reg = pd.DataFrame({
                                'Feature': X_reg.columns,
                                'Importance': model_reg.feature_importances_
                            }).sort_values('Importance', ascending=False).head(15)
                            
                            fig_imp_reg = px.bar(
                                feat_imp_reg,
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title='Top 15 Most Important Features'
                            )
                            fig_imp_reg.update_layout(height=500)
                            st.plotly_chart(fig_imp_reg, use_container_width=True)
                    
                    # Download
                    st.subheader("💾 Download Results")
                    
                    predictions_df_reg = pd.DataFrame({
                        'Actual': y_test_reg,
                        'Predicted': best_result_reg['predictions'],
                        'Residual': residuals
                    })
                    
                    csv_reg = predictions_df_reg.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv_reg,
                        file_name="regression_predictions.csv",
                        mime="text/csv"
                    )
            else:
                st.info("👈 Select models and click 'Train Models' to start analysis")
    
    # ========================================================================
    # TAB 5: DYNAMIC PRICING
    # ========================================================================
    
    with tab5:
        st.header("💰 Dynamic Pricing Engine")
        st.markdown("Real-time price prediction based on customer features")
        
        # First, train a regression model if not already done
        if 'Willingness_To_Pay' not in df.columns:
            st.error("❌ 'Willingness_To_Pay' column not found in dataset")
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🎛️ Customer Profile Input")
                
                # Input fields for customer features
                input_features = {}
                
                # Age
                if 'Age' in df.columns:
                    age_options = df['Age'].unique().tolist()
                    input_features['Age'] = st.selectbox("Age Group", age_options)
                
                # Income
                if 'Income' in df.columns:
                    income_options = df['Income'].unique().tolist()
                    input_features['Income'] = st.selectbox("Income Level", income_options)
                
                # Professional Field
                if 'Professional_Field' in df.columns:
                    field_options = df['Professional_Field'].unique().tolist()
                    input_features['Professional_Field'] = st.selectbox("Professional Field", field_options)
                
                # Learning Hours
                if 'Learning_Hours' in df.columns:
                    hours_options = df['Learning_Hours'].unique().tolist()
                    input_features['Learning_Hours'] = st.selectbox("Weekly Learning Hours", hours_options)
                
                # Interest Level
                if 'Interest_Level' in df.columns:
                    interest_options = df['Interest_Level'].unique().tolist()
                    input_features['Interest_Level'] = st.selectbox("Interest Level", interest_options)
                
                # Current Spending
                if 'Current_Spending' in df.columns:
                    spending_options = df['Current_Spending'].unique().tolist()
                    input_features['Current_Spending'] = st.selectbox("Current Monthly Spending", spending_options)
                
                # Pricing strategy
                st.markdown("---")
                st.subheader("💡 Pricing Strategy")
                
                base_multiplier = st.slider("Base Price Multiplier", 0.8, 1.5, 1.0, 0.05)
                urgency_discount = st.slider("Urgency/Promotion Discount (%)", 0, 30, 10, 5)
                
                predict_btn = st.button("💰 Calculate Optimal Price", type="primary")
            
            with col2:
                if predict_btn:
                    with st.spinner("Calculating optimal price..."):
                        # Train a quick model if needed
                        X_price, y_price = prepare_regression_data(df, 'Willingness_To_Pay')
                        
                        # Use Random Forest for prediction
                        scaler_price = StandardScaler()
                        X_price_scaled = scaler_price.fit_transform(X_price)
                        
                        model_price = RandomForestRegressor(n_estimators=100, random_state=42)
                        model_price.fit(X_price_scaled, y_price)
                        
                        # Prepare input data
                        input_df = pd.DataFrame([input_features])
                        
                        # Encode categorical features to match training data
                        for col in input_df.columns:
                            if input_df[col].dtype == 'object':
                                if col in df.columns:
                                    le = LabelEncoder()
                                    le.fit(df[col].astype(str))
                                    input_df[col] = le.transform(input_df[col].astype(str))
                        
                        # Add missing features (fill with median/mode)
                        for col in X_price.columns:
                            if col not in input_df.columns:
                                if df[col].dtype in ['int64', 'float64']:
                                    input_df[col] = df[col].median()
                                else:
                                    input_df[col] = df[col].mode()[0] if len(df[col].mode()) > 0 else 0
                        
                        # Reorder columns to match training data
                        input_df = input_df[X_price.columns]
                        
                        # Scale and predict
                        input_scaled = scaler_price.transform(input_df)
                        base_prediction = model_price.predict(input_scaled)[0]
                        
                        # Apply pricing strategy
                        adjusted_price = base_prediction * base_multiplier * (1 - urgency_discount/100)
                        
                        # Get prediction interval (using model's predictions on training data)
                        predictions_on_train = model_price.predict(X_price_scaled)
                        residuals = y_price - predictions_on_train
                        std_residual = np.std(residuals)
                        
                        confidence_interval_lower = adjusted_price - 1.96 * std_residual
                        confidence_interval_upper = adjusted_price + 1.96 * std_residual
                        
                        st.success("✅ Price Calculated!")
                        
                        # Display results
                        st.markdown("### 🎯 Pricing Results")
                        
                        col_p1, col_p2, col_p3 = st.columns(3)
                        
                        col_p1.metric(
                            "Base Prediction",
                            f"${base_prediction:.2f}",
                            help="Model's base prediction"
                        )
                        
                        col_p2.metric(
                            "Recommended Price",
                            f"${adjusted_price:.2f}",
                            f"{((adjusted_price - base_prediction)/base_prediction*100):+.1f}%"
                        )
                        
                        col_p3.metric(
                            "Confidence Interval",
                            f"${confidence_interval_lower:.2f} - ${confidence_interval_upper:.2f}",
                            help="95% confidence interval"
                        )
                        
                        # Pricing breakdown
                        st.markdown("### 📊 Pricing Breakdown")
                        
                        breakdown_df = pd.DataFrame({
                            'Component': ['Base Prediction', 'Price Multiplier', 'Discount Applied', 'Final Price'],
                            'Value': [f"${base_prediction:.2f}", 
                                    f"{base_multiplier:.2f}x",
                                    f"-{urgency_discount}%",
                                    f"${adjusted_price:.2f}"]
                        })
                        
                        st.table(breakdown_df)
                        
                        # Visualization
                        st.markdown("### 📈 Price Comparison")
                        
                        # Create comparison chart
                        comparison_prices = pd.DataFrame({
                            'Scenario': ['Conservative\n(-10%)', 'Base\nPrediction', 'Recommended\nPrice', 
                                       'Aggressive\n(+10%)', 'Market Max'],
                            'Price': [base_prediction * 0.9, base_prediction, adjusted_price, 
                                    base_prediction * 1.1, y_price.max()],
                            'Color': ['lightblue', 'blue', 'green', 'orange', 'red']
                        })
                        
                        fig_price = px.bar(
                            comparison_prices,
                            x='Scenario',
                            y='Price',
                            title='Price Comparison Across Scenarios',
                            color='Color',
                            color_discrete_map={c: c for c in comparison_prices['Color']},
                            text='Price'
                        )
                        fig_price.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
                        fig_price.update_layout(showlegend=False, height=400)
                        st.plotly_chart(fig_price, use_container_width=True)
                        
                        # Recommendations
                        st.markdown("### 💡 Pricing Recommendations")
                        
                        if adjusted_price > base_prediction:
                            st.info("📈 **Premium Pricing Strategy**: Customer profile suggests higher willingness to pay. Consider bundling premium features.")
                        elif adjusted_price < base_prediction:
                            st.warning("📉 **Discount Pricing Strategy**: Price-sensitive customer. Consider offering payment plans or trial periods.")
                        else:
                            st.success("✅ **Standard Pricing**: Customer profile aligns with average market expectations.")
                        
                        # Market positioning
                        percentile = (y_price <= adjusted_price).mean() * 100
                        st.metric("Market Position", f"{percentile:.1f}th Percentile", 
                                help="Percentage of customers willing to pay less than recommended price")
                        
                        # Revenue optimization tips
                        with st.expander("🎯 Revenue Optimization Tips"):
                            st.markdown("""
                            **Based on this customer profile:**
                            
                            1. **Upselling Opportunities**: 
                               - Highlight premium features that align with customer's field
                               - Offer annual plan with 20% discount
                            
                            2. **Retention Strategies**:
                               - Personalized learning path recommendations
                               - Early access to new content
                            
                            3. **Conversion Tactics**:
                               - Limited-time offer (urgency)
                               - Money-back guarantee (reduce risk)
                               - Free trial period
                            
                            4. **Segment-Specific Offers**:
                               - Corporate/team pricing for business customers
                               - Student discounts for lower income segments
                            """)
                
                else:
                    st.info("👈 Enter customer profile and click 'Calculate Optimal Price'")
                    
                    # Show sample price distribution
                    st.markdown("### 📊 Market Price Distribution")
                    
                    fig_dist = px.histogram(
                        df,
                        x='Willingness_To_Pay',
                        nbins=30,
                        title='Distribution of Customer Willingness to Pay',
                        labels={'Willingness_To_Pay': 'Price ($)', 'count': 'Number of Customers'}
                    )
                    fig_dist.add_vline(x=df['Willingness_To_Pay'].median(), 
                                      line_dash="dash", line_color="red",
                                      annotation_text=f"Median: ${df['Willingness_To_Pay'].median():.2f}")
                    st.plotly_chart(fig_dist, use_container_width=True)

if __name__ == "__main__":
    main()