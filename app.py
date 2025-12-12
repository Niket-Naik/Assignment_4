import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Diamond Dynamics",
    page_icon="💎",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #F0F8FF;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 20px 0;
    }
    .cluster-box {
        background-color: #F0FFF4;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #10B981;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">💎 Diamond Dynamics</h1>', unsafe_allow_html=True)
st.markdown('<h3>Price Prediction & Market Segmentation</h3>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a module",
    ["🏠 Home", "💰 Price Prediction", "📊 Market Segmentation", "📈 Insights"]
)

# Load models (use cached for performance)
@st.cache_resource
def load_models():
    """Load pre-trained models"""
    try:
        regression_model = joblib.load('best_regression_model.pkl')
        clustering_model = joblib.load('kmeans_clustering_model.pkl')
        cluster_scaler = joblib.load('clustering_scaler.pkl')
        cluster_names = joblib.load('cluster_names.pkl')
        return regression_model, clustering_model, cluster_scaler, cluster_names
    except:
        st.error("Model files not found. Please ensure all model files are in the current directory.")
        return None, None, None, None

# Load models
regression_model, clustering_model, cluster_scaler, cluster_names = load_models()

# Home Page
if app_mode == "🏠 Home":
    st.markdown("""
    ## Welcome to Diamond Dynamics!
    
    This application helps you:
    
    ### 🎯 **Price Prediction Module**
    - Predict diamond prices based on physical attributes
    - Uses advanced machine learning models
    - Provides price in Indian Rupees (INR)
    
    ### 📊 **Market Segmentation Module**
    - Identify which market segment a diamond belongs to
    - Uses K-Means clustering algorithm
    - Provides insights on diamond characteristics
    
    ### 📈 **Insights Module**
    - View data visualizations
    - Understand diamond market trends
    - Explore cluster characteristics
    
    ### How to Use:
    1. Navigate to **Price Prediction** to estimate diamond value
    2. Use **Market Segmentation** to identify market category
    3. Explore **Insights** for market analysis
    """)

# Price Prediction Module
elif app_mode == "💰 Price Prediction":
    st.markdown('<h2 class="sub-header">💎 Diamond Price Prediction</h2>', unsafe_allow_html=True)
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Physical Attributes")
        carat = st.slider("Carat Weight", 0.2, 5.0, 1.0, 0.1)
        x = st.number_input("Length (x in mm)", 0.0, 10.0, 5.0, 0.1)
        y = st.number_input("Width (y in mm)", 0.0, 10.0, 5.0, 0.1)
        z = st.number_input("Depth (z in mm)", 0.0, 10.0, 3.0, 0.1)
        depth = st.slider("Depth Percentage", 40.0, 80.0, 60.0, 0.1)
        table = st.slider("Table Percentage", 40.0, 80.0, 55.0, 0.1)
    
    with col2:
        st.subheader("Quality Attributes")
        cut = st.selectbox("Cut Quality", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
        color = st.selectbox("Color Grade", ["D", "E", "F", "G", "H", "I", "J"])
        clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
        
        # Derived features
        volume = x * y * z
        price_per_carat_pred = st.number_input("Expected Price per Carat (USD)", 1000, 20000, 5000, 100)
        
        # Carat category
        if carat < 0.5:
            carat_category = "Light"
        elif carat <= 1.5:
            carat_category = "Medium"
        else:
            carat_category = "Heavy"
    
    # Calculate derived features
    dimension_ratio = (x + y) / (2 * z) if z > 0 else 0
    surface_area = 2 * (x*y + x*z + y*z)
    density = carat / volume if volume > 0 else 0
    
    # Display derived features
    with st.expander("View Derived Features"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Volume", f"{volume:.2f} mm³")
        with col2:
            st.metric("Dimension Ratio", f"{dimension_ratio:.2f}")
        with col3:
            st.metric("Surface Area", f"{surface_area:.2f} mm²")
    
    # Prediction button
    if st.button("🔮 Predict Price", use_container_width=True):
        if regression_model:
            # Create input dataframe
            input_data = pd.DataFrame({
                'carat': [carat],
                'cut': [cut],
                'color': [color],
                'clarity': [clarity],
                'depth': [depth],
                'table': [table],
                'x': [x],
                'y': [y],
                'z': [z],
                'carat_category': [carat_category],
                'volume': [volume],
                'dimension_ratio': [dimension_ratio],
                'surface_area': [surface_area],
                'density': [density]
            })
            
            try:
                # Make prediction
                predicted_price_usd = regression_model.predict(input_data)[0]
                predicted_price_inr = predicted_price_usd * 83  # Convert to INR
                
                # Display prediction
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Price (USD)", f"${predicted_price_usd:,.2f}")
                with col2:
                    st.metric("Predicted Price (INR)", f"₹{predicted_price_inr:,.2f}")
                
                # Price per carat
                price_per_carat = predicted_price_usd / carat if carat > 0 else 0
                st.metric("Price per Carat", f"${price_per_carat:,.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Price range indicator
                st.subheader("💡 Price Range Indicator")
                
                if predicted_price_usd < 2000:
                    st.success("💰 **Budget Range**: This diamond falls in the affordable price range")
                elif predicted_price_usd < 10000:
                    st.info("💎 **Mid-Range**: This is a reasonably priced diamond with good value")
                else:
                    st.success("👑 **Premium Range**: This is a high-value premium diamond")
                    
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
        else:
            st.error("Regression model not loaded. Please check model files.")

# Market Segmentation Module
elif app_mode == "📊 Market Segmentation":
    st.markdown('<h2 class="sub-header">📊 Diamond Market Segmentation</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Diamond Attributes")
        carat = st.slider("Carat Weight", 0.2, 5.0, 1.0, 0.1, key="cluster_carat")
        cut = st.selectbox("Cut Quality", ["Fair", "Good", "Very Good", "Premium", "Ideal"], key="cluster_cut")
        color = st.selectbox("Color Grade", ["D", "E", "F", "G", "H", "I", "J"], key="cluster_color")
        clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], key="cluster_clarity")
    
    with col2:
        st.subheader("Physical Dimensions")
        x = st.number_input("Length (mm)", 0.0, 10.0, 5.0, 0.1, key="cluster_x")
        y = st.number_input("Width (mm)", 0.0, 10.0, 5.0, 0.1, key="cluster_y")
        z = st.number_input("Depth (mm)", 0.0, 10.0, 3.0, 0.1, key="cluster_z")
        depth = st.slider("Depth %", 40.0, 80.0, 60.0, 0.1, key="cluster_depth")
        table = st.slider("Table %", 40.0, 80.0, 55.0, 0.1, key="cluster_table")
    
    if st.button("🔍 Predict Market Segment", use_container_width=True):
        if clustering_model and cluster_scaler and cluster_names:
            # Encode categorical variables
            cut_mapping = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
            color_mapping = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
            clarity_mapping = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}
            
            # Prepare input for clustering
            input_features = np.array([[
                carat, x, y, z, depth, table,
                cut_mapping[cut], color_mapping[color], clarity_mapping[clarity]
            ]])
            
            # Scale features
            input_scaled = cluster_scaler.transform(input_features)
            
            # Predict cluster
            cluster = clustering_model.predict(input_scaled)[0]
            cluster_name = cluster_names.get(cluster, f"Cluster {cluster}")
            
            # Display results
            st.markdown('<div class="cluster-box">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Market Segment", cluster_name)
                st.metric("Segment Number", cluster)
            
            with col2:
                # Display characteristics based on cluster
                if "Premium" in cluster_name:
                    st.info("✨ **Premium Segment**: High-value diamonds with excellent characteristics")
                    st.write("**Typical Characteristics:**")
                    st.write("- High carat weight")
                    st.write("- Excellent cut quality")
                    st.write("- Premium color grades (D-F)")
                elif "Affordable" in cluster_name:
                    st.success("💰 **Affordable Segment**: Budget-friendly diamonds")
                    st.write("**Typical Characteristics:**")
                    st.write("- Smaller carat weight")
                    st.write("- Good value for money")
                    st.write("- Popular for entry-level purchases")
                else:
                    st.info("⚖️ **Balanced Segment**: Good balance of quality and price")
                    st.write("**Typical Characteristics:**")
                    st.write("- Moderate carat weight")
                    st.write("- Good overall quality")
                    st.write("- Best value proposition")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show cluster distribution
            st.subheader("📈 Market Segment Distribution")
            
            # Create sample distribution (in real app, load from saved data)
            clusters_dist = pd.DataFrame({
                'Segment': ['Premium Heavy', 'High-Quality Premium', 
                          'Mid-range Balanced', 'Affordable Small'],
                'Percentage': [25, 30, 35, 10],
                'Avg Price (USD)': [15000, 8000, 4000, 1500],
                'Avg Carat': [2.1, 1.2, 0.8, 0.4]
            })
            
            # Highlight current cluster
            colors = ['lightblue' if name != cluster_name else 'gold' 
                     for name in clusters_dist['Segment']]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(clusters_dist['Segment'], clusters_dist['Percentage'], 
                          color=colors)
            ax.set_xlabel('Percentage of Market (%)')
            ax.set_title('Diamond Market Segments Distribution')
            ax.bar_label(bars, fmt='%.0f%%')
            
            st.pyplot(fig)
            
        else:
            st.error("Clustering model not loaded. Please check model files.")

# Insights Module
elif app_mode == "📈 Insights":
    st.markdown('<h2 class="sub-header">📈 Diamond Market Insights</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Market Overview", "Price Trends", "Cluster Analysis"])
    
    with tab1:
        st.subheader("💎 Diamond Market Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Price", "$3,928")
            st.caption("Across all diamonds")
        with col2:
            st.metric("Average Carat", "0.80 ct")
            st.caption("Mean carat weight")
        with col3:
            st.metric("Most Common Cut", "Ideal")
            st.caption("53% of diamonds")
        
        # Price distribution
        st.subheader("Price Distribution")
        
        # Sample data for visualization
        price_data = pd.DataFrame({
            'Price Range': ['<$1K', '$1K-$5K', '$5K-$10K', '$10K-$20K', '>$20K'],
            'Percentage': [25, 40, 20, 10, 5]
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(price_data['Price Range'], price_data['Percentage'], color='skyblue')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Diamond Price Distribution')
        ax.bar_label(bars, fmt='%.0f%%')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with tab2:
        st.subheader("📊 Price Trends by Characteristics")
        
        # Create sample trend data
        trend_data = pd.DataFrame({
            'Carat': [0.5, 1.0, 1.5, 2.0, 2.5],
            'Avg Price': [1500, 4000, 9000, 16000, 25000],
            'Price per Carat': [3000, 4000, 6000, 8000, 10000]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.line_chart(trend_data.set_index('Carat')['Avg Price'])
            st.caption("Average Price by Carat Weight")
        
        with col2:
            st.bar_chart(trend_data.set_index('Carat')['Price per Carat'])
            st.caption("Price per Carat by Weight")
        
        # Cut quality impact
        cut_impact = pd.DataFrame({
            'Cut': ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
            'Price Premium': [0, 15, 25, 35, 40],
            'Market Share': [3, 9, 22, 26, 40]
        })
        
        st.subheader("Cut Quality Impact")
        st.dataframe(cut_impact, use_container_width=True)
    
    with tab3:
        st.subheader("🎯 Cluster Characteristics")
        
        # Sample cluster data
        clusters = pd.DataFrame({
            'Segment': ['Premium Heavy Diamonds', 'High-Quality Premium', 
                       'Mid-range Balanced', 'Affordable Small'],
            'Avg Price': ['$15,000', '$8,000', '$4,000', '$1,500'],
            'Avg Carat': ['2.1 ct', '1.2 ct', '0.8 ct', '0.4 ct'],
            'Main Market': ['Luxury Retail', 'High-end Jewelry', 
                           'Mass Market', 'Entry Level'],
            'Typical Customer': ['Collectors', 'Affluent Buyers', 
                                'Middle Class', 'First-time Buyers']
        })
        
        st.dataframe(clusters, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Business Recommendations")
        
        with st.expander("For Retailers"):
            st.write("""
            1. **Premium Segment**: Focus on certification and provenance
            2. **Mid-range**: Highlight value proposition and quality
            3. **Affordable**: Emphasize accessibility and design
            """)
        
        with st.expander("For Buyers"):
            st.write("""
            1. **Investment**: Consider Premium Heavy diamonds
            2. **Value**: Mid-range Balanced offer best value
            3. **Entry**: Affordable Small for first purchases
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>💎 <b>Diamond Dynamics</b> | Price Prediction & Market Segmentation</p>
    <p>Made with ❤️ using Streamlit | Machine Learning | Data Science</p>
</div>
""", unsafe_allow_html=True)
