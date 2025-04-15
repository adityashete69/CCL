import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page Title
st.set_page_config(page_title="Customer Insights & Prediction", layout='wide')
st.title("💬 Customer Satisfaction Insights & Prediction")

# Session State for 'Get Started' Button
if 'started' not in st.session_state:
    st.session_state['started'] = False

# Home Page
st.markdown("<div id='home'></div>", unsafe_allow_html=True)
st.subheader("🌟 Welcome to Customer Satisfaction Prediction System")

# "Get Started" Button in the Center
if st.button("🚀 Get Started"):
    st.session_state['started'] = True

# Main Content Only After Clicking 'Get Started'
if st.session_state['started']:

    # Navigation Bar
    st.markdown("""
        <style>
            .navbar {
                background-color: #4CAF50;
                overflow: hidden;
                display: flex;
                justify-content: center;
            }
            .navbar a {
                float: left;
                display: block;
                color: white;
                text-align: center;
                padding: 14px 20px;
                text-decoration: none;
                font-weight: bold;
                cursor: pointer;
            }
            .navbar a:hover {
                background-color: #45a049;
            }
        </style>
        <div class="navbar">
            <a href="#visualizations">📊 Visualizations</a>
            <a href="#scatter_plot">📌 Customer Segmentation</a>
            <a href="#prediction">🤖 Prediction System</a>
        </div>
    """, unsafe_allow_html=True)

    # Load Data
    @st.cache_data
    def load_data():
        try:
            data = pd.read_csv('customer_support_tickets.csv')
            st.success("✅ Dataset loaded successfully.")
            return data
        except FileNotFoundError:
            st.error("❌ Error: Dataset not found. Please ensure 'customer_support_tickets.csv' is in the correct location.")
            st.stop()

    data = load_data()

    # Data Cleaning
    def clean_data(data):
        drop_cols = ['Ticket ID', 'Customer Name', 'Customer Email', 'Ticket Description', 'Resolution', 'Date of Purchase']
        data_cleaned = data.drop(columns=drop_cols, errors='ignore')
        data_cleaned['Customer Satisfaction Rating'].fillna(data_cleaned['Customer Satisfaction Rating'].median(), inplace=True)

        if 'Response Time (hrs)' not in data_cleaned.columns:
            data_cleaned['First Response Time'] = pd.to_datetime(data_cleaned['First Response Time'], errors='coerce')
            data_cleaned['Time to Resolution'] = pd.to_datetime(data_cleaned['Time to Resolution'], errors='coerce')
            data_cleaned['Response Time (hrs)'] = (
                (data_cleaned['Time to Resolution'] - data_cleaned['First Response Time']).dt.total_seconds() / 3600
            )
            data_cleaned['Response Time (hrs)'].fillna(data_cleaned['Response Time (hrs)'].median(), inplace=True)

        return data_cleaned

    data_cleaned = clean_data(data)

  # Align Main Features, About Us, and Contact Us in Same Row
    st.markdown("""
        <div style="display: flex; justify-content: space-around; align-items: center; margin-top: 20px;">
            <div>
                <h3>🌟 Main Features</h3>
                <img src='images/features.WEBP' width='300'>
                <p>✅ Interactive Visualizations<br>✅ Advanced Prediction Model<br>✅ Customer Segmentation</p>
            </div>
            <div>
                <h3>📧 Contact Us</h3>
                <img src='images/contact_us.jpg' width='300'>
                <p>📞 +1 234 567 890<br>📧 support@customersatisfaction.com</p>
            </div>
            <div>
                <h3>ℹ️ About Us</h3>
                <img src='images/about_us.jpg' width='300'>
                <p>Helping businesses improve customer retention through data insights.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Visualizations
    st.markdown("<div id='visualizations'></div>", unsafe_allow_html=True)
    st.subheader("📊 Data Visualizations")

    visualization_options = [
        "Customer Satisfaction Distribution",
        "Ticket Channel Distribution",
        "Customer Gender Distribution",
        "Ticket Type Distribution",
        "Tickets Raised by Age Groups",
        "Customer Ticket Trend Over Time",
        "Top 10 Products Purchased",
        "Top Items Purchased by Gender",
        "Average Satisfaction by Gender",
        "Response Time Distribution",
        "Customer Age Distribution",
        "Satisfaction vs Response Time",
        "Satisfaction vs Customer Age"
    ]

    for viz_option in visualization_options:
        if st.checkbox(viz_option):
            if viz_option == "Customer Satisfaction Distribution":
                sns.histplot(data_cleaned['Customer Satisfaction Rating'], kde=True, bins=5)
                st.pyplot(plt)

            elif viz_option == "Ticket Channel Distribution":
                sns.countplot(x='Ticket Channel', data=data_cleaned)
                st.pyplot(plt)

            elif viz_option == "Satisfaction vs Customer Age":
                sns.scatterplot(x='Customer Age', y='Customer Satisfaction Rating', data=data_cleaned)
                st.pyplot(plt)

    # Customer Segmentation (3D Scatter Plot)
    st.markdown("<div id='scatter_plot'></div>", unsafe_allow_html=True)
    st.subheader("🔷 Customer Segmentation Based on Satisfaction")

    if st.checkbox("🔎 View 3D Plot"):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_cleaned[['Customer Age', 'Customer Satisfaction Rating']])

        kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        data_cleaned['Customer Segments'] = kmeans.fit_predict(data_scaled)

        segment_labels = {0: 'Low Satisfaction', 1: 'Moderate Satisfaction', 2: 'High Satisfaction'}
        data_cleaned['Customer Segments'] = data_cleaned['Customer Segments'].map(segment_labels)

        plot = go.Figure()

        for segment in data_cleaned['Customer Segments'].unique():
            segment_data = data_cleaned[data_cleaned['Customer Segments'] == segment]
            plot.add_trace(go.Scatter3d(
                x=segment_data['Customer Age'],
                y=[0] * len(segment_data),
                z=segment_data['Customer Satisfaction Rating'],
                mode='markers',
                marker=dict(size=6, line=dict(width=1)),
                name=str(segment)
            ))

        plot.update_layout(
            width=800,
            height=800,
            autosize=True,
            scene=dict(
                xaxis_title='Customer Age',
                zaxis_title='Satisfaction Rating'
            )
        )

        st.plotly_chart(plot, use_container_width=True)

    # Prediction System
    st.markdown("<div id='prediction'></div>", unsafe_allow_html=True)
    st.subheader("🤖 Prediction System")
    response_time = st.number_input("Response Time (hrs)", value=data_cleaned['Response Time (hrs)'].median())
    customer_age = st.number_input("Customer Age", value=25)

    if st.button("🔍 Predict Satisfaction"):
        X = data_cleaned[['Response Time (hrs)', 'Customer Age']]
        y = data_cleaned['Customer Satisfaction Rating'].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = HistGradientBoostingClassifier(max_iter=100, max_depth=8, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)

        sample_input = pd.DataFrame([[response_time, customer_age]], columns=['Response Time (hrs)', 'Customer Age'])
        prediction = model.predict(sample_input)[0]
        st.success(f'Predicted Customer Satisfaction Rating: {prediction}')

  
