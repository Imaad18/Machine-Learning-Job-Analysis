import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from collections import Counter
import re
import numpy as np

# Set page config
st.set_page_config(
    page_title="ML Jobs Data Analysis",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Machine Learning Jobs Analysis Dashboard")
st.markdown("This dashboard analyzes trends in ML job postings across the US.")

# File uploader
uploaded_file = st.file_uploader("Upload your ML jobs CSV file", type=["csv"])

# Initialize session state if not already done
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Function to process the data
def process_data(df):
    # Convert posting date to datetime
    df["job_posted_date"] = pd.to_datetime(df["job_posted_date"], errors='coerce')
    
    # Feature Engineering: Extract Year, Month, Day, Weekday
    df["year"] = df["job_posted_date"].dt.year
    df["month"] = df["job_posted_date"].dt.month_name()
    df["day"] = df["job_posted_date"].dt.day
    df["weekday"] = df["job_posted_date"].dt.day_name()
    
    return df

# Main application logic
if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    df = process_data(df)
    st.session_state.data_loaded = True
    st.session_state.df = df
    
    # Show success message
    st.success("Data loaded successfully!")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Data Overview", "Temporal Analysis", "Location & Companies", "Job Titles Analysis"])

# Check if data is loaded before displaying visualizations
if st.session_state.data_loaded:
    df = st.session_state.df
    
    if page == "Data Overview":
        st.header("Basic Data Overview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Information")
            st.write(f"Total records: {df.shape[0]}")
            st.write(f"Number of columns: {df.shape[1]}")
        
        with col2:
            st.subheader("Missing Values")
            st.write(df.isnull().sum())
        
        st.subheader("Sample Data")
        st.dataframe(df.head())
        
        st.subheader("Data Types")
        st.write(df.dtypes)
        
    elif page == "Temporal Analysis":
        st.header("Temporal Analysis of Job Postings")
        
        # Job Postings per Month
        st.subheader("Job Postings per Month")
        month_order = ["January", "February", "March", "April", "May", "June", 
                      "July", "August", "September", "October", "November", "December"]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(data=df, x="month", order=month_order, palette="viridis", ax=ax)
        plt.xticks(rotation=45)
        plt.xlabel("Month")
        plt.ylabel("Number of Jobs")
        plt.tight_layout()
        st.pyplot(fig)
        
        # Job Postings per Weekday
        st.subheader("Job Postings by Day of the Week")
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.countplot(data=df, x="weekday", order=weekday_order, palette="coolwarm", ax=ax)
        plt.xlabel("Weekday")
        plt.ylabel("Number of Jobs")
        plt.tight_layout()
        st.pyplot(fig)
        
        # Job Postings per Day of Month
        st.subheader("Job Postings per Day of Month")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(data=df, x="day", bins=31, kde=False, color="steelblue", ax=ax)
        plt.xlabel("Day of Month")
        plt.ylabel("Number of Jobs")
        plt.tight_layout()
        st.pyplot(fig)
        
        # Heatmap: Jobs by Month and Weekday
        st.subheader("Heat Map: Job Postings by Weekday and Month")
        heatmap_data = df.groupby(["month", "weekday"]).size().reset_index(name='count')
        pivot_table = heatmap_data.pivot("weekday", "month", "count").reindex(index=weekday_order, columns=month_order)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(pivot_table, annot=True, fmt='g', cmap="YlGnBu", ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Trend Over Time (Interactive)
        st.subheader("Job Posting Trend Over Time")
        monthly_trend = df['job_posted_date'].dt.to_period('M').value_counts().sort_index()
        monthly_trend.index = monthly_trend.index.astype(str)
        
        fig = px.line(
            x=monthly_trend.index, 
            y=monthly_trend.values,
            labels={'x': 'Month', 'y': 'Number of Jobs'},
            title='Monthly Job Posting Trend'
        )
        fig.update_traces(mode="lines+markers")
        st.plotly_chart(fig, use_container_width=True)
        
    elif page == "Location & Companies":
        st.header("Top Locations and Companies")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Job Locations")
            top_locations = df["company_address_locality"].value_counts().head(10)
            st.bar_chart(top_locations)
            
            # Show table of values
            st.table(top_locations.reset_index().rename(columns={"index": "Location", "company_address_locality": "Count"}))
        
        with col2:
            st.subheader("Top Hiring Companies")
            top_companies = df["company_name"].value_counts().head(10)
            st.bar_chart(top_companies)
            
            # Show table of values
            st.table(top_companies.reset_index().rename(columns={"index": "Company", "company_name": "Count"}))
            
        # Plotly choropleth if data available
        if "company_address_region" in df.columns:
            st.subheader("Jobs by State")
            state_counts = df["company_address_region"].value_counts().reset_index()
            state_counts.columns = ['state', 'count']
            
            fig = px.choropleth(
                state_counts,
                locations='state',
                locationmode="USA-states",
                color='count',
                scope="usa",
                color_continuous_scale="Viridis",
                title="Job Distribution by State"
            )
            st.plotly_chart(fig, use_container_width=True)
            
    elif page == "Job Titles Analysis":
        st.header("Job Titles Analysis")
        
        # Top Job Titles
        st.subheader("Top Job Titles")
        top_titles = df["job_title"].value_counts().head(10)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(y=top_titles.index, x=top_titles.values, palette="mako", ax=ax)
        plt.xlabel("Count")
        plt.ylabel("Job Title")
        plt.tight_layout()
        st.pyplot(fig)
        
        # Top Words in Job Titles
        st.subheader("Most Common Words in Job Titles")
        
        # Extract words from job titles
        text = ' '.join(df['job_title'].dropna().astype(str).tolist())
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter common stop words
        stop_words = ['and', 'the', 'in', 'of', 'to', 'for', 'a', 'with']
        words = [word for word in words if word not in stop_words]
        
        word_freq = Counter(words)
        top_words = pd.DataFrame(word_freq.most_common(20), columns=['Word', 'Frequency'])
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x="Frequency", y="Word", data=top_words, palette="mako", ax=ax)
        plt.title("Top 20 Frequent Words in Job Titles")
        plt.tight_layout()
        st.pyplot(fig)
        
        # Check if seniority level is available
        if 'seniority_level' in df.columns:
            st.subheader("Seniority Level Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(data=df, y="seniority_level", 
                         order=df["seniority_level"].value_counts().index, 
                         palette="Set2", ax=ax)
            plt.xlabel("Count")
            plt.ylabel("Seniority Level")
            plt.tight_layout()
            st.pyplot(fig)

else:
    st.info("Please upload a CSV file to get started.")
    
    # Display sample data format
    st.subheader("Expected Data Format")
    st.markdown("""
    Your CSV file should have the following columns:
    - job_title: Title of the job posting
    - company_name: Name of the hiring company
    - job_posted_date: Date when the job was posted (in a format convertible to datetime)
    - company_address_locality: City or locality of the job
    
    Optional columns:
    - seniority_level: Level of seniority for the position
    - company_address_region: State or region of the job
    """)
    
    # Display a sample dataframe
    sample_data = {
        'job_title': ['Data Scientist', 'ML Engineer', 'AI Researcher'],
        'company_name': ['Tech Co', 'AI Solutions', 'Research Labs'],
        'job_posted_date': ['2023-01-15', '2023-02-20', '2023-03-10'],
        'company_address_locality': ['San Francisco', 'New York', 'Boston']
    }
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("ML Jobs Analysis Dashboard built with Streamlit")
