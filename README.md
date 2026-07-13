# 🚀 Intelligent Sales Forecasting & Analytics Dashboard

**Live Demo:** [View the Streamlit App Here](https://sales-forecasting-app-bjwfg7z224zdfl7k5sps7e.streamlit.app/)

Welcome to the Intelligent Sales Forecasting & Analytics Dashboard! This project leverages Machine Learning to transform raw transactional data into actionable business intelligence. It is designed to help supply chain managers and business owners make data-driven decisions regarding inventory management and sales strategies.

## 🌟 Key Features

1. **📊 Sales Overview**
   - High-level executive summary of revenue growth, historical trends, and category performance.
   - Built with interactive Plotly charts for deep-dive explorations.

2. **🔮 Forecast Explorer (Facebook Prophet)**
   - Utilizes advanced time-series forecasting using Facebook's Prophet model.
   - Predicts future sales demand for specific product categories to prevent stockouts and overstocking.
   - Includes Confidence Intervals to show best-case and worst-case scenarios.

3. **🚨 Anomaly Detection (Isolation Forest)**
   - Employs Scikit-Learn's `Isolation Forest` algorithm to automatically flag unusual sales spikes or drops.
   - Helps identify supply chain disruptions, missed opportunities, or massive bulk orders.

4. **🎯 Product Segmentation (K-Means Clustering)**
   - Automatically groups products into strategic clusters (e.g., *Stable High Volume*, *High Volatility High Value*) using K-Means and PCA.
   - Provides tailored supply-chain rules (like bulk-ordering vs. flexible safety stock) based on the exact mathematical behavior of the products.

## 📂 Dataset Information

This project is built using the **Superstore Sales Dataset** (`train.csv`), which contains 4 years (2015-2018) of daily transactional records for a retail store. 

- **Why this dataset was chosen:** To perform highly accurate Machine Learning tasks like Time-Series Forecasting (Prophet) and Anomaly Detection, the algorithm requires precise, granular dates and daily/monthly fluctuations. The Superstore dataset provides exact order dates and detailed product categories, making it the perfect foundation for these advanced analytics.
- **Key Features Used:** `Order Date` (for time-series index), `Sales` (for predicting volume), `Sub-Category` (for segmentation), and `Region`.

## 🛠️ Technology Stack
- **Frontend & UI:** [Streamlit](https://streamlit.io/) (with Custom CSS styling for adaptive Light/Dark modes)
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** `prophet` (Forecasting), `scikit-learn` (Clustering & Anomaly Detection)
- **Data Visualization:** Plotly Express, Plotly Graph Objects

## 📁 Repository Structure
- `app.py`: The core Streamlit application containing the dashboard UI, plotting logic, and live model execution.
- `analysis.ipynb`: A comprehensive Jupyter Notebook detailing the Exploratory Data Analysis (EDA), model training, and mathematical validation for the algorithms used in the app.
- `train.csv`: The Superstore sales dataset used to train the models and power the dashboard.
- `requirements.txt`: The list of Python dependencies required to run the application.

## 🚀 How to Run Locally

If you want to run this dashboard on your own machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## 📝 Author
Developed as a comprehensive data science and machine learning project focusing on time-series analytics and interactive dashboarding.
