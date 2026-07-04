import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error



st.set_page_config(
    page_title="Advertising Sales Prediction",
    page_icon="📈",
    layout="wide"
)



st.markdown("""
<style>



.stApp{
    background: linear-gradient(135deg,#0f172a,#1e293b,#334155);
}

/* Hide Streamlit Elements */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}


html, body, div, p, span, label, li {
    color: #f8fafc !important;
}

/* Headings */
h1,h2,h3,h4,h5,h6{
    color:#38bdf8 !important;
    font-weight:700 !important;
}


.main-title{
    text-align:center;
    font-size:52px;
    font-weight:800;
    color:#38bdf8 !important;
}

.sub-title{
    text-align:center;
    font-size:18px;
    color:#cbd5e1 !important;
}



section[data-testid="stSidebar"]{
    background:#111827;
}

section[data-testid="stSidebar"] *{
    color:white !important;
}



[data-testid="stMetricLabel"]{
    color:white !important;
}

[data-testid="stMetricValue"]{
    color:#38bdf8 !important;
    font-weight:bold;
}



button[role="tab"]{
    color:white !important;
}

button[aria-selected="true"]{
    color:#38bdf8 !important;
}



.stRadio label{
    color:white !important;
}



.stNumberInput label,
.stSelectbox label,
.stFileUploader label{
    color:white !important;
}



[data-testid="stDataFrame"] *{
    color:black !important;
}



[data-testid="stAlertContainer"]{
    color:white !important;
}



.glass{
    background:rgba(255,255,255,0.08);
    backdrop-filter:blur(15px);
    border-radius:20px;
    padding:20px;
    border:1px solid rgba(255,255,255,0.15);
    box-shadow:0 8px 20px rgba(0,0,0,0.25);
}



[data-testid="stFileUploader"]{
    background: rgba(37,99,235,0.15);
    border: 2px dashed #38bdf8;
    border-radius: 15px;
    padding: 15px;
}

/* Drag & Drop Area */
[data-testid="stFileUploaderDropzone"]{
    background: rgba(15,23,42,0.8);
    border: 2px dashed #38bdf8;
    border-radius: 15px;
}

/* Upload Text */
[data-testid="stFileUploaderDropzone"] *{
    color: white !important;
}

/* Browse Files Button */
[data-testid="stFileUploaderDropzone"] button{
    background: linear-gradient(135deg,#2563eb,#06b6d4);
    color: white !important;
    border-radius: 10px;
    border: none;
    font-weight: bold;
}

[data-testid="stFileUploaderDropzone"] button:hover{
    background: linear-gradient(135deg,#1d4ed8,#0891b2);
}
            


.metric-card{
    background:linear-gradient(135deg,#2563eb,#06b6d4);
    border-radius:18px;
    padding:20px;
    text-align:center;
    color:white !important;
    box-shadow:0px 6px 15px rgba(0,0,0,0.3);
}

.metric-value{
    font-size:30px;
    font-weight:bold;
    color:white !important;
}

.metric-label{
    font-size:16px;
    color:white !important;
}

</style>
""", unsafe_allow_html=True)



st.markdown("""
<h1 class="main-title">
📈 Advertising Sales Prediction Dashboard
</h1>

<p class="sub-title">
Marketing Analytics & Sales Forecasting
</p>
""", unsafe_allow_html=True)



st.sidebar.title("📊 Dashboard Menu")

menu = st.sidebar.radio(
    "Navigate",
    [
        "Project Overview",
        "Dataset Analysis",
        "Model Performance",
        "Sales Prediction",
        "Business Insights"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Advertising.csv",
    type=["csv"]
)



if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)

    

    if menu == "Project Overview":

        st.markdown("""
        <div class="glass">
        <h2>📝 Project Objectives</h2>

        <ul>
        <li>Predict future sales based on advertising spend.</li>
        <li>Prepare data through cleaning, transformation and feature selection.</li>
        <li>Use regression models to forecast sales performance.</li>
        <li>Analyze how advertising investments impact sales.</li>
        <li>Identify the most effective marketing channels.</li>
        <li>Deliver actionable insights for marketing strategies.</li>
        <li>Support business decision-making using predictive analytics.</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class='metric-card'>
            <div class='metric-label'>📺 TV Budget Avg</div>
            <div class='metric-value'>{df['TV'].mean():.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='metric-card'>
            <div class='metric-label'>📻 Radio Avg</div>
            <div class='metric-value'>{df['Radio'].mean():.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class='metric-card'>
            <div class='metric-label'>📰 Newspaper Avg</div>
            <div class='metric-value'>{df['Newspaper'].mean():.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class='metric-card'>
            <div class='metric-label'>💰 Sales Avg</div>
            <div class='metric-value'>{df['Sales'].mean():.2f}</div>
            </div>
            """, unsafe_allow_html=True)


    elif menu == "Dataset Analysis":

        st.subheader("📋 Dataset Preview")
        st.dataframe(df, use_container_width=True)

        st.subheader("🧹 Data Cleaning Summary")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

        c2.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )

        c3.metric(
            "Total Features",
            len(df.columns)-1
        )

        st.subheader("📊 Advertising vs Sales")

        tab1, tab2, tab3 = st.tabs(
            ["TV", "Radio", "Newspaper"]
        )

        with tab1:
            fig = px.scatter(
                df,
                x="TV",
                y="Sales",
                trendline="ols",
                title="TV vs Sales"
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = px.scatter(
                df,
                x="Radio",
                y="Sales",
                trendline="ols",
                title="Radio vs Sales"
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            fig = px.scatter(
                df,
                x="Newspaper",
                y="Sales",
                trendline="ols",
                title="Newspaper vs Sales"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔥 Correlation Heatmap")

        fig = px.imshow(
            df.corr(),
            text_auto=True,
            color_continuous_scale="RdBu_r"
        )

        st.plotly_chart(fig, use_container_width=True)

   

    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    

    if menu == "Model Performance":

        st.subheader("🤖 Regression Model Performance")

        c1, c2, c3 = st.columns(3)

        c1.metric("R² Score", f"{r2:.4f}")
        c2.metric("MAE", f"{mae:.4f}")
        c3.metric("RMSE", f"{rmse:.4f}")

        result_df = pd.DataFrame({
            "Actual Sales": y_test,
            "Predicted Sales": y_pred
        })

        fig = px.scatter(
            result_df,
            x="Actual Sales",
            y="Predicted Sales",
            title="Actual vs Predicted Sales"
        )

        st.plotly_chart(fig, use_container_width=True)

        coef_df = pd.DataFrame({
            "Channel": X.columns,
            "Impact": model.coef_
        })

        fig = px.bar(
            coef_df,
            x="Channel",
            y="Impact",
            text="Impact",
            title="Advertising Impact Analysis"
        )

        st.plotly_chart(fig, use_container_width=True)


    elif menu == "Sales Prediction":

        st.markdown("""
        <div class='glass'>
        <h2>🔮 Future Sales Forecast</h2>
        <p>Enter advertising budgets to predict future sales.</p>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        col1, col2, col3 = st.columns(3)

        tv = col1.number_input(
            "TV Advertising Budget",
            min_value=0.0,
            value=200.0
        )

        radio = col2.number_input(
            "Radio Advertising Budget",
            min_value=0.0,
            value=30.0
        )

        newspaper = col3.number_input(
            "Newspaper Budget",
            min_value=0.0,
            value=20.0
        )

        if st.button(
            "🚀 Predict Sales",
            use_container_width=True
        ):

            prediction = model.predict(
                [[tv, radio, newspaper]]
            )[0]

            st.markdown(f"""
            <div style="
            background:linear-gradient(135deg,#22c55e,#16a34a);
            padding:25px;
            border-radius:20px;
            text-align:center;
            color:white;
            font-size:30px;
            font-weight:bold;">
            📈 Predicted Sales<br><br>
            {prediction:.2f} Units
            </div>
            """, unsafe_allow_html=True)

            st.balloons()


    elif menu == "Business Insights":

        coef_df = pd.DataFrame({
            "Channel": X.columns,
            "Impact": model.coef_
        })

        best_channel = coef_df.sort_values(
            by="Impact",
            ascending=False
        ).iloc[0]["Channel"]

        st.success(
            f"🏆 Highest Impact Advertising Channel: {best_channel}"
        )

        st.info(f"""
### ✨Key Findings

• {best_channel} contributes the most to sales growth.

• Higher advertising budgets generally lead to increased sales.

• Data-driven marketing improves ROI.

• Predictive analytics helps optimize campaign performance.

• Budget allocation should prioritize high-performing channels.

• Forecasting enables better business planning.
""")

        st.markdown("""
### 🎯 Marketing Recommendations

1. Increase investment in high-performing advertising channels.

2. Continuously monitor campaign effectiveness.

3. Optimize spending on low-performing channels.

4. Forecast future sales before campaign launches.

5. Track ROI and customer engagement regularly.

6. Use machine learning insights for strategic planning.

7. Improve marketing efficiency through data analytics.
""")

else:

    st.info("📂 Upload Advertising.csv from the sidebar to start analysis.")