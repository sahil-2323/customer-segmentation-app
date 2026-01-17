import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Segmentation App", layout="wide")
st.title("Customer Segmentation Dashboard")
st.markdown("### K-Means Clustering | Data Science Project")

@st.cache_data
def load_data():
    return pd.read_csv("customer_segmentation_output.csv")

df= load_data()

CLUSTER_DESCRIPTIONS = {
    0: "Low Income, Low Spending – Budget-conscious customers",
    1: "High Income, Low Spending – Careful and conservative customers",
    2: "Low Income, High Spending – Impulsive buyers, promotion-sensitive",
    3: "High Income, High Spending – Premium and high-value customers",
    4: "Average Income, Average Spending – Standard customers"
}

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

@st.cache_resource
def train_kmeans(data):
    X = data[["Annual_Income_k$", "Spending_Score"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X_scaled)

    return kmeans, scaler

kmeans_model, scaler = train_kmeans(df)

st.sidebar.header("Filter Options")

gender_filter= st.sidebar.multiselect("Select Gender",options=df["Gender"].unique(), default=df["Gender"].unique())
cluster_filter= st.sidebar.multiselect("Select Cluster", options=df["Clusters"].unique(), default=df["Clusters"].unique())
filtered_df= df[(df["Gender"].isin(gender_filter)) & (df["Clusters"].isin(cluster_filter))]

col1, col2, col3, col4= st.columns(4)
col1.metric("Total Customers", filtered_df.shape[0])
col2.metric("Avg Income (k$)", round(filtered_df["Annual_Income_k$"].mean(), 2))
col3.metric("Avg Spending Score", round(filtered_df["Spending_Score"].mean(), 2))
col4.metric("Total Segements", filtered_df["Clusters"].nunique())

st.divider()

st.subheader("Income vs Spending Score (with New Customer)")

fig, ax = plt.subplots()

# Existing customers
sns.scatterplot(
    x="Annual_Income_k$",
    y="Spending_Score",
    hue="Clusters",
    data=df,
    palette="Set2",
    ax=ax,
    alpha=0.7
)

# New customer overlay
if "new_customer" in st.session_state:
    ax.scatter(
        st.session_state["new_customer"]["income"],
        st.session_state["new_customer"]["spending"],
        color="red",
        s=200,
        marker="X",
        label="New Customer"
    )

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score")
ax.legend()
st.pyplot(fig)

col5, col6 =st.columns(2)

with col5:
    st.subheader("Customer Distribution by Clusters")
    cluster_counts= filtered_df["Clusters"].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.pie(cluster_counts, labels= cluster_counts.index, autopct="%1.1f%%")
    st.pyplot(fig2)
    
with col6:
    st.subheader("Gender Distribution")
    fig3, ax3= plt.subplots()
    sns.countplot(x="Gender", data=filtered_df, ax=ax3)
    st.pyplot(fig3)

st.divider()

st.subheader("Customer Data")
st.dataframe(filtered_df)

st.divider()
st.subheader("Predict Cluster for a New Customer")

col1, col2 = st.columns(2)

with col1:
    new_income = st.number_input(
        "Annual Income (k$)",
        min_value=0.0,
        max_value=200.0,
        value=50.0
    )

with col2:
    new_spending = st.number_input(
        "Spending Score (1–100)",
        min_value=1.0,
        max_value=100.0,
        value=50.0
    )

if st.button("Predict Customer Segment"):
    new_customer = [[new_income, new_spending]]
    new_customer_scaled = scaler.transform(new_customer)

    predicted_cluster = kmeans_model.predict(new_customer_scaled)[0]

    st.success(f"Predicted Cluster: {predicted_cluster}")
    st.info(f"Segment Description: {CLUSTER_DESCRIPTIONS[predicted_cluster]}")

    # Save for visualization
    st.session_state["new_customer"] = {
        "income": new_income,
        "spending": new_spending,
        "cluster": predicted_cluster
    }

st.markdown("---")
st.markdown("**Project by Sahil Mahto**")