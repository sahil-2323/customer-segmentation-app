import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Segmentation App", layout="wide")
st.title("Customer Segmentation Dashboard")
st.amrkdown("### K-Means Clustering | Data Science Project")

@st.cache_data
def load_data():
    return pd.read_csv("customer_segmentation_output.csv")

df= load_data()

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

st.subheader("Income vs Spending Score")

fig1, ax1 = plt.subplots()
sns.scatterplot(x="Annual_Income_k$", y="Spending_Score", hue="Clusters", data=df, palette="Set2", ax=ax1)
ax1.set_xlabel("Annual_Income_k$")
ax1.set_ylabel("Spending Score")
st.pyplot(fig1)

col5, col6 =st.columns(2)

with col5:
    st.subheader("Customer Distribution by Clusters")
    clusetr_counts= filtered_df["Clusters"].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.pie(clusetr_counts, labels= clusetr_counts.index, autopct="%1.1f%")
    st.pyplot(fig2)
    
with col6:
    st.subheader("Gender Distribution")
    fig3, ax3= plt.subplots()
    sns.countplot(x="Gender", data=filtered_df, ax=ax3)
    st.pyplot(fig3)

st.divider()

st.subheader("Customer Data")
st.dataframe(filtered_df)

st.markdown("---")
st.markdown("**Project by Sahil Mahto**")