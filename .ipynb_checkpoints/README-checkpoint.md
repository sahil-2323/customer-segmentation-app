# Customer Segementation App

## Project Overview

This project implements Customer Segmentation using K-Means clustering and deploys it as an interactive Streamlit web application.

In addition to clustering existing customers, the app allows users to:

-Predict the cluster of a new customer

-Automatically explain the business meaning of that cluster

-Visualize the new customer directly on the clustering scatter plot

This simulates a real-world analytics dashboard used by marketing and business teams.

## Business Problem

Businesses often struggle to understand diverse customer behaviors.
By segmenting customers based on Annual Income and Spending Score, companies can:

-Identify high-value customers

-Design targeted marketing campaigns

-Optimize pricing and promotions

-Improve customer engagement

This project solves that problem using unsupervised machine learning.

## Machine Learning Approach

-Learning Type: Unsupervised Learning

-Algorithm: K-Means Clustering

-Features Used:

-Annual Income (k$)

 -Spending Score (1–100)
 ### WorkFlow
-Data loading and preprocessing

-Exploratory Data Analysis (EDA)

-Feature scaling

-Optimal cluster selection (Elbow Method)

-K-Means model training

-Cluster interpretation (business meaning)

I-nteractive deployment using Streamlit

## Application Features

-Interactive Streamlit dashboard

-Customer segmentation visualization

-Income vs Spending Score scatter plot

-Cluster-wise customer distribution

-New customer cluster prediction

-Automatic cluster description (business insight)

-Visualization of new customer on cluster plot

## Visualization Highlight

When a new customer is entered:

-The app predicts the customer’s cluster

-Displays the business interpretation

-Plots the customer as a highlighted point on the scatter plot

This helps stakeholders visually understand where the customer belongs.

## Tech Stack

Language: Python

Libraries & Tools: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit