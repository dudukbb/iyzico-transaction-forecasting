# Iyzico Transaction Count Forecasting

This project is an end-to-end time series forecasting and machine learning project. It predicts the daily total transaction count of merchants for the last 3 months of 2020 using historical data.

---

## 📌 Project Overview & Business Problem
* **Business Problem:** iyzico provides payment infrastructure for e-commerce companies, marketplaces, and individual users. The goal of this project is to accurately forecast the daily total transaction count (`Total_Transaction`) for each merchant (`merchant_id`) for the last 3 months of 2020.
* **Model Performance:** Achieved a successful validation error rate of **20.89% SMAPE** using a LightGBM model optimized at the 313th iteration.

---

## 📂 Dataset Story & Features
The dataset contains historical transaction records spanning from 2018 to 2021 across multiple categories and merchant profiles. 

* **Total Size:** 7,667 observations and 5 primary variables.
* **Variables Description:**
  * `transaction_date`: The date of the sales/transaction records.
  * `merchant_id`: Unique identification number for each merchant.
  * `Category`: The business category of the merchant.
  * `Total_Transaction`: The total count of transactions made on that specific day.
  * `Total_Paid`: The total payment amount (price) processed.

---

## ⚙️ Feature Engineering Methodology
To help the machine learning model capture trends, seasonality, and complex patterns, a wide range of features were engineered:
* **Date Features:** Extracted month, day, day of year, week of year, day of week, year, weekend indicator (`is_wknd`), month start/end, quarter, and year start/end indicators. Special calendar events like **Black Friday** and **Summer Solstice** were also incorporated.
* **Lag / Shifted Features:** Created historical lagged features ranging from 91 days up to 722 days to capture annual and periodic patterns.
* **Rolling Mean Features:** Computed rolling window averages to smooth out short-term fluctuations and capture underlying trends.
* **Exponentially Weighted Mean (EWM) Features:** Generated exponentially weighted moving averages using various alpha values and lag combinations.

---

## 🛠️ Tech Stack & Prerequisites
* **Python 3.x**
* **Libraries:**
  * `pandas`, `numpy` (Data manipulation & preprocessing)
  * `lightgbm` (Gradient boosting machine learning model)
  * `matplotlib`, `seaborn` (Exploratory data analysis & visualization)

---

## 🚀 Installation & Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/dudukbb/iyzico-transaction-forecasting.git](https://github.com/dudukbb/iyzico-transaction-forecasting.git)
   cd iyzico-transaction-forecasting
