# 🌮 Taco Delivery Data Analysis

## Project Overview

This project analyses a taco delivery dataset using Python and Pandas.

The goal is to explore customer ordering behaviour, delivery performance, pricing, and tipping patterns through Exploratory Data Analysis (EDA).

---

## Dataset

The dataset contains 1,000 taco delivery orders including:

- Restaurant
- City
- Order Time
- Delivery Time
- Delivery Duration
- Taco Size
- Taco Type
- Number of Toppings
- Delivery Distance
- Price
- Tip Amount
- Weekend Indicator

---

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook

---

## Data Cleaning

Performed the following preprocessing steps:

- Converted dates to datetime format
- Created Order Hour feature
- Calculated delivery duration
- Checked missing values
- Checked duplicate records
- Verified data types

---

## Business Questions

The analysis answers the following questions:

- What are the most popular taco types?
- Which taco size is ordered the most?
- What is the average delivery duration?
- Does delivery duration depend on taco type or size?
- Which restaurants are the fastest?
- Which restaurants are the slowest?
- How many toppings do customers usually choose?
- How far do deliveries usually travel?
- What is the average order price?
- Is there a relationship between price and tip?
- Do customers tip more on weekends?
- Does taco size or toppings affect price?
- Are weekends busier than weekdays?
- What time of day has the most orders?
- Is delivery slower on weekends?
- Do customers choose more toppings on weekends?

---

## Key Insights

Examples:

- Chicken tacos are the most popular.
- Most customers choose Regular size tacos.
- The average delivery time is approximately XX minutes.
- Weekend orders increase compared to weekdays.
- Larger tacos generally cost more.
- Tips increase with order price.

*(Replace these with your actual findings.)*

---

## Project Structure

```
Taco-Delivery-Analysis/
│
├── data/
│   └── taco_delivery.csv
│
├── notebook/
│   └── Taco_EDA.ipynb
│
├── images/
│   ├── orders_by_hour.png
│   ├── price_vs_tip.png
│   ├── delivery_time.png
│   └── toppings_distribution.png
│
├── README.md
└── requirements.txt
```

---

## How to Run

```bash
git clone https://github.com/yourusername/taco-delivery-analysis.git

cd taco-delivery-analysis

pip install -r requirements.txt

jupyter notebook
```

---

## Future Improvements

- Build a dashboard using Power BI or Tableau
- Create an interactive dashboard with Streamlit
- Train a machine learning model to predict delivery time
- Predict customer tips
- Deploy the project online

---

## Author

Adel

Python | Data Analysis | Machine Learning

