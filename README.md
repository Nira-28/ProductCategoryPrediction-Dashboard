# 🧠 Product Category Prediction Dashboard

A **Streamlit-based Machine Learning Dashboard** that predicts **Product Categories** based on sales data.  
Users can input **Units Sold**, **Unit Price**, **Region**, and **Payment Method** to get instant predictions.  

---

## 🚀 Features

- 📊 **Interactive Dashboard** using Streamlit  
- 🧮 **Automatic Total Revenue Calculation** (`Units Sold × Unit Price`)  
- 🤖 **Trained Classification Model** for predicting product categories  
- 💾 **Encoders & Model saved** for consistent predictions  
- 🌍 **Supports multiple regions and payment methods**  
- 📈 Real-time visualization of model predictions  

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Joblib** (for model saving)
- **Matplotlib / Seaborn** (optional visualizations)

---

## 📂 Dataset Columns

| Feature | Description |
|----------|-------------|
| Units Sold | Number of items sold |
| Unit Price | Price per item |
| Total Revenue | Calculated automatically |
| Region | Geographical region |
| Payment Method | Type of payment used |
| Product Category | Target variable (predicted) |

---

## ⚙️ How to Run

1️⃣ Clone this repository  
```bash
git clone https://github.com/<your-username>/ProductCategoryPrediction-Dashboard.git
cd ProductCategoryPrediction-Dashboard
