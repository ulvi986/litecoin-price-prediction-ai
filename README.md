# litecoin-price-prediction-ai
AI-powered prediction and natural language explanation of Litecoin prices using neural networks and real market data.

# 🧠 Litecoin Price Prediction Using Neural Networks and AI Explanations

This repository presents a research project aimed at predicting Litecoin (LTC) prices using machine learning techniques, particularly Multi-Layer Perceptron (MLP) neural networks. The system also integrates OpenAI’s GPT model to generate human-understandable explanations for each prediction.

---

## 📚 Research Objective

The objective of this study is to explore the potential of artificial intelligence in predicting cryptocurrency market prices and to improve interpretability by combining numerical models with natural language generation.

---

## 📊 Abstract

Cryptocurrency markets present challenges such as volatility and rapid price movements, making accurate predictions complex. This research leverages real historical data retrieved from the Binance API to train a regression-based neural network model on features like Open, High, Low, and Volume to predict the Close price of Litecoin.

The resulting model achieved an R² score of **~0.99**, indicating high accuracy. Additionally, the output is passed to **ChatGPT (GPT-4o)**, which provides a natural language explanation of each prediction, enhancing user understanding.

---

## 🛠️ Methodology

1. **Data Source**: Binance Spot API (OHLCV)
2. **Model**: MLPRegressor from scikit-learn
3. **Metrics**: R² score and Cross-Validation
4. **Explanation Layer**: OpenAI ChatGPT (GPT-4o)
5. **Programming Language**: Python

---

## 🧪 Tools & Libraries

- Python 3.x
- pandas, numpy
- scikit-learn
- openai
- requests
- matplotlib (for potential visualization)

---

## 🔬 Research Pipeline

1. **Data Collection** from Binance (1500 daily rows)
2. **Preprocessing & Feature Selection**
3. **Model Training** with MLPRegressor
4. **Evaluation** using R² and cross-validation
5. **ChatGPT Integration** for interpretability
6. **User Interaction** via terminal input (date selection)

---

## 🧠 Sample Output

```bash
📅 Date: 2025-04-15
🤖 Predicted Price: $86.75
✅ Actual Price: $87.20

🧠 GPT-4o Explanation:
Based on the input values such as opening price, daily high and low, and the trading volume, the AI model predicted Litecoin's closing price with high confidence. The volume indicates strong market activity, supporting the predicted trend.
