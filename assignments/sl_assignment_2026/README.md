# SL Assignment 2026

This folder contains two Python solutions for the assignment tasks.

## 1. LSTM Sine/Cosine Wave Prediction

Creates 1000 random sine and cosine sequences, trains an LSTM with 100 hidden units for a maximum of 10 epochs, then predicts the held-out test wave.

```powershell
python assignments/sl_assignment_2026/lstm_wave_prediction.py
```

Outputs:

- `wave_prediction.png`: true vs predicted test wave plot
- Console metrics: final training loss, validation loss, and test MSE

## 2. TextBlob Sentiment Analysis For A Large Text File

Analyzes a text file in chunks and classifies the overall sentiment as positive, neutral, or negative.

```powershell
python assignments/sl_assignment_2026/textblob_sentiment_large_file.py --file path\to\large_text_file.txt
```

Optional:

```powershell
python assignments/sl_assignment_2026/textblob_sentiment_large_file.py --file path\to\large_text_file.txt --chunk-size 8000
```

Outputs:

- `sentiment_analysis.png`: Matplotlib bar chart for positive, neutral, and negative chunks
- Console results: chunk counts, polarity, subjectivity, and final sentiment

## Dependencies

Install the assignment dependencies with:

```powershell
pip install -r assignments/sl_assignment_2026/requirements_assignment.txt
```
