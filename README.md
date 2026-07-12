# Fake News Detection using Machine Learning and Python

This is a simple full-stack final year project that predicts whether a news text is **Fake** or **Real**.
The project uses a machine learning model with a Flask backend and basic HTML/CSS/JS frontend.

## Project Overview

The main idea is:
- train a text classification model on fake and true news datasets
- use text preprocessing (lowercase, stopword removal, stemming)
- transform text using TF-IDF
- call the model through Flask API endpoints
- show result + confidence on a web page
- store predictions in SQLite database

## Technologies Used

- Python
- Flask
- pandas, numpy
- scikit-learn
- nltk
- SQLite
- HTML, CSS, JavaScript

## Folder Structure

```text
FakeNewsProject/
│
├── Fake.csv
├── True.csv
├── requirements.txt
├── README.md
│
├── model/
│   ├── __init__.py
│   ├── text_preprocessing.py
│   ├── train_model.py
│   └── artifacts/                 # generated after training
│       ├── fake_news_model.pkl
│       └── tfidf_vectorizer.pkl
│
├── backend/
│   ├── app.py
│   └── database.py
│
├── database/
│   ├── __init__.py
│   ├── db.py
│   └── predictions.db             # auto created
│
├── templates/
│   ├── index.html
│   ├── predict.html
│   └── about.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
│
└── main.py                        # old single-file script (optional)
```

## Setup and Run (Local System)

### 1) Create virtual environment

On Windows PowerShell:

```bash
python -m venv venv
.\venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Download NLTK stopwords (one-time)

```bash
python -c "import nltk; nltk.download('stopwords')"
```

### 4) Train machine learning model

```bash
python model/train_model.py
```

This will print:
- Accuracy
- Precision
- Recall

and save model files inside `model/artifacts/`.

### 5) Run Flask backend

```bash
python backend/app.py
```

Open browser:

http://127.0.0.1:5000

## API Endpoints

- `POST /predict`
  - input JSON:
    ```json
    { "news_text": "Your news text here", "country": "India" }
    ```
    - `country` is optional (default `Global`). Examples: `India`, `USA`, `UK`, …
  - output JSON:
    ```json
    {
      "prediction": "Fake",
      "confidence": 93.71,
      "country": "India",
      "message": "This looks like Fake news."
    }
    ```

### Optional backend assist (no UI change)

Set `OPENAI_API_KEY` and/or `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) in the environment to enable optional enhancement. If unset, the app uses only country-aware heuristics after the ML model. No provider name is shown in the UI.

- `POST /train`
  - retrains model and reloads saved files

- `POST /auth/signup`
  - input JSON: `username`, `email`, `password`
  - creates optional user account and stores session

- `POST /auth/login`
  - input JSON: `email`, `password`
  - logs in user and stores session

- `POST /auth/logout`
  - logs out current user

## Notes

- Input validation is added for empty text.
- Prediction confidence is shown in UI.
- Every prediction is stored in SQLite table `prediction_logs`.
- Optional auth users are stored in SQLite table `users` with hashed password.
- You can set custom session key with env var `FLASK_SECRET_KEY`.


