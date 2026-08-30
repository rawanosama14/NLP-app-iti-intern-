
<p align="center">
  <img src="https://img.shields.io/badge/Course_5-NLP_Applications-purple?style=for-the-badge&logo=huggingface&logoColor=white" alt="NLP Applications" />
</p>

# 🚀 Course 5: NLP Applications & Capstone Project

This repository contains instructional notebooks, hands-on lab projects, and an interactive Gradio web application covering end-to-end Natural Language Processing workflows[cite: 1].

---

## 📁 Repository Structure

```text
├── 01_NLP_Workflows_and_Text_Classification.ipynb    # Lecture 1: End-to-End NLP & Classification[cite: 1]
├── 02_Text_Summarization_and_Text_Generation.ipynb    # Lecture 2: Summarization & Generative AI[cite: 1]
├── 03_Chatbot_Basics_and_Conversational_AI.ipynb      # Lecture 3: Conversational AI & UIs[cite: 1]
├── 04_Lab_Student_Project_NLP_Application.ipynb       # Student Capstone Lab Project[cite: 1]
├── app2.py                                            # Multi-tab Gradio Web Deployment App[cite: 1]
├── requirements.txt                                   # Python dependencies[cite: 1]
└── README.md                                          # Course guide & instructions[cite: 1]

```

---

## 🎯 Curriculum Overview & Key Features

* **Text Classification & Sentiment Analysis:** Building classical pipelines using `TF-IDF` alongside `Logistic Regression` and `Naive Bayes`, complete with evaluation metrics and model serialization (`joblib`).
* **Abstractive Text Summarization:** Implementing sequence-to-sequence models (`t5-small`) with custom decoding parameters (Beam Search, Min/Max lengths).
* **Conversational AI & Chatbots:** Multi-turn dialogue management, intent routing, persona switching, and rule-based fallback handlers.
* **RAG & Agent Workflows:** Retrieval-Augmented Generation using semantic vector embeddings and tool-calling agent capabilities.

---

## 🌐 Running the Gradio Web Application

1. **Install dependencies:**
```bash
pip install -r requirements.txt

```


2. **Launch the application locally:**
```bash
python app2.py

```


Open `http://127.0.0.1:7861` in your browser to interact with the multi-tab AI suite.

---

