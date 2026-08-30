"""
Course 5: NLP Applications - Interactive Gradio Web Application
===============================================================
This web application connects directly to the student's trained model pipeline
exported from Notebook 04 ('models/sentiment_classifier.joblib') and provides
interactive interfaces for:
  1. Sentiment & Text Classification (Student Trained Model)
  2. Abstractive Text Summarization (Transformers Pipeline)
  3. Support Chatbot (Interactive Multi-Turn Dialogue)
  4. Batch Testing & Error Diagnostics Playground
  5. RAG-Powered Knowledge Assistant (Retrieval-Augmented Generation)
  6. Agent AI Playground (LLM + Native Tool-Calling Agent)

LLM backbone: Qwen2.5-3B-Instruct (local, no API key required). Chosen over
smaller seq2seq models like flan-t5 for native function-calling support,
stronger reasoning/instruction-following, and solid Arabic support. First
run downloads ~6GB of weights; ~8GB RAM/VRAM recommended for smooth use.

Usage:
  python app.py
"""

import os
import re
import time
import collections
import gradio as gr
import numpy as np
import pandas as pd

# Support both joblib and pickle for loading model artifacts
try:
    import joblib
except ImportError:
    import pickle as joblib

# -------------------------------------------------------------
# 1. Model Artifact Loading (Direct from Student Lab Export)
# -------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "sentiment_classifier.joblib")

def clean_text(text: str) -> str:
    """Same preprocessing function used in Notebook 04."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_student_model():
    """Loads the trained pipeline artifact exported from Notebook 04."""
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            return model, "✅ Loaded model artifact from `models/sentiment_classifier.joblib`"
        except Exception as e:
            return None, f"⚠️ Error loading model: {e}"
    else:
        return None, "⚠️ Model file not found at `models/sentiment_classifier.joblib`. Please run Task 6 in Notebook 04!"

# Lazy loaded transformers
# الكود الجديد (خفيف وسريع):
LLM_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
HF_MODELS = {
    "summarizer": None,
    "llm_model": None,
    "llm_tokenizer": None,
    "embedder": None,
}

def get_summarizer_pipeline():
    if HF_MODELS["summarizer"] is None:
        try:
            from transformers import pipeline
            HF_MODELS["summarizer"] = pipeline("summarization", model="t5-small")
        except Exception as e:
            print(f"Transformers summarizer note: {e}. Will use extractive summarizer fallback.")
            HF_MODELS["summarizer"] = "fallback"
    return HF_MODELS["summarizer"]

def get_llm():
    """Lazy-loads a local instruction-following chat LLM (Qwen2.5-3B-Instruct).
    No API key required — runs fully offline/local, which fits a classroom
    setting and Hugging Face Spaces deployment without secrets management.
    Chosen over flan-t5-base for native chat-style function calling, much
    stronger reasoning/instruction-following, and solid Arabic support.
    NOTE: first call downloads ~6GB of weights and needs ~8GB RAM/VRAM to
    run comfortably; on CPU-only machines generation will be slow (seconds
    per reply rather than sub-second).
    """
    if HF_MODELS["llm_model"] is None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_ID,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            HF_MODELS["llm_model"] = model
            HF_MODELS["llm_tokenizer"] = tokenizer
        except Exception as e:
            print(f"LLM load note: {e}. Will use canned fallback responses.")
            HF_MODELS["llm_model"] = "fallback"
            HF_MODELS["llm_tokenizer"] = "fallback"
    return HF_MODELS["llm_model"], HF_MODELS["llm_tokenizer"]

def get_embedder():
    """Lazy-loads a sentence-embedding model used for RAG retrieval."""
    if HF_MODELS["embedder"] is None:
        try:
            from sentence_transformers import SentenceTransformer
            HF_MODELS["embedder"] = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Embedder note: {e}. Will use TF-IDF fallback for retrieval.")
            HF_MODELS["embedder"] = "fallback"
    return HF_MODELS["embedder"]

def call_llm(messages, max_new_tokens: int = 200) -> str:
    """Single entry point for LLM chat generation, with a safe fallback.
    `messages` can be a plain string (treated as one user turn) or a full
    chat-format list of {"role": ..., "content": ...} dicts.
    """
    model, tokenizer = get_llm()
    if model == "fallback" or model is None:
        return "(LLM not available in this environment) I can't generate a free-form answer right now, but here is the most relevant information I found above."

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    try:
        import torch
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()
    except Exception as e:
        return f"(LLM generation error: {e})"


# -------------------------------------------------------------
# 2. Tab 1: Inference with Student's Trained Model
# -------------------------------------------------------------
def classify_text_fn(user_text):
    if not user_text or not user_text.strip():
        return "Please enter some text to analyze.", {}, "N/A"
    
    start_time = time.time()
    model, status = load_student_model()
    
    if model is None:
        return "Model Not Found", {}, f"❌ {status}"
    
    cleaned = clean_text(user_text)
    if not cleaned:
        cleaned = user_text.lower()
        
    try:
        # Predict using student's trained pipeline
        pred_label = model.predict([cleaned])[0]
        
        # Get probability distributions
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([cleaned])[0]
            classes = model.classes_
            prob_dict = {str(cls): float(round(p, 4)) for cls, p in zip(classes, probs)}
            top_prob = prob_dict.get(pred_label, 1.0)
        else:
            prob_dict = {pred_label: 1.0}
            top_prob = 1.0
            
        elapsed = round((time.time() - start_time) * 1000, 2)
        
        badge_color = "green" if pred_label == "Positive" else ("red" if pred_label == "Negative" else "orange")
        result_badge = f"**Predicted Class:** <span style='color:{badge_color}; font-size:1.2em;'>**{pred_label}**</span> ({top_prob:.1%} confidence) | Latency: `{elapsed} ms`"
        
        return pred_label, prob_dict, result_badge
    except Exception as e:
        return f"Prediction Error: {e}", {}, "Error"


# -------------------------------------------------------------
# 3. Tab 2: Text Summarization Logic
# -------------------------------------------------------------
def extractive_summary_fallback(text, num_sentences=2):
    sentences = [s.strip() for s in text.strip().split('.') if s.strip()]
    if len(sentences) <= num_sentences:
        return text
    words = re.findall(r'\w+', text.lower())
    stopwords = {"and", "the", "in", "to", "of", "a", "is", "that", "it", "on", "for", "as", "with", "these"}
    filtered = [w for w in words if w not in stopwords]
    freq = collections.Counter(filtered)
    scores = {}
    for i, s in enumerate(sentences):
        s_words = re.findall(r'\w+', s.lower())
        scores[i] = sum(freq[w] for w in s_words if w in freq) / max(1, len(s_words))
    top_indices = sorted(sorted(scores, key=scores.get, reverse=True)[:num_sentences])
    return ". ".join([sentences[i] for i in top_indices]) + "."

def summarize_text_fn(input_text, min_length, max_length, num_beams):
    if not input_text or len(input_text.strip().split()) < 10:
        return "⚠️ Please enter at least 10 words for meaningful summarization.", "0 words", "0%"
    
    orig_words = len(input_text.strip().split())
    pipe = get_summarizer_pipeline()
    
    if pipe != "fallback" and pipe is not None:
        try:
            res = pipe(
                input_text, 
                min_length=int(min_length), 
                max_length=int(max_length), 
                num_beams=int(num_beams), 
                do_sample=False
            )
            summary_text = res[0]['summary_text']
        except Exception as e:
            print(f"Summarizer inference error: {e}, falling back to extractive.")
            summary_text = extractive_summary_fallback(input_text, num_sentences=2)
    else:
        summary_text = extractive_summary_fallback(input_text, num_sentences=2)
        
    summary_words = len(summary_text.strip().split())
    compression = round((1 - (summary_words / max(1, orig_words))) * 100, 1)
    
    stats_text = f"**Original:** {orig_words} words | **Summary:** {summary_words} words"
    compression_badge = f"📉 **Compression Ratio:** {compression}% reduction"
    
    return summary_text, stats_text, compression_badge


# -------------------------------------------------------------
# 4. Tab 3: Chatbot Logic (rule-based / intent matching)
# -------------------------------------------------------------
CHAT_FAQS = {
    "greeting": {
        "patterns": ["hi", "hello", "hey", "good morning", "start"],
        "reply": "Hello! I am your AI Support Assistant. How can I help you with our application today?"
    },
    "pricing": {
        "patterns": ["price", "cost", "how much", "subscription", "plan", "free"],
        "reply": "Our Starter plan is $19/mo, Pro is $49/mo, and Enterprise is custom. We offer a 14-day free trial!"
    },
    "refund": {
        "patterns": ["refund", "money back", "cancel", "return"],
        "reply": "Refunds can be requested within 30 days of purchase under Account Settings > Billing > Request Refund."
    },
    "tech": {
        "patterns": ["crash", "error", "broken", "bug", "freeze", "fail"],
        "reply": "We apologize for the inconvenience! Please provide your account email and error details so our engineering team can investigate."
    },
    "nlp": {
        "patterns": ["nlp", "model", "summarize", "classify", "transformer", "gradio"],
        "reply": "This application demonstrates Text Classification, Abstractive Summarization, and Chatbots built with Hugging Face & Gradio!"
    }
}

def chatbot_response(user_message, history, persona):
    if not user_message:
        return ""
    
    msg_lower = user_message.lower()
    
    # 1. Match FAQ / Intent
    for intent, data in CHAT_FAQS.items():
        if any(p in msg_lower for p in data["patterns"]):
            reply = data["reply"]
            if persona == "Technical Specialist":
                reply = f"[Tech Desk]: {reply} Let us know if you need stack logs or API endpoints."
            elif persona == "Concise Assistant":
                reply = f"{reply}"
            return reply
            
    # 2. Fallback Response
    if persona == "Technical Specialist":
        return f"[Tech Desk]: Received query: '{user_message}'. Checking system diagnostics... Please specify if this is an API, database, or UI issue."
    else:
        return f"Thank you for your message! I noted your question regarding '{user_message}'. A customer specialist will follow up shortly, or you can ask about pricing, refunds, or technical support."


# -------------------------------------------------------------
# 5. Tab 4: Batch Tester Playground Logic
# -------------------------------------------------------------
SAMPLE_BATCH_DATA = [
    "Amazing platform, saved our team over 10 hours of manual work every week!",
    "Customer support answered in less than 2 minutes and solved my issue. Outstanding!",
    "Decent application, does basic tasks as expected but lacks advanced filters.",
    "Received my order invoice today via email as requested.",
    "Terrible experience. The app crashes continuously upon startup.",
    "I was billed twice and support has not responded for five days!",
    "Completely unusable on mobile browsers. Waste of money."
]

def run_batch_evaluation():
    model, status = load_student_model()
    if model is None:
        return pd.DataFrame([{"Status": "Model Not Found", "Message": "Please run Task 6 in Notebook 04 to export models/sentiment_classifier.joblib"}])
        
    results = []
    for text in SAMPLE_BATCH_DATA:
        cleaned = clean_text(text)
        try:
            pred = model.predict([cleaned])[0]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([cleaned])[0]
                conf = f"{max(probs):.1%}"
            else:
                conf = "100%"
        except Exception as e:
            pred, conf = f"Error: {e}", "N/A"
            
        results.append({
            "Input Feedback Sample": text,
            "Model Prediction": pred,
            "Confidence": conf
        })
    return pd.DataFrame(results)


# -------------------------------------------------------------
# 6. Tab 5: RAG Knowledge Assistant (Retrieval-Augmented Generation)
# -------------------------------------------------------------
# A small "knowledge base" of course + product documents. In a real system
# these chunks would come from your docs, PDFs, or a database — here they
# are hardcoded so the RAG pipeline is fully runnable out of the box.
RAG_KNOWLEDGE_BASE = [
    "Our Starter plan costs $19/month and includes text classification and summarization for up to 1,000 requests. The Pro plan costs $49/month with unlimited requests and priority support. Enterprise pricing is custom and includes on-premise deployment.",
    "Refund requests can be submitted within 30 days of purchase from Account Settings > Billing > Request Refund. Refunds are processed within 5-7 business days back to the original payment method.",
    "The Text Classification tab uses a Scikit-learn pipeline (TF-IDF + Logistic Regression or SVM) trained by students in Notebook 04, saved as models/sentiment_classifier.joblib.",
    "The Summarization tab uses the t5-small sequence-to-sequence transformer model for abstractive summarization, with an extractive TF-IDF fallback if the transformer model fails to load.",
    "Chatbot architectures covered in this course include Rule-Based matching, Intent & Entity classification, Retrieval-Augmented Generation (RAG), and Generative LLM agents.",
    "Technical issues such as app crashes, login errors, or bugs should be reported with the account email and a screenshot so the engineering team can investigate. Most issues are resolved within 48 hours.",
    "This application is built with Gradio and can be deployed for free on Hugging Face Spaces by uploading app.py and requirements.txt to a new Space with the Gradio SDK selected.",
    "Retrieval-Augmented Generation (RAG) works by embedding a knowledge base into vectors, retrieving the most semantically similar chunks to a user's question, and feeding them as context into an LLM prompt to generate a grounded answer.",
]

def _build_tfidf_index(corpus):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer().fit(corpus)
    matrix = vectorizer.transform(corpus)
    return vectorizer, matrix

_RAG_TFIDF_VECTORIZER, _RAG_TFIDF_MATRIX = _build_tfidf_index(RAG_KNOWLEDGE_BASE)
_RAG_EMBEDDINGS = None  # populated lazily if sentence-transformers is available

def retrieve_relevant_chunks(query: str, top_k: int = 3):
    """Retrieves the top_k most relevant knowledge base chunks for a query.
    Tries dense embeddings (sentence-transformers) first; falls back to TF-IDF
    cosine similarity if the embedding model isn't available.
    """
    global _RAG_EMBEDDINGS
    embedder = get_embedder()

    if embedder != "fallback" and embedder is not None:
        try:
            if _RAG_EMBEDDINGS is None:
                _RAG_EMBEDDINGS = embedder.encode(RAG_KNOWLEDGE_BASE, normalize_embeddings=True)
            query_vec = embedder.encode([query], normalize_embeddings=True)
            sims = np.dot(_RAG_EMBEDDINGS, query_vec[0])
            top_idx = np.argsort(sims)[::-1][:top_k]
            return [(RAG_KNOWLEDGE_BASE[i], float(sims[i])) for i in top_idx]
        except Exception as e:
            print(f"Embedder retrieval error: {e}, falling back to TF-IDF.")

    # TF-IDF fallback
    from sklearn.metrics.pairwise import cosine_similarity
    query_vec = _RAG_TFIDF_VECTORIZER.transform([query])
    sims = cosine_similarity(query_vec, _RAG_TFIDF_MATRIX)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [(RAG_KNOWLEDGE_BASE[i], float(sims[i])) for i in top_idx]

def rag_answer_fn(user_question):
    if not user_question or not user_question.strip():
        return "Please ask a question.", ""

    retrieved = retrieve_relevant_chunks(user_question, top_k=3)
    context = "\n".join([f"- {chunk}" for chunk, score in retrieved])

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the user's question using only "
                       "the provided context. If the answer isn't in the context, say you don't know. "
                       "Keep the answer short and direct.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {user_question}",
        },
    ]
    # الكود الجديد:
    answer = call_llm(messages, max_new_tokens=100)

    sources_md = "### 📚 Retrieved Sources\n" + "\n".join(
        [f"{i+1}. *(relevance {score:.2f})* {chunk}" for i, (chunk, score) in enumerate(retrieved)]
    )
    return answer, sources_md


# -------------------------------------------------------------
# 7. Tab 6: Agent AI Playground (LLM + Native Tool-Calling)
# -------------------------------------------------------------
# Qwen2.5-Instruct supports native function calling: we describe each tool
# as a JSON Schema, hand that schema to the model via the chat template's
# `tools=` argument, and let the MODEL ITSELF decide whether to call a tool,
# which one, and with what arguments — instead of a hand-written keyword
# router. The model replies with a <tool_call>{...}</tool_call> block when
# it wants to use a tool, which we parse, execute, and feed back so it can
# phrase the final answer.

AGENT_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "classify_sentiment",
            "description": "Classifies the sentiment/category of a piece of customer feedback or review text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The feedback or review text to classify."}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_text",
            "description": "Produces a short abstractive summary of a long piece of text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The long text to summarize."}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "knowledge_search",
            "description": "Searches the product/course knowledge base (pricing, refunds, deployment, "
                           "how the app works) to answer factual questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The factual question to search for."}
                },
                "required": ["query"],
            },
        },
    },
]

AGENT_TOOL_IMPLEMENTATIONS = {
    "classify_sentiment": lambda args: classify_text_fn(args.get("text", ""))[0],
    "summarize_text": lambda args: summarize_text_fn(args.get("text", ""), 10, 40, 2)[0],
    "knowledge_search": lambda args: rag_answer_fn(args.get("query", ""))[0],
}

AGENT_SYSTEM_PROMPT = (
    "You are a helpful assistant for an NLP application. You have access to tools for "
    "sentiment classification, text summarization, and knowledge-base search. Use a tool "
    "whenever the user's request matches one; otherwise answer directly. Keep replies short."
)

def parse_tool_call(raw_output: str):
    """Extracts a single Qwen-style <tool_call>{...}</tool_call> JSON block, if present."""
    import json as _json
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", raw_output, re.DOTALL)
    if not match:
        return None
    try:
        return _json.loads(match.group(1))
    except Exception:
        return None

def agent_run(user_message, history):
    if not user_message or not user_message.strip():
        return "", history or []

    history = history or []
    model, tokenizer = get_llm()
    trace_lines = []

    if model == "fallback" or model is None:
        bot_reply = call_llm(user_message)  # returns the fallback message
        history = history + [(user_message, bot_reply)]
        return "", history

    try:
        import torch
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tools=AGENT_TOOL_SCHEMAS, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        raw_reply = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()

        tool_call = parse_tool_call(raw_reply)

        if tool_call:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})
            trace_lines.append(f"🧭 **Model chose tool:** `{tool_name}`")

            if tool_name in AGENT_TOOL_IMPLEMENTATIONS:
                try:
                    tool_result = AGENT_TOOL_IMPLEMENTATIONS[tool_name](tool_args)
                except Exception as e:
                    tool_result = f"(tool error: {e})"
                trace_lines.append(f"🔧 Executed `{tool_name}` → result captured")

                # Feed the tool result back so the model phrases the final answer
                followup_messages = messages + [
                    {"role": "assistant", "content": raw_reply},
                    {"role": "tool", "content": str(tool_result)},
                ]
                final_answer = call_llm(followup_messages, max_new_tokens=150)
            else:
                final_answer = f"(Model requested unknown tool `{tool_name}`)"
        else:
            trace_lines.append("🧭 **Model chose:** answer directly (no tool)")
            final_answer = raw_reply

    except Exception as e:
        trace_lines.append(f"⚠️ Agent error: {e}")
        final_answer = "I hit an error trying to process that — please try rephrasing."

    bot_reply = f"{final_answer}\n\n<sub>{' | '.join(trace_lines)}</sub>"
    history = history + [(user_message, bot_reply)]
    return "", history


# -------------------------------------------------------------
# 8. Gradio UI Layout Construction (Cross-Version Compatible)
# -------------------------------------------------------------
with gr.Blocks(title="Course 5: NLP Applications Suite", theme=gr.themes.Soft()) as demo:
    
    with gr.Row():
        gr.Markdown("""
        # 🚀 Course 5: NLP Applications Interactive Web App
        ### End-to-End NLP: Classification • Summarization • Conversational AI • RAG • Agents
        *Directly connects to the student model artifact (`models/sentiment_classifier.joblib`)*
        """)

    with gr.Tabs():
        
        # ================= TAB 1: CLASSIFIER (STUDENT'S MODEL) =================
        with gr.TabItem("🏷️ 1. Sentiment Classifier (Student Model)"):
            gr.Markdown("### 🔍 Live Predictions from Student Trained Pipeline")
            gr.Markdown("Enter any feedback or review below to pass it through your trained `sentiment_classifier.joblib` pipeline.")
            
            with gr.Row():
                with gr.Column(scale=3):
                    text_input = gr.Textbox(
                        label="Input Text / Customer Review",
                        placeholder="e.g., The customer service was exceptionally helpful and resolved my billing issue immediately!",
                        lines=4
                    )
                    classify_btn = gr.Button("⚡ Run Model Prediction", variant="primary")
                    
                    gr.Examples(
                        examples=[
                            ["Amazing platform, saved our team over 10 hours of manual work every week!"],
                            ["The application runs okay, though occasionally requires a page refresh."],
                            ["Terrible experience. The app crashes continuously upon startup."],
                            ["Support answered within 2 minutes and solved my problem. Outstanding!"]
                        ],
                        inputs=text_input
                    )
                    
                with gr.Column(scale=2):
                    out_category = gr.Textbox(label="📌 Predicted Sentiment Class")
                    out_sentiment = gr.Markdown()
                    out_probs = gr.Label(label="📊 Class Probability Distribution", num_top_classes=3)
                    
            classify_btn.click(
                fn=classify_text_fn,
                inputs=text_input,
                outputs=[out_category, out_probs, out_sentiment]
            )

        # ================= TAB 2: SUMMARIZER =================
        with gr.TabItem("📝 2. Abstractive Text Summarizer"):
            gr.Markdown("### ✨ Transformer-Based Document Summarization")
            gr.Markdown("Condense long text into concise executive summaries using Sequence-to-Sequence models (`t5-small`).")
            
            with gr.Row():
                with gr.Column(scale=3):
                    summary_input = gr.Textbox(
                        label="Long Document / Review Text",
                        placeholder="Paste a long paragraph here...",
                        lines=7,
                        value="""I have been using your enterprise software for three months across our 50-person marketing team. While the automated report generation is remarkably fast and accurate, the permission management system is currently very confusing. Junior team members cannot view shared dashboards without manual admin approval, which slows down our weekly sprint review meetings significantly. We would appreciate a more flexible role-based access control."""
                    )
                    
                    with gr.Row():
                        slider_min = gr.Slider(minimum=10, maximum=40, value=15, step=5, label="Min Length")
                        slider_max = gr.Slider(minimum=20, maximum=100, value=45, step=5, label="Max Length")
                        slider_beams = gr.Slider(minimum=1, maximum=4, value=2, step=1, label="Beam Width")
                        
                    summarize_btn = gr.Button("✨ Generate Summary", variant="primary")
                    
                with gr.Column(scale=3):
                    summary_output = gr.Textbox(label="📄 Abstractive Summary Output", lines=5)
                    stats_output = gr.Markdown()
                    compression_output = gr.Markdown()
                    
            summarize_btn.click(
                fn=summarize_text_fn,
                inputs=[summary_input, slider_min, slider_max, slider_beams],
                outputs=[summary_output, stats_output, compression_output]
            )

        # ================= TAB 3: CHATBOT =================
        with gr.TabItem("💬 3. Customer Care AI Chatbot"):
            gr.Markdown("### 🤖 Conversational Support Assistant")
            gr.Markdown("Test multi-turn dialogue handling, intent routing, and persona adaptation.")
            
            with gr.Row():
                persona_selector = gr.Radio(
                    choices=["Friendly Assistant", "Technical Specialist", "Concise Assistant"],
                    value="Friendly Assistant",
                    label="Assistant Persona"
                )
            
            chatbot_component = gr.Chatbot(label="Conversation History", height=320)
            chat_msg = gr.Textbox(label="Your Message", placeholder="Ask about pricing, refunds, technical issues...")
            
            with gr.Row():
                chat_send_btn = gr.Button("💬 Send Message", variant="primary")
                chat_clear_btn = gr.Button("🗑️ Clear Chat")

            def user_turn(user_message, history, persona):
                if not user_message or not user_message.strip():
                    return "", history
                bot_reply = chatbot_response(user_message, history, persona)
                if history is None:
                    history = []
                history = history + [(user_message, bot_reply)]
                return "", history

            chat_send_btn.click(
                fn=user_turn,
                inputs=[chat_msg, chatbot_component, persona_selector],
                outputs=[chat_msg, chatbot_component]
            )
            chat_msg.submit(
                fn=user_turn,
                inputs=[chat_msg, chatbot_component, persona_selector],
                outputs=[chat_msg, chatbot_component]
            )
            chat_clear_btn.click(
                fn=lambda: None,
                inputs=None,
                outputs=chatbot_component
            )

        # ================= TAB 4: BATCH PLAYGROUND =================
        with gr.TabItem("🧪 4. Batch Evaluator & Diagnostics"):
            gr.Markdown("### 📊 Test Batch Inference with Student Model")
            gr.Markdown("Run a batch of test sentences through the loaded `sentiment_classifier.joblib` pipeline.")
            
            batch_btn = gr.Button("🚀 Run Batch Test on Model", variant="secondary")
            batch_table = gr.DataFrame(
                label="Batch Prediction Results",
                interactive=False
            )
            
            batch_btn.click(
                fn=run_batch_evaluation,
                inputs=[],
                outputs=batch_table
            )

        # ================= TAB 5: RAG KNOWLEDGE ASSISTANT =================
        with gr.TabItem("📚 5. RAG Knowledge Assistant"):
            gr.Markdown("### 🔎 Retrieval-Augmented Generation")
            gr.Markdown(
                "Ask a question about pricing, refunds, deployment, or how this app works. "
                "The system retrieves the most relevant knowledge base chunks (via embeddings) "
                "and feeds them to a local LLM (`Qwen2.5-3B-Instruct`) to generate a grounded answer."
            )

            with gr.Row():
                with gr.Column(scale=3):
                    rag_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., How much does the Pro plan cost?",
                        lines=2
                    )
                    rag_btn = gr.Button("🔎 Retrieve & Answer", variant="primary")
                    gr.Examples(
                        examples=[
                            ["How much does the Pro plan cost?"],
                            ["How do I get a refund?"],
                            ["How does RAG work in this app?"],
                            ["How can I deploy this app for free?"],
                        ],
                        inputs=rag_input
                    )
                with gr.Column(scale=3):
                    rag_answer_output = gr.Textbox(label="🤖 Generated Answer", lines=4)
                    rag_sources_output = gr.Markdown()

            rag_btn.click(
                fn=rag_answer_fn,
                inputs=rag_input,
                outputs=[rag_answer_output, rag_sources_output]
            )

        # ================= TAB 6: AGENT AI PLAYGROUND =================
        with gr.TabItem("🤖 6. Agent AI Playground"):
            gr.Markdown("### 🕹️ LLM Agent with Tool-Calling")
            gr.Markdown(
                "This agent reads your message, **decides which tool to use** "
                "(sentiment classifier, summarizer, or RAG knowledge search), executes it, "
                "then uses the LLM to phrase the final answer. The router decision is shown "
                "under each reply so you can trace the agent's reasoning."
            )

            agent_chatbot = gr.Chatbot(label="Agent Conversation", height=360)
            agent_msg = gr.Textbox(
                label="Your Message",
                placeholder="e.g., 'Summarize this: ...' / 'Classify this feedback: ...' / 'What is the refund policy?'"
            )
            with gr.Row():
                agent_send_btn = gr.Button("🚀 Run Agent", variant="primary")
                agent_clear_btn = gr.Button("🗑️ Clear")

            agent_send_btn.click(
                fn=agent_run,
                inputs=[agent_msg, agent_chatbot],
                outputs=[agent_msg, agent_chatbot]
            )
            agent_msg.submit(
                fn=agent_run,
                inputs=[agent_msg, agent_chatbot],
                outputs=[agent_msg, agent_chatbot]
            )
            agent_clear_btn.click(
                fn=lambda: None,
                inputs=None,
                outputs=agent_chatbot
            )

    gr.Markdown("--- \n *Course 5: NLP Applications • ITI Applied AI Program*")

# -------------------------------------------------------------
# 9. Application Entrypoint
# -------------------------------------------------------------
# الكود الجديد:
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)