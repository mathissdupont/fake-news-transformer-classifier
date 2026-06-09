"""Chat-style Streamlit UI for Turkish fake news detection."""
from pathlib import Path

import joblib
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


st.set_page_config(
    page_title="Fake News Detector",
    layout="centered",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    .stApp {
        background: #f5f7fa;
        color: #111827;
    }

    .block-container {
        max-width: 920px;
        padding-top: 32px;
        padding-bottom: 36px;
    }

    h1, h2, h3, p, span, label, div {
        letter-spacing: 0;
    }

    .app-title {
        background: #ffffff;
        border: 1px solid #d7dde7;
        border-radius: 8px;
        padding: 20px 22px;
        margin-bottom: 16px;
    }

    .app-title h1 {
        color: #111827;
        font-size: 2rem;
        line-height: 1.12;
        margin: 0 0 8px;
    }

    .app-title p {
        color: #4b5563;
        font-size: 1rem;
        margin: 0;
    }

    .model-panel {
        background: #ffffff;
        border: 1px solid #d7dde7;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }

    .model-panel strong {
        color: #111827;
    }

    .model-panel p {
        color: #4b5563;
        margin: 4px 0 0;
        font-size: 0.92rem;
    }

    div[data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #d7dde7;
        border-radius: 8px;
        color: #111827;
        box-shadow: 0 1px 2px rgba(17, 24, 39, 0.04);
    }

    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] span,
    div[data-testid="stChatMessage"] div {
        color: #111827;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d7dde7;
        border-radius: 8px;
        padding: 12px 14px;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] div {
        color: #111827;
    }

    .result-box {
        border: 1px solid #d7dde7;
        border-radius: 8px;
        padding: 18px;
        background: #ffffff;
        margin: 8px 0 16px;
    }

    .result-box.fake {
        border-color: #f1a59c;
        background: #fff4f2;
    }

    .result-box.real {
        border-color: #8bd19b;
        background: #edf8f0;
    }

    .result-label {
        color: #111827;
        font-size: 1.28rem;
        font-weight: 750;
        margin-bottom: 6px;
    }

    .result-copy {
        color: #374151;
        font-size: 0.98rem;
        line-height: 1.55;
    }

    .small-note {
        color: #4b5563;
        font-size: 0.88rem;
        margin-top: 8px;
    }

    .missing-model {
        color: #7c2d12;
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 8px;
        padding: 10px 12px;
        margin-top: 10px;
        font-size: 0.92rem;
    }

    .stChatInput {
        background: #f5f7fa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


MODEL_CONFIGS = {
    "tfidf": {
        "label": "TF-IDF + Logistic Regression",
        "description": "Kelime ve n-gram örüntülerine göre hızlı baseline model.",
        "model_path": MODELS_DIR / "tfidf_model.pkl",
        "vectorizer_path": MODELS_DIR / "tfidf_vectorizer.pkl",
        "kind": "tfidf",
    },
    "embedding_lr": {
        "label": "Embedding + Logistic Regression",
        "description": "Cümle embeddingleriyle anlamsal benzerlik tarafını daha çok yakalar.",
        "model_path": MODELS_DIR / "embedding_lr.pkl",
        "kind": "embedding",
    },
    "embedding_svm": {
        "label": "Embedding + SVM",
        "description": "Embedding uzayında lineer SVM ile karar verir.",
        "model_path": MODELS_DIR / "embedding_svm.pkl",
        "kind": "embedding",
    },
}


@st.cache_resource(show_spinner=False)
def load_joblib(path):
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_embedding_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def available_model_keys():
    keys = []
    for key, config in MODEL_CONFIGS.items():
        model_exists = config["model_path"].exists()
        vectorizer_exists = config.get("vectorizer_path", Path()).exists()
        if config["kind"] == "tfidf" and model_exists and vectorizer_exists:
            keys.append(key)
        elif config["kind"] == "embedding" and model_exists:
            keys.append(key)
    return keys


def missing_model_labels():
    available = set(available_model_keys())
    return [
        config["label"]
        for key, config in MODEL_CONFIGS.items()
        if key not in available
    ]


def get_probability(model, probabilities, label):
    classes = list(getattr(model, "classes_", []))
    if label not in classes:
        return 0.0
    return float(probabilities[classes.index(label)])


def features_for_text(model_key, text):
    config = MODEL_CONFIGS[model_key]
    if config["kind"] == "tfidf":
        vectorizer = load_joblib(config["vectorizer_path"])
        return vectorizer.transform([text])

    encoder = load_embedding_encoder()
    return encoder.encode([text], convert_to_numpy=True)


def analyze_with_model(model_key, text):
    config = MODEL_CONFIGS[model_key]
    model = load_joblib(config["model_path"])
    if model is None:
        return None

    features = features_for_text(model_key, text)
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    fake_probability = get_probability(model, probabilities, "fake")
    real_probability = get_probability(model, probabilities, "real")

    return {
        "model_key": model_key,
        "model_label": config["label"],
        "prediction": prediction,
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "confidence": max(fake_probability, real_probability),
    }


def percent(value):
    return f"{value * 100:.1f}%"


def build_explanation(result, text):
    prediction = result["prediction"]
    confidence = result["confidence"]
    text_length = len(text.split())
    model_label = result.get("model_label", "Seçilen model")

    if prediction == "fake":
        headline = "Bu haber şüpheli görünüyor."
        stance = (
            f"{model_label}, metindeki örüntüleri fake haber sınıfına daha yakın buldu."
        )
    else:
        headline = "Bu haber doğru olma tarafına daha yakın görünüyor."
        stance = (
            f"{model_label}, metindeki örüntüleri real haber sınıfına daha yakın buldu."
        )

    if confidence >= 0.80:
        certainty = "Tahmin güçlü."
    elif confidence >= 0.65:
        certainty = "Tahmin orta seviyenin üstünde."
    else:
        certainty = "Tahmin kararsız; bu metin için kesin konuşmamak gerekir."

    length_note = (
        "Metin kısa olduğu için modelin bağlamı sınırlı kalmış olabilir."
        if text_length < 25
        else "Metin uzunluğu modelin bağlam yakalaması için yeterli görünüyor."
    )

    return headline, f"{stance} {certainty} {length_note}"


def render_single_result(result, text):
    headline, explanation = build_explanation(result, text)
    style = "fake" if result["prediction"] == "fake" else "real"

    st.markdown(
        f"""
        <div class="result-box {style}">
            <div class="result-label">{headline}</div>
            <div class="result-copy">{explanation}</div>
            <div class="small-note">Seçilen model: {result.get("model_label", "TF-IDF + Logistic Regression")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    left.metric("Doğru olma olasılığı", percent(result["real_probability"]))
    right.metric("Doğru olmama olasılığı", percent(result["fake_probability"]))

    st.progress(result["real_probability"], text="Doğru olma skoru")
    st.progress(result["fake_probability"], text="Doğru olmama skoru")
    st.caption(
        "Not: Bu sonuç doğrulama kararı değil, modelin istatistiksel tahminidir. "
        "Haber yine de kaynak kontrolüyle değerlendirilmelidir."
    )


def render_comparison(results, text):
    st.write("Modellerin aynı haber için verdiği sonuçlar:")
    for result in results:
        render_single_result(result, text)


def render_assistant_content(content):
    if content.get("mode") == "compare":
        render_comparison(content["results"], content["text"])
    else:
        render_single_result(content["result"], content["text"])


st.markdown(
    """
    <div class="app-title">
        <h1>Fake News Detection AI</h1>
        <p>Haber içeriğini yapıştır, seçtiğin model doğru olma ve olmama olasılığını açıklasın.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

available_keys = available_model_keys()
if not available_keys:
    st.error("Hiç model dosyası bulunamadı. Önce en az TF-IDF modelini eğitmek gerekiyor.")
    st.code("python src/train_tfidf.py", language="bash")
    st.stop()

with st.container():
    st.markdown(
        """
        <div class="model-panel">
            <strong>Model seçimi</strong>
            <p>Farklı modelleri seçerek aynı haber için olasılıkların nasıl değiştiğini görebilirsin.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_label = st.selectbox(
        "Kullanılacak model",
        options=[MODEL_CONFIGS[key]["label"] for key in available_keys],
        label_visibility="collapsed",
    )
    selected_model_key = next(
        key for key in available_keys if MODEL_CONFIGS[key]["label"] == selected_label
    )
    compare_models = st.toggle(
        "Mevcut tüm modellerle karşılaştır",
        value=False,
        disabled=len(available_keys) < 2,
    )

missing = missing_model_labels()
if missing:
    st.markdown(
        f"""
        <div class="missing-model">
            Şu an aktif olmayan modeller: {", ".join(missing)}.
            Embedding modellerini görmek için <code>python src/train_embeddings.py</code>
            çalıştırılmalı.
        </div>
        """,
        unsafe_allow_html=True,
    )

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Merhaba. Haber metnini gönder; seçtiğin modele göre doğru olma "
                "ve doğru olmama olasılığını açıklayayım."
            ),
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], dict):
            render_assistant_content(message["content"])
        else:
            st.write(message["content"])

prompt = st.chat_input("Haber içeriğini buraya yapıştır...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    if compare_models and len(available_keys) > 1:
        with st.spinner("Modeller karşılaştırılıyor..."):
            results = [analyze_with_model(key, prompt) for key in available_keys]
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": {"mode": "compare", "results": results, "text": prompt},
            }
        )
    else:
        with st.spinner("Haber analiz ediliyor..."):
            result = analyze_with_model(selected_model_key, prompt)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": {"mode": "single", "result": result, "text": prompt},
            }
        )

    st.rerun()
