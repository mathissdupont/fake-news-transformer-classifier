"""Chat-style Streamlit UI for Turkish fake news detection."""
import csv
from html import escape
from pathlib import Path

import joblib
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


st.set_page_config(
    page_title="Haber Doğruluk Asistanı",
    layout="centered",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    .stApp {
        background: #f3f6f8;
        color: #111827;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: #f3f6f8 !important;
        color: #111827 !important;
    }

    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] {
        color: #111827 !important;
    }

    * {
        color-scheme: light;
    }

    .block-container {
        max-width: 980px;
        padding-top: 26px;
        padding-bottom: 36px;
    }

    h1, h2, h3, p, span, label, div {
        letter-spacing: 0;
    }

    p, span, label, div, button, input, textarea {
        color: #111827;
    }

    [data-baseweb="select"] > div,
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [data-baseweb="option"] {
        background: #ffffff !important;
        color: #111827 !important;
        border-color: #cfd6e2 !important;
    }

    [data-baseweb="select"] span,
    [data-baseweb="popover"] span,
    [data-baseweb="menu"] div,
    [data-baseweb="option"] div {
        color: #111827 !important;
    }

    [data-baseweb="option"]:hover {
        background: #eef6f7 !important;
    }

    div[role="listbox"],
    ul[role="listbox"],
    li[role="option"],
    div[role="option"] {
        background: #ffffff !important;
        color: #111827 !important;
    }

    li[role="option"] *,
    div[role="option"] *,
    div[role="listbox"] *,
    ul[role="listbox"] * {
        background: transparent !important;
        color: #111827 !important;
    }

    li[role="option"]:hover,
    div[role="option"]:hover,
    li[aria-selected="true"],
    div[aria-selected="true"] {
        background: #eef6f7 !important;
        color: #111827 !important;
    }

    div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li {
        background: #ffffff !important;
        color: #111827 !important;
    }

    [data-testid="stCaptionContainer"],
    [data-testid="stMarkdownContainer"] {
        color: #111827 !important;
    }

    .app-title {
        background: #ffffff;
        border: 1px solid #d7dde7;
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 12px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 14px;
    }

    .brand-mark {
        width: 34px;
        height: 34px;
        border-radius: 8px;
        background: #146c75;
        color: #ffffff;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
    }

    .brand-name {
        color: #374151;
        font-weight: 700;
        font-size: 0.95rem;
    }

    .app-title h1 {
        color: #111827;
        font-size: 1.45rem;
        line-height: 1.12;
        margin: 0 0 6px;
    }

    .app-title p {
        color: #4b5563;
        font-size: 1rem;
        margin: 0;
        max-width: 760px;
    }

    .chat-layout-note {
        color: #4b5563;
        font-size: 0.92rem;
        margin: 8px 0 14px;
    }

    .top-controls {
        background: #ffffff;
        border: 1px solid #d7dde7;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 14px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
    }

    .control-label {
        color: #111827;
        font-weight: 800;
        font-size: 0.95rem;
        margin-bottom: 2px;
    }

    .control-copy {
        color: #4b5563;
        font-size: 0.86rem;
        margin-bottom: 8px;
    }

    .empty-chat {
        background: #ffffff;
        border: 1px solid #d7dde7;
        border-radius: 8px;
        padding: 22px;
        margin: 16px 0;
        text-align: center;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
    }

    .empty-chat h3 {
        color: #111827;
        margin: 0 0 8px;
        font-size: 1.25rem;
    }

    .empty-chat p {
        color: #4b5563;
        margin: 0;
    }

    .model-panel {
        background: #ffffff;
        border: 1px solid #d7dde7;
        border-radius: 8px;
        padding: 14px 16px 10px;
        margin-bottom: 16px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
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
        box-shadow: 0 6px 18px rgba(17, 24, 39, 0.05);
    }

    div[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"],
    div[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
        background: #eef6f7 !important;
        color: #146c75 !important;
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

    div[data-testid="stAlert"] {
        background: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #d7dde7 !important;
        border-radius: 8px !important;
    }

    div[data-testid="stAlert"] div,
    div[data-testid="stAlert"] p {
        color: #111827 !important;
    }

    div[data-testid="stProgress"] div {
        color: #111827 !important;
    }

    div[role="progressbar"] {
        background-color: #146c75 !important;
    }

    .result-box {
        border: 1px solid #d7dde7;
        border-radius: 8px;
        padding: 20px;
        background: #ffffff;
        margin: 8px 0 16px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
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
        font-size: 1.38rem;
        font-weight: 750;
        margin-bottom: 6px;
    }

    .verdict-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 5px 10px;
        margin-bottom: 10px;
        font-size: 0.82rem;
        font-weight: 750;
        color: #0f5132;
        background: #dff5e6;
        border: 1px solid #a7dfb6;
    }

    .result-box.fake .verdict-pill {
        color: #842029;
        background: #ffe3df;
        border-color: #f3b6af;
    }

    .result-copy {
        color: #374151;
        font-size: 0.98rem;
        line-height: 1.55;
    }

    .answer-topline {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 12px;
    }

    .probability-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 10px;
        margin: 12px 0;
    }

    .probability-card {
        background: #ffffff;
        border: 1px solid #d7dde7;
        border-radius: 8px;
        padding: 12px 14px;
    }

    .probability-card span {
        display: block;
        color: #6b7280;
        font-size: 0.82rem;
        margin-bottom: 4px;
    }

    .probability-card strong {
        color: #111827;
        font-size: 1.3rem;
    }

    .mini-bar {
        height: 8px;
        background: #e5e7eb;
        border-radius: 999px;
        overflow: hidden;
        margin-top: 8px;
    }

    .mini-bar span {
        height: 100%;
        display: block;
        background: #146c75;
        border-radius: 999px;
    }

    .probability-card.false .mini-bar span {
        background: #b42318;
    }

    .trust-note {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 8px;
        color: #7c2d12;
        padding: 11px 12px;
        margin-top: 12px;
        line-height: 1.45;
        font-size: 0.92rem;
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
        background: #f3f6f8 !important;
    }

    div[data-testid="stChatInput"] {
        background: #f3f6f8 !important;
    }

    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottom"],
    [data-testid="stChatInputSubmitButton"],
    section[data-testid="stBottomBlockContainer"],
    div:has(> div[data-testid="stChatInput"]),
    div:has(div[data-testid="stChatInput"]) {
        background: #f3f6f8 !important;
        color: #111827 !important;
    }

    div[style*="position: fixed"]:has(textarea),
    div[style*="position: sticky"]:has(textarea),
    div[style*="bottom: 0"]:has(textarea),
    div[style*="bottom:0"]:has(textarea) {
        background: #f3f6f8 !important;
        color: #111827 !important;
    }

    div[data-testid="stChatInput"] textarea {
        background: #ffffff !important;
        color: #111827 !important;
        caret-color: #111827 !important;
        border: 1px solid #cfd6e2 !important;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08) !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: #6b7280 !important;
        opacity: 1 !important;
    }

    div[data-testid="stChatInput"] button {
        color: #146c75 !important;
        background: #ffffff !important;
    }

    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] form,
    div[data-testid="stChatInput"] label {
        background: #f3f6f8 !important;
        color: #111827 !important;
    }

    div[data-testid="stButton"] button {
        background: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #cfd6e2 !important;
    }

    div[data-testid="stButton"] button:hover {
        border-color: #146c75 !important;
        color: #146c75 !important;
    }

    .reason-panel {
        background: #ffffff;
        border: 1px solid #d7dde7;
        border-radius: 8px;
        padding: 14px 16px;
        margin: 14px 0 8px;
    }

    .reason-panel h4 {
        color: #111827;
        font-size: 1rem;
        margin: 0 0 10px;
    }

    .reason-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 8px;
        margin-top: 10px;
    }

    .reason-chip {
        background: #f8fafc;
        border: 1px solid #d7dde7;
        border-radius: 8px;
        padding: 9px 10px;
        color: #111827;
        font-size: 0.92rem;
        line-height: 1.35;
    }

    .reason-chip.predicted {
        border-color: #9bd0d4;
        background: #edfafa;
    }

    .reason-chip.counter {
        border-color: #e5d2a0;
        background: #fffaf0;
    }

    .reason-token {
        color: #111827;
        font-weight: 750;
        margin-bottom: 4px;
    }

    .reason-meta {
        color: #4b5563;
        font-size: 0.82rem;
        margin-bottom: 7px;
    }

    .reason-bar {
        height: 7px;
        border-radius: 999px;
        background: #e5e7eb;
        overflow: hidden;
    }

    .reason-bar span {
        display: block;
        height: 100%;
        border-radius: 999px;
        background: #146c75;
    }

    .reason-chip.counter .reason-bar span {
        background: #b45309;
    }

    .reason-copy strong {
        color: #146c75;
    }

    .reason-copy {
        color: #374151;
        font-size: 0.94rem;
        line-height: 1.55;
        margin: 0;
    }

    .reason-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 8px;
        margin: 10px 0 12px;
    }

    .reason-stat {
        background: #f8fafc;
        border: 1px solid #d7dde7;
        border-radius: 8px;
        padding: 9px 10px;
    }

    .reason-stat span {
        display: block;
        color: #6b7280;
        font-size: 0.78rem;
        margin-bottom: 2px;
    }

    .reason-stat strong {
        color: #111827;
        font-size: 0.98rem;
    }

    .reason-section-title {
        color: #111827;
        font-size: 0.93rem;
        font-weight: 750;
        margin: 12px 0 4px;
    }

    .reason-warning {
        background: #f8fafc;
        border-left: 4px solid #94a3b8;
        color: #374151;
        padding: 9px 11px;
        border-radius: 6px;
        margin-top: 12px;
        font-size: 0.9rem;
        line-height: 1.45;
    }

    .quality-panel {
        background: #ffffff;
        border: 1px solid #d7dde7;
        border-radius: 8px;
        padding: 12px 14px;
        margin: 10px 0 14px;
    }

    .quality-panel strong {
        color: #111827;
    }

    .quality-panel ul {
        margin: 8px 0 0 18px;
        padding: 0;
        color: #374151;
    }

    .quality-panel li {
        margin: 4px 0;
    }

    .model-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        font-size: 0.92rem;
    }

    .model-table th,
    .model-table td {
        border-bottom: 1px solid #e5e7eb;
        padding: 8px 6px;
        text-align: left;
        color: #111827;
    }

    .model-table th {
        color: #4b5563;
        font-weight: 750;
    }

    .recommendation {
        background: #eef8f9;
        border: 1px solid #b9dfe3;
        border-radius: 8px;
        padding: 12px 14px;
        color: #164e53;
        margin-top: 12px;
        line-height: 1.5;
    }

    .analyzer-shell {
        background: #ffffff;
        border: 1px solid #d7dde7;
        border-radius: 8px;
        padding: 18px;
        margin: 16px 0;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
    }

    .section-kicker {
        color: #146c75;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .section-title {
        color: #111827;
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .section-copy {
        color: #4b5563;
        font-size: 0.92rem;
        margin-bottom: 12px;
    }

    .feature-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 8px;
        margin-top: 12px;
    }

    .feature-pill {
        background: #f8fafc;
        border: 1px solid #e1e7ef;
        border-radius: 8px;
        padding: 9px 10px;
        color: #374151;
        font-size: 0.88rem;
    }

    div[data-testid="stTextArea"] textarea {
        background: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #cfd6e2 !important;
        border-radius: 8px !important;
        min-height: 190px !important;
        box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.04) !important;
    }

    div[data-testid="stTextArea"] textarea::placeholder {
        color: #6b7280 !important;
        opacity: 1 !important;
    }

    div[data-testid="stButton"] button {
        border-radius: 8px;
        min-height: 44px;
        font-weight: 800;
    }

    div[role="radiogroup"] {
        background: #f8fafc;
        border: 1px solid #d7dde7;
        border-radius: 8px;
        padding: 8px 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


MODEL_CONFIGS = {
    "tfidf": {
        "label": "TF-IDF + Logistic Regression",
        "short_label": "Kelime tabanlı model",
        "description": "Başlık ve metindeki kelime örüntülerini dikkate alan hızlı model.",
        "model_path": MODELS_DIR / "tfidf_model.pkl",
        "vectorizer_path": MODELS_DIR / "tfidf_vectorizer.pkl",
        "kind": "tfidf",
    },
    "embedding_lr": {
        "label": "Embedding + Logistic Regression",
        "short_label": "Anlamsal model",
        "description": "Metnin genel anlamını ve cümle benzerliğini dikkate alan model.",
        "model_path": MODELS_DIR / "embedding_lr.pkl",
        "kind": "embedding",
    },
    "embedding_svm": {
        "label": "Embedding + SVM",
        "short_label": "Anlamsal karşılaştırma modeli",
        "description": "Metni anlam uzayında konumlandırarak karar veren alternatif model.",
        "model_path": MODELS_DIR / "embedding_svm.pkl",
        "kind": "embedding",
    },
}

STOPWORDS = {
    "acaba",
    "ama",
    "bir",
    "bu",
    "da",
    "de",
    "diye",
    "en",
    "gibi",
    "in",
    "ile",
    "ise",
    "ki",
    "mi",
    "mu",
    "mü",
    "nasıl",
    "ne",
    "o",
    "olan",
    "oldu",
    "olarak",
    "şu",
    "ve",
    "veya",
    "ya",
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
        model_display_name(config)
        for key, config in MODEL_CONFIGS.items()
        if key not in available
    ]


def get_probability(model, probabilities, label):
    classes = list(getattr(model, "classes_", []))
    if label not in classes:
        return 0.0
    return float(probabilities[classes.index(label)])


def display_label(label):
    return {"real": "doğru", "fake": "yanlış"}.get(label, label)


def model_display_name(config):
    return f'{config["label"]} ({config["short_label"]})'


def verdict_title(label):
    if label == "real":
        return "Bu haber doğru olma eğiliminde"
    return "Bu haber yanlış olma eğiliminde"


def probability_label(label):
    if label == "real":
        return "Doğru yönde sinyal"
    return "Yanlış yönde sinyal"


def features_for_text(model_key, text):
    config = MODEL_CONFIGS[model_key]
    if config["kind"] == "tfidf":
        vectorizer = load_joblib(config["vectorizer_path"])
        return vectorizer.transform([text])

    encoder = load_embedding_encoder()
    return encoder.encode([text], convert_to_numpy=True)


def is_explainable_token(token):
    parts = [part.strip().lower() for part in token.split() if part.strip()]
    if not parts:
        return False
    if all(part in STOPWORDS or len(part) < 3 for part in parts):
        return False
    return True


def top_tfidf_reasons(model, vectorizer, features, prediction, limit=10):
    if not hasattr(model, "coef_"):
        return []

    class_names = list(getattr(model, "classes_", []))
    if len(class_names) != 2:
        return []

    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]
    contributions = features.multiply(coef).tocoo()
    items = []

    for _, feature_index, score in zip(contributions.row, contributions.col, contributions.data):
        token = feature_names[feature_index]
        if not is_explainable_token(token):
            continue
        target_class = class_names[1] if score >= 0 else class_names[0]
        items.append(
            {
                "token": token,
                "score": float(score),
                "strength": abs(float(score)),
                "target_class": target_class,
                "aligns_with_prediction": target_class == prediction,
            }
        )

    items.sort(key=lambda item: item["strength"], reverse=True)
    return items[:limit]


def tfidf_reference_analysis(text):
    config = MODEL_CONFIGS["tfidf"]
    if not config["model_path"].exists() or not config["vectorizer_path"].exists():
        return None

    model = load_joblib(config["model_path"])
    vectorizer = load_joblib(config["vectorizer_path"])
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    return {
        "prediction": prediction,
        "fake_probability": get_probability(model, probabilities, "fake"),
        "real_probability": get_probability(model, probabilities, "real"),
        "reasons": top_tfidf_reasons(model, vectorizer, features, prediction),
    }


def model_reason_summary(model_key, result, text):
    text_length = len(text.split())
    confidence = result["confidence"]

    if model_key == "tfidf":
        return (
            "Bu model metni kelime ve kısa ifade ağırlıklarına ayırır. Aşağıdaki "
            "sinyaller, kararın doğru ya da yanlış tarafına kaymasında en etkili "
            "görünen parçalardır."
        )

    if confidence < 0.65:
        certainty = "Model iki sınıf arasında belirgin ayrım kuramadı."
    elif confidence < 0.80:
        certainty = "Model bir sınıfa daha yakın buldu ama sonuç tamamen net değil."
    else:
        certainty = "Model metni seçilen sınıfa belirgin biçimde yakın buldu."

    length_note = (
        "Metin kısa olduğu için anlamsal bağlam sınırlı kalmış olabilir."
        if text_length < 25
        else "Metin yeterince uzun olduğu için cümle düzeyinde bağlam kullanılabildi."
    )

    return (
        "Anlamsal model metni tek tek kelimeler yerine cümle düzeyinde bir anlam "
        "vektörüne çevirir. Bu yüzden karar kelime listesiyle birebir açıklanamaz; "
        f"aşağıdaki özet, olasılık farkı ve kelime tabanlı referans sinyalleriyle okunmalıdır. "
        f"{certainty} {length_note}"
    )


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
    reasons = []

    if config["kind"] == "tfidf":
        vectorizer = load_joblib(config["vectorizer_path"])
        reasons = top_tfidf_reasons(model, vectorizer, features, prediction)

    lexical_reference = None
    if config["kind"] == "embedding":
        lexical_reference = tfidf_reference_analysis(text)

    return {
        "model_key": model_key,
        "model_label": config["label"],
        "model_short_label": config["short_label"],
        "model_display_name": model_display_name(config),
        "prediction": prediction,
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "confidence": max(fake_probability, real_probability),
        "reasons": reasons,
        "reason_summary": model_reason_summary(model_key, {
            "confidence": max(fake_probability, real_probability)
        }, text),
        "lexical_reference": lexical_reference,
    }


@st.cache_data(show_spinner=False, max_entries=64)
def cached_analyze_with_model(model_key, text):
    return analyze_with_model(model_key, text)


def percent(value):
    return f"{value * 100:.1f}%"


def text_quality_notes(text):
    word_count = len(text.split())
    notes = []
    if word_count < 20:
        notes.append("Metin kısa; başlıkla birlikte haber açıklaması da eklenirse tahmin daha sağlıklı olur.")
    if "http" in text.lower() or "www." in text.lower():
        notes.append("Metinde bağlantı var; model bağlantının kendisini doğrulamaz, yalnızca yazı içeriğini analiz eder.")
    if text.isupper() and len(text) > 20:
        notes.append("Metin tamamen büyük harfli; bu durum bazı kelime örüntülerini etkileyebilir.")
    if not any(char in text for char in ".!?"):
        notes.append("Metin tek parça görünüyor; birkaç cümlelik bağlam daha iyi sonuç verebilir.")
    if not notes:
        notes.append("Metin uzunluğu ve biçimi analiz için uygun görünüyor.")
    return word_count, notes


def consensus_from_results(results):
    real_probability = sum(result["real_probability"] for result in results) / len(results)
    fake_probability = sum(result["fake_probability"] for result in results) / len(results)
    prediction = "real" if real_probability >= fake_probability else "fake"
    agreement = sum(1 for result in results if result["prediction"] == prediction)

    return {
        "model_key": "consensus",
        "model_label": "Konsensüs Analizi",
        "model_display_name": "Konsensüs Analizi (Tüm modellerin ortalaması)",
        "prediction": prediction,
        "real_probability": real_probability,
        "fake_probability": fake_probability,
        "confidence": max(real_probability, fake_probability),
        "agreement": agreement,
        "total_models": len(results),
        "reasons": [],
        "reason_summary": (
            "Konsensüs analizi, mevcut modellerin doğru ve yanlış olasılıklarını "
            "ortalayarak daha dengeli bir karar özeti üretir."
        ),
    }


@st.cache_data(show_spinner=False)
def load_result_metrics():
    path = RESULTS_DIR / "final_results_table.csv"
    if not path.exists():
        return {}

    metrics = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            key = (row.get("Model", ""), row.get("Classifier", ""))
            metrics[key] = row
    return metrics


def metric_for_result(result):
    metrics = load_result_metrics()
    model_key = result.get("model_key")
    if model_key == "tfidf":
        return metrics.get(("tfidf", "LogisticRegression"))
    if model_key == "embedding_lr":
        return metrics.get(("embedding", "LogisticRegression"))
    if model_key == "embedding_svm":
        return metrics.get(("embedding", "SVM"))
    return None


def model_trust_note(result):
    if result.get("model_key") == "consensus":
        return (
            "Bu konsensüs, modellerin ortak eğilimini gösterir. Yine de modeller aynı veri "
            "ailesinden geldiği için ortak yanılgıları olabilir."
        )

    metric = metric_for_result(result)
    metric_text = ""
    if metric:
        metric_text = (
            f" Test setinde F1 {float(metric['F1-Score']):.3f}, "
            f"accuracy {float(metric['Accuracy']):.3f} ölçülmüş."
        )

    return (
        "Bu proje modeli haber doğrulama otoritesi değildir; sınırlı veriyle eğitildiği "
        "için yeni konulara, eksik bağlama ve manipülatif dile karşı yanılabilir."
        f"{metric_text} Sonuç, kaynak kontrolüyle birlikte okunmalıdır."
    )


def build_explanation(result, text):
    prediction = result["prediction"]
    confidence = result["confidence"]
    text_length = len(text.split())
    model_label = result.get("model_display_name", result.get("model_label", "Seçilen model"))

    if prediction == "fake":
        headline = "Bu haber yanlış olma tarafına daha yakın görünüyor."
        stance = (
            f"{model_label}, metindeki örüntüleri yanlış haber örneklerine daha yakın buldu."
        )
    else:
        headline = "Bu haber doğru olma tarafına daha yakın görünüyor."
        stance = (
            f"{model_label}, metindeki örüntüleri doğru haber örneklerine daha yakın buldu."
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


def render_single_result(result, text, compact=False):
    headline, explanation = build_explanation(result, text)
    style = "fake" if result["prediction"] == "fake" else "real"
    verdict = "Doğruya yakın" if result["prediction"] == "real" else "Yanlışa yakın"
    real_width = int(result["real_probability"] * 100)
    fake_width = int(result["fake_probability"] * 100)

    st.markdown(
        f"""
        <div class="result-box {style}">
            <div class="answer-topline">
                <div>
                    <div class="verdict-pill">{verdict}</div>
                    <div class="result-label">{headline}</div>
                </div>
            </div>
            <div class="result-copy">{explanation}</div>
            <div class="small-note">Seçilen model: {result.get("model_display_name", result.get("model_label", "TF-IDF + Logistic Regression"))}</div>
            <div class="probability-grid">
                <div class="probability-card true">
                    <span>Doğru olma olasılığı</span>
                    <strong>{percent(result["real_probability"])}</strong>
                    <div class="mini-bar"><span style="width: {real_width}%"></span></div>
                </div>
                <div class="probability-card false">
                    <span>Yanlış olma olasılığı</span>
                    <strong>{percent(result["fake_probability"])}</strong>
                    <div class="mini-bar"><span style="width: {fake_width}%"></span></div>
                </div>
            </div>
            <div class="trust-note">{escape(model_trust_note(result))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if compact:
        return

    with st.expander("Gerekçe sinyalleri", expanded=True):
        render_reason_panel(result)
    with st.expander("Metin kalite kontrolü ve sınırlılıklar", expanded=False):
        render_quality_panel(text)
    st.caption(
        "Not: Bu sonuç doğrulama kararı değil, modelin istatistiksel tahminidir. "
        "Haber yine de kaynak kontrolüyle değerlendirilmelidir."
    )


def render_quality_panel(text):
    word_count, notes = text_quality_notes(text)
    items = "".join(f"<li>{escape(note)}</li>" for note in notes)
    html = (
        '<div class="quality-panel">'
        f"<strong>Metin kalite kontrolü: {word_count} kelime</strong>"
        f"<ul>{items}</ul>"
        "<p class=\"reason-copy\" style=\"margin-top:10px;\">"
        "Model sınırlı proje verisiyle eğitildiği için sonuçlar araştırma/demo amacıyla "
        "okunmalıdır; gerçek doğrulama için haber kaynağı, tarih, bağlam ve bağımsız "
        "kaynaklar kontrol edilmelidir."
        "</p>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_reason_panel(result):
    reasons = result.get("reasons", [])
    summary = result.get(
        "reason_summary",
        "Model metindeki örüntüleri eğitim verisindeki örneklerle karşılaştırarak karar verdi.",
    )
    model_key = result.get("model_key", "tfidf")
    probability_gap = abs(result["real_probability"] - result["fake_probability"])
    confidence_label = confidence_text(result["confidence"])
    stats = (
        '<div class="reason-stats">'
        '<div class="reason-stat"><span>Tahmin</span>'
        f'<strong>{escape(display_label(result["prediction"]))}</strong></div>'
        '<div class="reason-stat"><span>Olasılık farkı</span>'
        f"<strong>{percent(probability_gap)}</strong></div>"
        '<div class="reason-stat"><span>Güven seviyesi</span>'
        f"<strong>{escape(confidence_label)}</strong></div>"
        "</div>"
    )

    if model_key == "tfidf":
        body = render_reason_groups(reasons, result["prediction"])
        limitation = (
            "Bu açıklama modelin gerçek iç mantığına en yakın okunabilir özetidir; "
            "yine de haberin doğru olup olmadığını kanıtlamaz, sadece hangi metin "
            "parçalarının sınıflandırmayı ittiğini gösterir."
        )
    elif model_key == "consensus":
        agreement = result.get("agreement", 0)
        total_models = result.get("total_models", 0)
        body = (
            '<div class="recommendation">'
            f"Mevcut {total_models} modelin {agreement} tanesi "
            f"{escape(display_label(result['prediction']))} kararına yakın sonuç verdi. "
            "Bu özet tek bir model yerine modellerin ortalama eğilimini gösterir."
            "</div>"
        )
        limitation = (
            "Konsensüs sonucu modellerin ortak eğilimini verir; yine de dış kaynak "
            "doğrulaması yerine geçmez."
        )
    else:
        reference = result.get("lexical_reference")
        reference_body = ""
        if reference:
            reference_body = (
                f'<div class="reason-section-title">Görünür metin sinyalleri '
                f'(kelime tabanlı referans: {escape(display_label(reference["prediction"]))}, '
                f'doğru {percent(reference["real_probability"])}, '
                f'yanlış {percent(reference["fake_probability"])})</div>'
                + render_reason_groups(reference["reasons"], reference["prediction"])
            )
        body = (
            '<div class="reason-warning">'
            "Anlamsal model kararını cümle anlam vektöründen verdiği için "
            "tek tek kelime ağırlıkları doğrudan bu modelin kararı değildir. "
            "Aşağıdaki sinyaller, aynı metnin kelime tabanlı modelde hangi görünür ifadelerle "
            "hangi yöne çekildiğini destekleyici olarak gösterir."
            "</div>"
            + reference_body
        )
        limitation = (
            "Anlamsal açıklama daha soyuttur; çünkü model kelime saymak yerine "
            "haberin genel anlamını eğitimdeki örneklere göre konumlandırır."
        )

    html = (
        '<div class="reason-panel">'
        "<h4>Neye göre böyle düşünüyor?</h4>"
        f'<p class="reason-copy">{escape(summary)}</p>'
        f"{stats}{body}"
        f'<div class="reason-warning">{escape(limitation)}</div>'
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def confidence_text(confidence):
    if confidence >= 0.80:
        return "Yüksek"
    if confidence >= 0.65:
        return "Orta"
    return "Düşük"


def render_reason_groups(reasons, prediction):
    if not reasons:
        return (
            '<p class="reason-copy">'
            "Bu metinde model sözlüğünde güçlü bir görünür kelime/n-gram sinyali bulunamadı."
            "</p>"
        )

    predicted = [item for item in reasons if item.get("aligns_with_prediction")]
    counter = [item for item in reasons if not item.get("aligns_with_prediction")]
    max_strength = max((item["strength"] for item in reasons), default=1.0)

    html = ""
    if predicted:
        html += (
            f'<div class="reason-section-title">{escape(display_label(prediction))} tahminini destekleyen sinyaller</div>'
            + render_reason_chips(predicted[:6], max_strength, "predicted")
        )
    if counter:
        html += (
            '<div class="reason-section-title">Karşı yönde görünen sinyaller</div>'
            + render_reason_chips(counter[:4], max_strength, "counter")
        )
    return html


def render_reason_chips(items, max_strength, chip_type):
    chips = []
    for item in items:
        width = max(8, min(100, int((item["strength"] / max_strength) * 100)))
        chips.append(
            f'<div class="reason-chip {chip_type}">'
            f'<div class="reason-token">{escape(item["token"])}</div>'
            f'<div class="reason-meta">{escape(probability_label(item["target_class"]))}</div>'
            f'<div class="reason-bar"><span style="width: {width}%"></span></div>'
            "</div>"
        )
    return f'<div class="reason-grid">{"".join(chips)}</div>'


def render_comparison(results, text, compact=False):
    consensus = consensus_from_results(results)
    render_single_result(consensus, text, compact=compact)
    if compact:
        return
    render_model_table(results)


def render_model_table(results):
    rows = []
    for result in results:
        rows.append(
            "<tr>"
            f"<td>{escape(result['model_display_name'])}</td>"
            f"<td>{escape(display_label(result['prediction']))}</td>"
            f"<td>{percent(result['real_probability'])}</td>"
            f"<td>{percent(result['fake_probability'])}</td>"
            "</tr>"
        )

    html = (
        '<div class="reason-panel">'
        "<h4>Model karşılaştırması</h4>"
        '<table class="model-table">'
        "<thead><tr><th>Model</th><th>Karar</th><th>Doğru</th><th>Yanlış</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        '<div class="recommendation">'
        "Modeller aynı yöne yakınsa sonuç daha güvenilir okunabilir. "
        "Modeller ayrışıyorsa haber metninde daha fazla bağlam veya kaynak kontrolü gerekir."
        "</div>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_assistant_content(content, compact=False):
    if content.get("mode") == "compare":
        render_comparison(content["results"], content["text"], compact=compact)
    else:
        render_single_result(content["result"], content["text"], compact=compact)


st.markdown(
    """
    <div class="app-title">
        <div class="brand-row">
            <div class="brand-mark">AI</div>
            <div class="brand-name">Haber Doğruluk Asistanı</div>
        </div>
        <h1>Haber metnini gönder, doğruluk analizini konuşarak al.</h1>
        <p>Modeli üstten seç, haberi alttaki kutuya yaz. Asistan olasılığı, güveni ve gerekçeyi sohbet akışında açıklar.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

available_keys = available_model_keys()
if not available_keys:
    st.error("Hiç model dosyası bulunamadı. Önce en az kelime tabanlı modeli eğitmek gerekiyor.")
    st.code("python src/train_tfidf.py", language="bash")
    st.stop()

st.markdown(
    """
    <div class="top-controls">
        <div class="control-label">Model</div>
        <div class="control-copy">ChatGPT'de model seçer gibi analiz modelini seç.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

model_options = [model_display_name(MODEL_CONFIGS[key]) for key in available_keys]
if len(available_keys) > 1:
    model_options.append("Konsensüs Analizi (Tüm modeller)")

selected_model_label = st.selectbox(
    "Model",
    options=model_options,
    label_visibility="collapsed",
)

if st.button("Yeni sohbet", use_container_width=False):
    st.session_state.pop("messages", None)
    st.rerun()

selected_model_key = None
if selected_model_label == "Konsensüs Analizi (Tüm modeller)":
    st.caption("Tüm modeller çalışır; ortak karar, ortalama olasılık ve karşılaştırma tablosu gösterilir.")
else:
    selected_model_key = next(
        key for key in available_keys if model_display_name(MODEL_CONFIGS[key]) == selected_model_label
    )
    st.caption(MODEL_CONFIGS[selected_model_key]["description"])

missing = missing_model_labels()
if missing:
    st.markdown(
        f"""
        <div class="missing-model">
            Şu an aktif olmayan modeller: {", ".join(missing)}.
            Anlamsal modelleri görmek için <code>python src/train_embeddings.py</code>
            çalıştırılmalı.
        </div>
        """,
        unsafe_allow_html=True,
    )

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Merhaba. Haber başlığını veya içeriğini gönder; seçili modele göre analiz edeyim.",
        }
    ]

last_assistant_index = max(
    (
        index
        for index, message in enumerate(st.session_state.messages)
        if message["role"] == "assistant" and isinstance(message["content"], dict)
    ),
    default=-1,
)

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if isinstance(message["content"], dict):
            render_assistant_content(message["content"], compact=index != last_assistant_index)
        else:
            st.write(message["content"])

if len(st.session_state.messages) <= 1:
    st.markdown(
        """
        <div class="empty-chat">
            <h3>Bir haber metniyle başlayalım</h3>
            <p>Başlık, açıklama veya tam haber içeriğini alttaki kutuya yapıştırabilirsin.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sample_cols = st.columns(3)
    samples = [
        "Deprem sonrası bölgede 10 bin yeni konutun teslim edildiği açıklandı.",
        "Sosyal medyada yayılan iddiaya göre içme suyuna zararlı madde karıştırıldı.",
        "Bir bakanlığın yeni uygulamayı gelecek ay başlatacağı duyuruldu.",
    ]
    for col, sample in zip(sample_cols, samples):
        if col.button(sample, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": sample})
            if selected_model_label == "Konsensüs Analizi (Tüm modeller)" and len(available_keys) > 1:
                with st.spinner("Modeller birlikte düşünüyor..."):
                    results = [cached_analyze_with_model(key, sample) for key in available_keys]
                assistant_content = {"mode": "compare", "results": results, "text": sample}
            else:
                with st.spinner("Haber analiz ediliyor..."):
                    result = cached_analyze_with_model(selected_model_key, sample)
                assistant_content = {"mode": "single", "result": result, "text": sample}
            st.session_state.messages.append({"role": "assistant", "content": assistant_content})
            st.rerun()

prompt = st.chat_input("Haber başlığını veya içeriğini buraya yaz...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    if selected_model_label == "Konsensüs Analizi (Tüm modeller)" and len(available_keys) > 1:
        with st.spinner("Modeller birlikte düşünüyor..."):
            results = [cached_analyze_with_model(key, prompt) for key in available_keys]
        assistant_content = {"mode": "compare", "results": results, "text": prompt}
    else:
        with st.spinner("Haber analiz ediliyor..."):
            result = cached_analyze_with_model(selected_model_key, prompt)
        assistant_content = {"mode": "single", "result": result, "text": prompt}

    st.session_state.messages.append({"role": "assistant", "content": assistant_content})
    st.rerun()
