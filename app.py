import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import plotly.express as px
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
    DEV_METRICS_AVAILABLE = True
except Exception:
    DEV_METRICS_AVAILABLE = False

from core.detector import (
    normalize_events,
    build_features,
    load_profiles,
    load_assets,
    train_classifier_from_weak_labels,
    detect_incidents,
    build_report_markdown,
)

APP_TITLE = "Прототип интеллектуальной системы обнаружения аномалий"

st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")

# Косметика интерфейса. Цвета ползунков Streamlit удобно менять CSS-ом именно здесь.
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab"] {
    color: white !important;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: white !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
    background-color: #f97316 !important;   /* оранжевый */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_EVENTS = DATA_DIR / "sample_events.csv"
DEFAULT_PROFILES = DATA_DIR / "threat_profiles.json"
DEFAULT_ASSETS = DATA_DIR / "assets_inventory.csv"

# Русские названия классов. Внутри алгоритма классы остаются короткими английскими ключами,
# потому что так проще хранить их в JSON/CSV и использовать в коде.
CLASS_RU = {
    "normal": "нормальная активность",
    "scan": "сканирование",
    "bruteforce": "подбор учетных данных",
    "flood": "DDoS/флуд-активность",

    "web_probe": "подозрительная активность веб-сервиса",
    "web_attack": "подозрительная активность веб-сервиса",

    "dns_anomaly": "DNS-аномалия",
    "lateral_movement": "латеральное перемещение",
    "data_exfiltration": "подозрительный вывод данных",
    "malware_beaconing": "признаки командного канала",

    "suspicious_outbound": "подозрительная исходящая активность",
    "unknown_anomaly": "неизвестная аномалия",
}

CLASS_ORDER = [
    "сканирование",
    "подбор учетных данных",
    "DDoS/флуд-активность",
    "подозрительная активность веб-сервиса",
    "DNS-аномалия",
    "латеральное перемещение",
    "подозрительный вывод данных",
    "признаки командного канала",
    "подозрительная исходящая активность",
    "неизвестная аномалия",
]

FEATURE_RU = {
    "event_count": "количество событий",
    "unique_dst_ips": "уникальные адреса назначения",
    "unique_dst_ports": "уникальные порты назначения",
    "denied_rate": "доля отказов",
    "auth_fail_count": "неудачные входы",
    "unique_users": "уникальные пользователи",
    "http_request_count": "HTTP-запросы",
    "http_error_rate": "доля HTTP-ошибок",
    "ids_alert_count": "срабатывания IDS/IPS",
    "bytes_sum": "объем данных",
    "external_dst_count": "внешние направления",
    "anomaly_score": "показатель аномальности",

    "dns_query_count": "DNS-запросы",
    "unique_domains": "уникальные домены",
    "suspicious_domain_count": "подозрительные домены",
    "web_attack_pattern_count": "web-attack паттерны",
    "cross_segment_count": "межсегментные соединения",
    "lateral_service_count": "административные сервисы",
    "external_connection_count": "внешние соединения",
    "external_bytes_sum": "объем внешнего трафика",
    "small_external_connection_count": "малые внешние соединения",
}

CHART_COLORS = ["#60a5fa", "#f97316", "#22c55e", "#e879f9", "#facc15", "#fb7185", "#38bdf8", "#a78bfa", "#34d399", "#f472b6", "#fb923c", "#2dd4bf", "#c084fc", "#f87171", "#84cc16", "#06b6d4", "#eab308", "#8b5cf6", "#10b981", "#ef4444", "#3b82f6", "#d946ef", "#14b8a6", "#f59e0b", "#ec4899"]


def ru_label(value: str) -> str:
    return CLASS_RU.get(str(value), str(value))


def read_uploaded_file(uploaded_file):
    """Читает загруженные журналы или демонстрационный набор событий."""
    if uploaded_file is None:
        return pd.read_csv(DEFAULT_EVENTS)
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".json"):
        return pd.read_json(uploaded_file)
    if name.endswith(".jsonl"):
        return pd.read_json(uploaded_file, lines=True)
    raise ValueError("Поддерживаются только CSV, JSON и JSONL")


def safe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Streamlit отображает таблицы через Apache Arrow. Если в столбце смешаны
    строки и списки/словари, Arrow падает. Сложные значения переводятся в JSON-строки.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict, tuple, set)) else x
            )
    return out


def risk_badge(level: str) -> str:
    cls = {"Высокий": "risk-high", "Средний": "risk-medium", "Низкий": "risk-low"}.get(level, "")
    return f"<span class='{cls}'>{level}</span>"


def risk_map_interactive(anomalies: pd.DataFrame):
    """
    Интерактивная карта рисков.

    Ось X — ущерб / критичность воздействия.
    Ось Y — оцененная вероятность проявления.

    В ячейках отображается количество обнаруженных аномалий,
    попавших в соответствующую категорию риска.
    """

    x_labels = ["Очень низкая", "Низкая", "Средняя", "Высокая", "Очень высокая"]
    y_labels = ["Незначительные", "Низкие", "Средние", "Существенные", "Катастрофические"]

    # Фоновая матрица зон риска.
    # Чем больше значение, тем "опаснее" зона.
    z = [
        [1, 2, 3, 4, 5],
        [2, 4, 6, 8, 10],
        [3, 6, 9, 12, 15],
        [4, 8, 12, 16, 20],
        [5, 10, 15, 20, 25],
    ]

    count_matrix = [[0 for _ in range(5)] for _ in range(5)]

    if anomalies is not None and len(anomalies):
        plot_data = anomalies.copy()

        # X = вероятность
        x_prob = (
            plot_data["confidence"].astype(float) * 0.5
            + plot_data["anomaly_score"].astype(float) * 0.5
        )
        x_num = np.clip((x_prob * 5).astype(int), 0, 4)

        # Y = последствия
        y_num = np.clip((plot_data["risk_score"].astype(float) / 20).astype(int), 0, 4)

        for x, y in zip(x_num, y_num):
            count_matrix[y][x] += 1

    # Текст в ячейках. Если аномалий нет — оставляем пусто.
    text_matrix = []
    for row in count_matrix:
        text_matrix.append([str(value) for value in row])

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=22, color="black"),
            colorscale=[
                [0.00, "#22c55e"], [0.16, "#22c55e"],   # Низкий: 1-4
                [0.16, "#facc15"], [0.36, "#facc15"],   # Умеренный: 5-9
                [0.36, "#fb923c"], [0.64, "#fb923c"],   # Средний: 10-16
                [0.64, "#ef4444"], [1.00, "#ef4444"],   # Высокий: 17-25
            ],
            zmin=1,
            zmax=25,
            showscale=False,
            hovertemplate=(
                "Ущерб: %{x}<br>"
                "Вероятность: %{y}<br>"
                "Количество аномалий: %{text}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=560,
        margin=dict(l=30, r=30, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis_title="Вероятность**",
        yaxis_title="Последствия",
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.20)",
        side="bottom",
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.20)",
    )
    return fig

def add_risk_matrix_zone(anomalies: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет к аномалиям категорию риска по той же матрице,
    которая используется на карте рисков.

    X = вероятность
    Y = последствия
    Риск = X * Y
    """

    if anomalies is None or anomalies.empty:
        return anomalies

    out = anomalies.copy()

    # X = вероятность: оценка на основе уверенности модели и показателя аномальности
    x_prob = (
        out["confidence"].astype(float) * 0.5
        + out["anomaly_score"].astype(float) * 0.5
    )
    x_num = np.clip((x_prob * 5).astype(int), 0, 4)

    # Y = последствия: приближенно берем из итогового risk_score
    y_num = np.clip((out["risk_score"].astype(float) / 20).astype(int), 0, 4)

    # Значение матрицы риска: вероятность × последствия
    risk_value = (x_num + 1) * (y_num + 1)

    out["risk_matrix_value"] = risk_value

    def zone(value):
        if value <= 4:
            return "Низкий"
        if value <= 9:
            return "Умеренный"
        if value <= 16:
            return "Средний"
        return "Высокий"

    out["risk_matrix_zone"] = out["risk_matrix_value"].apply(zone)

    return out

def build_eval_frame(features: pd.DataFrame, classifier_info: dict, anomaly_threshold: float) -> pd.DataFrame:
    """
    Таблица для оценки качества. Если есть true_label / label — используется она.
    Иначе используется слабая разметка как демонстрационный эталон.
    """
    eval_df = features.copy()
    if classifier_info.get("status") == "trained":
        model = classifier_info["model"]
        cols = classifier_info["feature_cols"]
        x = eval_df[cols].fillna(0).astype(float)
        proba = model.predict_proba(x)
        pred = model.predict(x)
        eval_df["pred_label"] = pred
        eval_df["pred_confidence"] = proba.max(axis=1)
        classes = list(model.classes_)
        eval_df["attack_score"] = 1 - proba[:, classes.index("normal")] if "normal" in classes else eval_df["pred_confidence"]
    else:
        eval_df["pred_label"] = np.where(eval_df["anomaly_score"] >= anomaly_threshold, "unknown_anomaly", eval_df["weak_label"])
        eval_df["pred_confidence"] = eval_df["anomaly_score"].clip(0, 1)
        eval_df["attack_score"] = eval_df["pred_confidence"]

    if "true_label" in eval_df.columns and eval_df["true_label"].astype(str).str.strip().ne("").any():
        eval_df["eval_label"] = eval_df["true_label"].replace("", np.nan).fillna("normal").astype(str)
        eval_df["eval_source"] = "истинная разметка из файла"
    else:
        eval_df["eval_label"] = eval_df["weak_label"].astype(str)
        eval_df["eval_source"] = "слабая разметка (демонстрационная оценка)"

    eval_df["true_attack"] = (eval_df["eval_label"] != "normal").astype(int)
    eval_df["pred_attack"] = ((eval_df["pred_label"] != "normal") | (eval_df["anomaly_score"] >= anomaly_threshold)).astype(int)
    eval_df["eval_label_ru"] = eval_df["eval_label"].apply(ru_label)
    eval_df["pred_label_ru"] = eval_df["pred_label"].apply(ru_label)
    return eval_df


def plot_confusion_matrix(eval_df: pd.DataFrame):
    """
    Бинарная матрица ошибок для оценки главной задачи:
    отличить атаку/аномалию от нормальной активности.

    TP — атака правильно обнаружена.
    FP — норма ошибочно принята за атаку.
    FN — атака пропущена.
    TN — норма правильно определена как норма.
    """

    y_true = eval_df["true_attack"].astype(int)
    y_pred = eval_df["pred_attack"].astype(int)

    # sklearn возвращает матрицу в порядке:
    # [[TN, FP],
    #  [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    matrix = [[tp, fp], [fn, tn]]
    labels = [
        [f"TP\n{tp}", f"FP\n{fp}"],
        [f"FN\n{fn}", f"TN\n{tn}"],
    ]

    from matplotlib.colors import LinearSegmentedColormap

    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    orange_pink_cmap = LinearSegmentedColormap.from_list(
        "orange_pink_confusion",
        [
            "#fed7aa",  # пастельный оранжевый
            "#fdba74",  # мягкий насыщенный оранжевый
            "#f9a8d4",  # пастельный розовый
            "#f472b6",  # насыщенный розовый
        ]
    )

    im = ax.imshow(matrix, cmap=orange_pink_cmap)

    ax.set_title("Матрица ошибок бинарного обнаружения")
    ax.set_xlabel("Фактическое значение")
    ax.set_ylabel("Предсказанное значение")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Атака / аномалия", "Норма"], rotation=20, ha="right")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Атака / аномалия", "Норма"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                labels[i][j],
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    return fig


def plot_roc(eval_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5.6))
    if eval_df["true_attack"].nunique() < 2:
        ax.text(0.5, 0.5, "ROC-кривая недоступна:\nв разметке только один класс", ha="center", va="center")
        ax.set_axis_off()
        return fig
    fpr, tpr, _ = roc_curve(eval_df["true_attack"], eval_df["attack_score"])
    auc_value = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"Площадь под кривой = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC-кривая")
    ax.set_xlabel("Доля ложных срабатываний")
    ax.set_ylabel("Полнота обнаружения атак")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_feature_importance(classifier_info: dict):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    if classifier_info.get("status") != "trained" or not hasattr(classifier_info.get("model"), "feature_importances_"):
        ax.text(0.5, 0.5, "Важность признаков недоступна:\nмодель не обучена", ha="center", va="center")
        ax.set_axis_off()
        return fig
    imp = pd.Series(classifier_info["model"].feature_importances_, index=classifier_info["feature_cols"])
    imp.index = [FEATURE_RU.get(v, v) for v in imp.index]
    imp = imp.sort_values().tail(12)
    ax.barh(imp.index, imp.values)
    ax.set_title("Важность признаков")
    ax.set_xlabel("Вклад признака в решение модели")
    fig.tight_layout()
    return fig


st.title("Прототип интеллектуальной системы обнаружения аномалий")
st.caption("Анализ журналов: нормализация, признаки, профили угроз, слабая разметка, модель, показатель аномальности и карточки аномалий.")

with st.sidebar:
    st.header("Параметры анализа")
    uploaded = st.file_uploader("Загрузить журналы событий", type=["csv", "json", "jsonl"])
    st.caption("Если файл не выбран, используется демонстрационный набор журналов.")

try:
    raw_events = read_uploaded_file(uploaded)
except Exception as exc:
    st.error(f"Не удалось прочитать файл: {exc}")
    st.stop()

profiles = load_profiles(DEFAULT_PROFILES)
assets = load_assets(DEFAULT_ASSETS)

perf = {}
t0 = time.perf_counter()
events_all = normalize_events(raw_events)
perf["normalization_sec"] = time.perf_counter() - t0

with st.sidebar:
    st.subheader("Период анализа")
    if len(events_all):
        min_ts = events_all["timestamp"].min().to_pydatetime()
        max_ts = events_all["timestamp"].max().to_pydatetime()
        if min_ts == max_ts:
            selected_range = (min_ts, max_ts)
            st.info("В журнале найдено одно значение времени.")
        else:
            selected_range = st.slider(
                "Фильтр времени",
                min_value=min_ts,
                max_value=max_ts,
                value=(min_ts, max_ts),
                format="DD.MM.YYYY HH:mm:ss",
                help="Это фильтр периода, аналог выбора диапазона времени в Kibana. Он не заменяет окно агрегации.",
            )
    else:
        selected_range = None

    st.subheader("Окно агрегации")
    window_minutes = st.selectbox(
        "Период группировки событий",
        [1, 5, 15, 30, 60],
        index=1,
        help="Окно определяет, за какой интервал события объединяются в один вектор признаков. Например, сканирование считается не по одной строке, а по активности источника за 5 минут.",
    )

    st.subheader("Чувствительность")
    anomaly_threshold = st.slider(
        "Порог показателя аномальности",
        0.0, 1.0, 0.60, 0.05,
        help="Ниже порог — больше чувствительность и больше тревог. Выше порог — меньше ложных тревог, но выше риск пропуска слабой атаки.",
    )

    min_risk_to_show = st.selectbox(
    "Показывать аномалии от уровня",
    ["Низкий", "Умеренный", "Средний", "Высокий"],
    index=0,
)

    st.divider()
    st.subheader("Пороговые условия")
    scan_ports = st.slider("Порт-скан: уникальных портов на одном узле", 5, 80, 18, 1, help="Вертикальное сканирование: один источник перебирает много портов на одном узле.")
    scan_hosts = st.slider("Сканирование сети: уникальных узлов назначения", 5, 80, 20, 1, help="Горизонтальное сканирование: один источник обращается к множеству узлов.")
    brute_fails = st.slider("Подбор: неудачных входов", 3, 80, 10, 1)
    web_errors = st.slider("Web/Web-атаки: доля HTTP-ошибок", 0.1, 1.0, 0.45, 0.05)
    flood_events = st.slider("Флуд: событий на сервис", 20, 500, 120, 10)
    st.divider()
    st.caption("Запуск: streamlit run app.py")
    st.subheader("Расширенные профили")

    dns_queries = st.slider(
        "DNS: запросов за окно",
        5, 200, 25, 5,
        help="Используется для выявления DNS-аномалий и возможного DNS-туннелирования."
    )

    unique_domains = st.slider(
        "DNS: уникальных доменов",
        5, 200, 20, 5,
        help="Большое число уникальных доменов за короткое окно может указывать на DGA/DNS-туннелирование."
    )

    lateral_connections = st.slider(
        "Lateral movement: межсегментных соединений",
        3, 100, 10, 1,
        help="Срабатывает при нетипичных обращениях между VLAN/сегментами."
    )

    lateral_service_hits = st.slider(
        "Lateral movement: обращений к админским сервисам",
        1, 50, 5, 1,
        help="SMB/RDP/SSH/WinRM между сегментами."
    )

    exfil_bytes = st.slider(
        "Вывод данных: байт наружу",
        500_000, 50_000_000, 5_000_000, 500_000,
        help="Порог большого исходящего объема данных во внешний сегмент."
    )

    beacon_events = st.slider(
        "Beaconing: малых внешних соединений",
        3, 100, 8, 1,
        help="Количество повторяющихся малых внешних соединений на нетипичные порты."
    )
if selected_range:
    start_ts, end_ts = selected_range
    events = events_all[(events_all["timestamp"] >= pd.Timestamp(start_ts)) & (events_all["timestamp"] <= pd.Timestamp(end_ts))].copy()
else:
    events = events_all.copy()

thresholds = {
    "scan_ports": scan_ports,
    "scan_hosts": scan_hosts,
    "brute_fails": brute_fails,
    "web_errors": web_errors,
    "flood_events": flood_events,
    "anomaly_threshold": anomaly_threshold,
    "dns_queries": dns_queries,
    "unique_domains": unique_domains,
    "lateral_connections": lateral_connections,
    "lateral_service_hits": lateral_service_hits,
    "exfil_bytes": exfil_bytes,
    "beacon_events": beacon_events,
}

t1 = time.perf_counter()
features = build_features(events, window_minutes=window_minutes, thresholds=thresholds)
perf["feature_sec"] = time.perf_counter() - t1

t2 = time.perf_counter()
classifier_info = train_classifier_from_weak_labels(features)
perf["train_sec"] = time.perf_counter() - t2

t3 = time.perf_counter()
anomalies = detect_incidents(events, features, profiles, assets, classifier_info, thresholds)

# Добавляем категорию риска по матрице "вероятность × последствия",
# чтобы верхние счетчики совпадали с картой рисков.
anomalies = add_risk_matrix_zone(anomalies)

perf["detect_sec"] = time.perf_counter() - t3
perf["total_sec"] = sum(perf.values())

# Для совместимости с прежними именами столбцов функция detector.py пока возвращает incident_id.
risk_order = {"Низкий": 1, "Умеренный": 2, "Средний": 3, "Высокий": 4}

visible_anomalies = anomalies[
    anomalies["risk_matrix_zone"].map(risk_order).fillna(0) >= risk_order[min_risk_to_show]
].copy()

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

col1.metric("Событий обработано", f"{len(events):,}".replace(",", " "))
col2.metric("Векторов признаков", f"{len(features):,}".replace(",", " "))
col3.metric("Аномалий найдено", f"{len(anomalies):,}".replace(",", " "))

col4.metric(
    "Низкий риск",
    f"{(anomalies['risk_matrix_zone'] == 'Низкий').sum():,}".replace(",", " ")
)

col5.metric(
    "Умеренный риск",
    f"{(anomalies['risk_matrix_zone'] == 'Умеренный').sum():,}".replace(",", " ")
)

col6.metric(
    "Средний риск",
    f"{(anomalies['risk_matrix_zone'] == 'Средний').sum():,}".replace(",", " ")
)

col7.metric(
    "Высокий риск",
    f"{(anomalies['risk_matrix_zone'] == 'Высокий').sum():,}".replace(",", " ")
)

if classifier_info.get("status") == "trained":
    class_count = classifier_info.get(
        "class_count",
        len(classifier_info.get("classes", []))
    )

    train_rows = classifier_info.get(
        "train_rows",
        len(features)
    )

    st.success(
        f"Модель обучена на слабой разметке: классов — {class_count}, "
        f"объектов — {train_rows}. Это предварительная разметка, сформированная правилами и профилями угроз."
    )
else:
    st.warning(
        "Модель машинного обучения не обучена: недостаточно классов в слабой разметке. "
        "Правила и показатель аномальности продолжают работать."
    )

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Обзор", "Журналы", "Признаки", "Аномалии", "Активы", "Разработчик", "Отчет"])

with tab1:
    st.subheader("Общая картина")
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Распределение аномалий по типам")
        if len(anomalies):
            counts = (
                anomalies.groupby("scenario_ru")
                .size()
                .reindex(CLASS_ORDER, fill_value=0)
                .reset_index()
            )
            counts.columns = ["Тип аномалии", "Количество"]

            fig = px.bar(
                counts,
                x="Тип аномалии",
                y="Количество",
                color="Тип аномалии",
                color_discrete_sequence=CHART_COLORS
            )

            fig.update_layout(
                showlegend=False,
                height=560,
                xaxis_tickangle=-35,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=30, b=120),
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Аномалии не найдены.")
    
    with right:
        st.markdown("#### Карта рисков корпоративной сети")
        if DEV_METRICS_AVAILABLE:
            st.plotly_chart(risk_map_interactive(anomalies), use_container_width=True)

            c1, c2, c3, c4, c5 = st.columns(5)
            
            with c1:
                st.markdown("**Риск:**")
            
            with c2:
                st.markdown("🟩 Низкий")

            with c3:
                st.markdown("🟨 Умеренный")

            with c4:
                st.markdown("🟧 Средний")

            with c5:
                st.markdown("🟥 Высокий")
        else:
            st.info("Для карты рисков требуется matplotlib и plotly.")
    st.markdown("#### Последние события")
    st.dataframe(safe_for_streamlit(events.sort_values("timestamp", ascending=False).head(20)), use_container_width=True)

with tab2:
    st.subheader("Нормализованные журналы событий")
    st.caption("События приведены к единой схеме: время, источник, тип, адреса, порт, протокол, результат.")
    st.dataframe(safe_for_streamlit(events), use_container_width=True, height=520)
    st.download_button("Скачать нормализованные события CSV", data=events.to_csv(index=False).encode("utf-8-sig"), file_name="normalized_events.csv", mime="text/csv")

with tab3:
    st.subheader("Векторы признаков")
    st.caption("Признаки рассчитываются по временным окнам и используются для правил, слабой разметки, модели и показателя аномальности.")
    st.dataframe(safe_for_streamlit(features), use_container_width=True, height=520)
    st.download_button("Скачать признаки CSV", data=features.to_csv(index=False).encode("utf-8-sig"), file_name="features.csv", mime="text/csv")

with tab4:
    st.subheader("Карточки аномалий / потенциальных инцидентов")
    if visible_anomalies.empty:
        st.info("Аномалии по выбранному уровню риска не найдены.")
    else:
        anomaly_table = visible_anomalies[["incident_id", "risk_level", "scenario_ru", "time_start", "time_end", "src_ip", "dst_ip", "service", "confidence", "anomaly_score"]].sort_values(["risk_level", "confidence"], ascending=[True, False])
        st.dataframe(safe_for_streamlit(anomaly_table), use_container_width=True, height=300)
        selected_id = st.selectbox("Выберите аномалию для просмотра карточки", options=visible_anomalies["incident_id"].tolist())
        row = visible_anomalies[visible_anomalies["incident_id"] == selected_id].iloc[0]
        st.markdown("---")
        st.markdown(f"### Карточка аномалии: {row['incident_id']}")
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Тип:** {row['scenario_ru']}")
        c2.markdown(f"**Уровень риска:** {risk_badge(row['risk_level'])}", unsafe_allow_html=True)
        c3.markdown(f"**Оценка уверенности:** {row['confidence']:.2f}")
        c4, c5, c6 = st.columns(3)
        c4.markdown(f"**Источник:** `{row['src_ip']}`")
        c5.markdown(f"**Назначение:** `{row['dst_ip']}`")
        c6.markdown(f"**Сервис:** `{row['service']}`")
        st.markdown(f"**Период:** {row['time_start']} — {row['time_end']}")
        st.markdown(f"**Показатель аномальности:** `{row['anomaly_score']:.2f}`")
        st.markdown(f"**MITRE ATT&CK:** {row['mitre_tactic']} / {row['mitre_technique']}")
        st.markdown("#### Объяснение срабатывания")
        for item in json.loads(row["explanation_json"]):
            st.write(f"• {item}")
        st.markdown("#### Рекомендация")
        st.info(row["recommendation"])
        if row.get("vulnerability_context"):
            st.markdown("#### Контекст уязвимостей")
            st.warning(row["vulnerability_context"])
        st.download_button("Скачать выбранную карточку Markdown", data=build_report_markdown(visible_anomalies[visible_anomalies["incident_id"] == selected_id]).encode("utf-8"), file_name=f"{selected_id}_anomaly_report.md", mime="text/markdown")

with tab5:
    st.subheader("Активы и топология")
    st.caption("Инвентарь активов нужен для учета роли узла, критичности сервиса и контекста уязвимостей. От состава сети зависит приоритет угроз.")
    st.dataframe(safe_for_streamlit(assets), use_container_width=True)
    if len(anomalies):
        st.markdown("#### Аномалии по узлам назначения")
        asset_risk = anomalies.groupby(["dst_ip", "service", "risk_level"]).size().reset_index(name="anomaly_count")
        st.dataframe(safe_for_streamlit(asset_risk), use_container_width=True)

with tab6:
    st.subheader("Раздел разработчика: качество модели и производительность")
    eval_df = build_eval_frame(features, classifier_info, anomaly_threshold)
    if len(eval_df) and DEV_METRICS_AVAILABLE:
        st.caption(f"Источник эталонной разметки: {eval_df['eval_source'].iloc[0]}")
        y_true, y_pred = eval_df["true_attack"], eval_df["pred_attack"]
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fpr_value = fp / max(fp + tn, 1)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Точность", f"{precision:.3f}")
        m2.metric("Полнота", f"{recall:.3f}")
        m3.metric("F1-мера", f"{f1:.3f}")
        m4.metric("Доля ложных срабатываний", f"{fpr_value:.3f}")
        st.markdown("**Пояснение:** точность показывает долю корректных тревог, полнота — долю пойманных атак, F1-мера объединяет точность и полноту, доля ложных срабатываний показывает ошибочные тревоги на нормальном трафике.")
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_confusion_matrix(eval_df), use_container_width=True)
        with c2:
            st.pyplot(plot_roc(eval_df), use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            st.pyplot(plot_feature_importance(classifier_info), use_container_width=True)
        st.markdown("#### Производительность обработки")
        total = max(perf["total_sec"], 1e-9)
        events_per_sec = len(events) / total
        ms_per_event = total / max(len(events), 1) * 1000
        mbits = events["bytes"].sum() * 8 / 1_000_000
        approx_mbps = mbits / total
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Общее время обработки", f"{total:.3f} с")
        p2.metric("Событий в секунду", f"{events_per_sec:.1f}")
        p3.metric("Время на событие", f"{ms_per_event:.3f} мс")
        p4.metric("Оценка пропускной способности", f"{approx_mbps:.2f} Мбит/с")
        st.caption("Показатели рассчитаны для обработки журналов событий, а не для inline-фильтрации пакетов в сетевом разрыве.")
        st.markdown("#### Таблица оценки")
        dev_table = eval_df[["window_start", "src_ip", "eval_label_ru", "pred_label_ru", "true_attack", "pred_attack", "attack_score", "anomaly_score"]].rename(columns={
            "window_start": "окно времени",
            "src_ip": "источник",
            "eval_label_ru": "фактический класс",
            "pred_label_ru": "предсказанный класс",
            "true_attack": "факт атаки",
            "pred_attack": "прогноз атаки",
            "attack_score": "оценка атаки",
            "anomaly_score": "показатель аномальности",
        })
        st.dataframe(safe_for_streamlit(dev_table), use_container_width=True, height=300)
    else:
        st.info("Нет данных или зависимостей для оценки качества.")

with tab7:
    st.subheader("Сводный отчет")
    report_md = build_report_markdown(visible_anomalies)
    st.markdown(report_md)
    st.download_button("Скачать отчет Markdown", data=report_md.encode("utf-8"), file_name="anomaly_report.md", mime="text/markdown")
    st.download_button("Скачать аномалии CSV", data=visible_anomalies.to_csv(index=False).encode("utf-8-sig"), file_name="anomalies.csv", mime="text/csv")
