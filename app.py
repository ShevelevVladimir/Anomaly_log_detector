import json
import time
from pathlib import Path
import hmac
import os
import re
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

from ingest import read_many_logs, SUPPORTED_EXTENSIONS

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

def check_admin_auth() -> bool:
    """
    Простая авторизация для прототипа.
    Нужна для ограничения доступа к журналам, настройкам и карточкам инцидентов.
    """

    if "auth_ok" not in st.session_state:
        st.session_state["auth_ok"] = False
    if "auth_error" not in st.session_state:
        st.session_state["auth_error"] = ""

    if st.session_state["auth_ok"]:
        return True

    st.title("Авторизация администратора")
    st.caption("Введите учетные данные администратора прототипа.")

    admin_login = os.getenv("ANOMALY_ADMIN_LOGIN", "admin")
    admin_password = os.getenv("ANOMALY_ADMIN_PASSWORD", "admin123")

    with st.form("admin_auth_form"):
        login = st.text_input("Логин")
        password = st.text_input("Пароль", type="password")
        submitted = st.form_submit_button("Войти")

    if submitted:
        login_ok = hmac.compare_digest(str(login), str(admin_login))
        password_ok = hmac.compare_digest(str(password), str(admin_password))
        if login_ok and password_ok:
            st.session_state["auth_ok"] = True
            st.session_state["auth_error"] = ""
            st.rerun()
        else:
            st.session_state["auth_error"] = "Неверный логин или пароль. Проверьте учетные данные администратора."

    if st.session_state["auth_error"]:
        st.error(st.session_state["auth_error"])
        st.info("По умолчанию для локального прототипа используется логин admin и пароль admin123, если переменные окружения не переопределены.")

    st.stop()


check_admin_auth()

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

WINDOWS_RULES = DATA_DIR / "windows_event_rules.json"
LINUX_RULES = DATA_DIR / "linux_event_rules.json"
NETWORK_RULES = DATA_DIR / "network_event_rules.json"

POLICY_LABELS = {
    "ignore_single_wrong_password": "Windows: Игнорировать единичные ошибки пароля",
    "strict_unknown_user_control": "Windows: Детектировать вход под несуществующими пользователями",
    "detect_success_after_failures": "Windows/Auth: Детектировать успешный вход после серии ошибок",
    "allow_rdp_logons": "Windows: Разрешить RDP-входы в выбранной зоне",
    "allow_smb_rpc_access": "Windows: Разрешить SMB/RPC-доступ",
    "detect_remote_service_creation": "Windows: Детектировать создание новых служб",
    "exclude_web_ports_from_scan": "Сеть: Исключить порты 80/443 из анализа сканирования",
    "analyze_wfp_blocked_connections": "Windows: Детектировать сканирование по заблокированным WFP-соединениям",
    "enable_noisy_wfp_packet_mode": "Windows: Учитывать заблокированные WFP-пакеты 5152",
    "detect_web_server_shell": "Windows/Web: Детектировать запуск cmd/powershell веб-сервером",
    "detect_cmd_network_activity": "Windows: Детектировать сетевую активность cmd/powershell",
    "monitor_mass_file_read": "Windows: Детектировать массовое чтение файлов",
    "detect_suspicious_archives": "Windows: Детектировать создание архивов в подозрительных каталогах",
    "enable_dns_tunneling_detection": "Windows/DNS: Детектировать признаки DNS-туннелирования",
    "ignore_internal_dns_domains": "DNS: Игнорировать внутренние домены",
    "enable_beaconing_periodicity": "Windows/Сеть: Детектировать периодические соединения",
    "exclude_known_cloud_services": "Сеть: Игнорировать известные облачные сервисы",
    "enable_flood_detection": "Windows/Сеть: Детектировать аномальный объем соединений",

    "strict_invalid_user_control": "Linux/SSH: Детектировать вход под несуществующим пользователем",
    "use_ssh_password_failure_buffer": "Linux/SSH: Накопительно учитывать ошибки пароля",
    "allow_outbound_ssh_internal": "Linux: Разрешить исходящие SSH/SCP-сессии во внутренней сети",
    "allow_inbound_ssh_logins": "Linux: Разрешить входящие SSH-сессии",
    "analyze_linux_firewall_blocks": "Linux/Firewall: Детектировать сканирование по DROP/REJECT",
    "detect_shell_network_connections": "Linux: Детектировать сетевые соединения из shell",
    "monitor_linux_mass_file_read": "Linux: Детектировать массовое чтение файлов",
    "detect_linux_suspicious_archives": "Linux: Детектировать архивирование пользовательских данных",
    "monitor_linux_file_transfer_logs": "Linux: Учитывать события скачивания FTP/SFTP/SCP",
    "enable_linux_dns_tunneling_detection": "Linux/DNS: Детектировать признаки DNS-туннелирования",
    "monitor_dns_txt_queries": "Linux/DNS: Контролировать TXT-запросы",
    "enable_linux_beaconing_periodicity": "Linux/Сеть: Детектировать периодические connect-соединения",
    "ignore_linux_update_repositories": "Linux: Игнорировать обращения к репозиториям обновлений",
    "detect_kernel_syn_flood_warnings": "Linux: Детектировать предупреждения ядра о SYN flood",
    "detect_conntrack_overflow": "Linux: Детектировать переполнение conntrack",
    "enable_linux_flood_detection": "Linux/Сеть: Детектировать флуд по firewall/conntrack",
    "ignore_system_daemon_noise": "Linux: Игнорировать события системных демонов",
    "ignore_os_update_noise": "Linux: Игнорировать события обновления ОС",
    "enable_linux_web_uri_analysis": "Linux/Web: Анализировать URI на признаки веб-атак",
    "detect_linux_web_server_shell": "Linux/Web: Детектировать запуск shell веб-сервером",

    "detect_network_auth_bruteforce": "Сеть/Auth: Детектировать подбор учетных данных",
    "detect_password_spraying": "Сеть/Auth: Детектировать password spraying",
    "detect_port_scan": "Сеть: Детектировать сканирование портов",
    "detect_host_scan": "Сеть: Детектировать сканирование узлов",
    "detect_denied_scan": "Firewall: Учитывать блокировки при детекте сканирования",
    "use_ids_scan_alerts": "IDS/IPS: Учитывать сигнатуры сканирования",
    "detect_network_flood": "Сеть: Детектировать флуд по объему соединений",
    "detect_distributed_flood": "Сеть: Детектировать распределенный флуд",
    "detect_syn_flood": "Сеть: Детектировать SYN flood",
    "use_ids_flood_alerts": "IDS/IPS: Учитывать DDoS/flood-сигнатуры",
    "detect_web_attack_uri": "Web: Анализировать URI на SQLi/XSS/path traversal",
    "detect_web_probe_by_errors": "Web: Детектировать всплеск HTTP-ошибок",
    "use_ids_web_alerts": "IDS/IPS: Учитывать сигнатуры веб-атак",
    "enable_web_uri_analysis": "Web: Анализировать URI на признаки веб-атак",
    "detect_rare_external_ports": "Сеть: Детектировать соединения на редкие внешние порты",
    "detect_new_external_destinations": "Сеть: Детектировать новые внешние направления",
    "use_proxy_security_categories": "Proxy: Учитывать категории malware/phishing/C2",
    "detect_large_external_upload": "Сеть: Детектировать крупную внешнюю передачу данных",
    "detect_many_external_transfers": "Сеть: Детектировать множественные внешние передачи данных",
    "detect_dns_txt_exfiltration": "Сеть/DNS: Учитывать TXT-запросы как признак вывода данных",
    "detect_dns_tunneling": "Сеть/DNS: Детектировать признаки DNS-туннелирования",
    "detect_dns_domain_burst": "Сеть/DNS: Детектировать всплеск уникальных доменов",
    "detect_dns_txt_abuse": "Сеть/DNS: Детектировать аномальный объем TXT-запросов",
    "detect_dga_like_dns": "Сеть/DNS: Детектировать DGA/NXDOMAIN-поведение",
    "detect_beaconing_periodicity": "Сеть: Детектировать периодические соединения",
    "detect_small_repeated_connections": "Сеть: Детектировать малые повторяющиеся внешние соединения",
    "use_ids_c2_alerts": "IDS/IPS: Учитывать C2/beaconing-сигнатуры",
    "detect_internal_admin_service_access": "Сеть: Детектировать обращения к внутренним админ-сервисам",
    "detect_internal_spread": "Сеть: Детектировать распространение по внутренним узлам",
    "use_ids_lateral_alerts": "IDS/IPS: Учитывать сигнатуры латерального перемещения",
    "apply_network_allowlist": "Сеть: Применять белый список доверенных сервисов",
    "ignore_trusted_cloud_services": "Сеть: Игнорировать доверенные облачные сервисы",

    "show_unknown_anomalies": "Общие: Показывать неизвестные аномалии",
}


def _load_rule_file(path: Path) -> dict:
    if not path.exists():
        return {"_policy_options": {}, "_thresholds": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"_policy_options": {}, "_thresholds": {}}


def _collect_rule_defaults(rule_sets: dict[str, dict]) -> tuple[dict, dict]:
    policy_defaults = {}
    threshold_defaults = {}
    for rules in rule_sets.values():
        threshold_defaults.update(rules.get("_thresholds", {}))
        for key, value in rules.get("_policy_options", {}).items():
            if key not in policy_defaults:
                policy_defaults[key] = bool(value)
    return policy_defaults, threshold_defaults


def _render_policy_group(title: str, policy_defaults: dict, keys: list[str], used: set[str]) -> dict:
    values = {}
    available = [key for key in keys if key in policy_defaults and key not in used]
    if not available:
        return values
    with st.expander(title, expanded=False):
        for key in available:
            used.add(key)
            values[key] = st.checkbox(
                POLICY_LABELS.get(key, key),
                value=bool(policy_defaults[key]),
                key=f"policy_{key}",
            )
    return values


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
    "audit_tampering": "изменение или очистка журналов аудита",
    "account_change": "изменение учетных записей",
    "privilege_change": "изменение привилегий",
    "persistence": "создание службы или закрепление",
    "privileged_logon": "привилегированный вход",
    "suspicious_process": "подозрительный запуск процесса",
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
    "изменение или очистка журналов аудита",
    "изменение учетных записей",
    "изменение привилегий",
    "создание службы или закрепление",
    "привилегированный вход",
    "подозрительный запуск процесса",
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
    "win_logon_failed_count": "Windows: неудачные входы",
    "win_rdp_logon_count": "Windows: RDP-входы",
    "win_network_logon_count": "Windows: сетевые входы",
    "win_smb_access_count": "Windows: SMB-доступ",
    "win_service_install_count": "Windows: созданные службы",
    "win_audit_tampering_count": "Windows: изменения аудита",
    "win_account_change_count": "Windows: изменения учетных записей",
    "win_group_change_count": "Windows: изменения групп",
    "win_suspicious_process_count": "Windows: подозрительные процессы",
    "linux_ssh_failed_count": "Linux: ошибки SSH",
    "linux_firewall_drop_count": "Linux/firewall: блокировки",
    "linux_syn_flood_warning_count": "Linux: предупреждения SYN flood",
    "dns_txt_query_count": "DNS TXT-запросы",
    "long_dns_query_count": "длинные DNS-запросы",
}

CHART_COLORS = ["#60a5fa", "#f97316", "#22c55e", "#e879f9", "#facc15", "#fb7185", "#38bdf8", "#a78bfa", "#34d399", "#f472b6", "#fb923c", "#2dd4bf", "#c084fc", "#f87171", "#84cc16", "#06b6d4", "#eab308", "#8b5cf6", "#10b981", "#ef4444", "#3b82f6", "#d946ef", "#14b8a6", "#f59e0b", "#ec4899"]


def ru_label(value: str) -> str:
    return CLASS_RU.get(str(value), str(value))


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




ASSET_COLUMN_RU = {
    "ip": "IP-адрес",
    "role": "Узел / роль",
    "zone": "Зона",
    "segment": "Сегмент",
    "subnet": "Подсеть",
    "criticality": "Критичность",
    "service": "Сервис",
    "software": "ПО",
    "log_source": "Источник журналов",
    "vulnerability": "Контекст уязвимостей",
}

ANOMALY_COLUMN_RU = {
    "anomaly_id": "Идентификатор",
    "risk_matrix_zone": "Уровень риска",
    "risk_level": "Уровень риска",
    "scenario_ru": "Тип аномалии",
    "time_start": "Начало периода",
    "time_end": "Конец периода",
    "src_ip": "Источник",
    "dst_ip": "Назначение",
    "service": "Сервис",
    "confidence": "Уверенность",
    "anomaly_score": "Показатель аномальности",
}


def _id_number(value) -> int:
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else 10**9


def prepare_anomalies_for_ui(df: pd.DataFrame) -> pd.DataFrame:
    """
    В алгоритме идентификатор может оставаться incident_id для совместимости.
    В интерфейсе показываем нейтральный идентификатор AN_0001, AN_0002 и т.д.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    if "incident_id" in out.columns:
        out["_anomaly_sort_num"] = out["incident_id"].apply(_id_number)
    else:
        out["_anomaly_sort_num"] = range(1, len(out) + 1)
    out = out.sort_values(["_anomaly_sort_num", "time_start"], ascending=[True, True]).reset_index(drop=True)
    out["anomaly_id"] = [f"AN_{i:04d}" for i in range(1, len(out) + 1)]
    return out


def display_anomaly_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "anomaly_id", "risk_matrix_zone", "scenario_ru", "time_start", "time_end",
        "src_ip", "dst_ip", "service", "confidence", "anomaly_score"
    ]
    available = [c for c in cols if c in df.columns]
    table = df[available].copy()
    if "risk_matrix_zone" not in table.columns and "risk_level" in df.columns:
        table["risk_level"] = df["risk_level"]
    return table.rename(columns=ANOMALY_COLUMN_RU)


def display_assets_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in ASSET_COLUMN_RU.items() if k in df.columns})


def build_ui_report_markdown(df: pd.DataFrame) -> str:
    report_df = df.copy()
    if "anomaly_id" in report_df.columns:
        report_df["incident_id"] = report_df["anomaly_id"]
    return build_report_markdown(report_df).replace("Карточки инцидентов", "Карточки аномалий")
def risk_badge(level: str) -> str:
    cls = {
        "Высокий": "risk-high",
        "Средний": "risk-medium",
        "Умеренный": "risk-medium",
        "Низкий": "risk-low",
    }.get(str(level), "")
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

    if anomalies is None:
        anomalies = pd.DataFrame()

    out = anomalies.copy()

    # Даже если аномалий нет или входной файл не распознан, интерфейс не должен падать.
    # Создаем служебные столбцы, которые дальше используются фильтрами и счетчиками.
    if out.empty:
        out["risk_matrix_value"] = pd.Series(dtype="float")
        out["risk_matrix_zone"] = pd.Series(dtype="object")
        return out

    for required_col in ["confidence", "anomaly_score", "risk_score"]:
        if required_col not in out.columns:
            out[required_col] = 0

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
        eval_df["eval_label_raw"] = eval_df["true_label"].replace("", np.nan).fillna("normal").astype(str)
        eval_df["eval_source"] = "истинная разметка из файла"
    else:
        eval_df["eval_label_raw"] = eval_df["weak_label"].astype(str)
        eval_df["eval_source"] = "слабая разметка (демонстрационная оценка)"

    def normalize_eval_label(value: object) -> str:
        text = str(value).strip()
        low = text.lower()

        if not low or low in {"nan", "none", "benign", "normal", "normal activity", "0"}:
            return "normal"

        if "ddos" in low or "dos" in low or "flood" in low or "heartbleed" in low:
            return "flood"

        if "portscan" in low or "scan" in low or "recon" in low:
            return "scan"

        if "brute" in low or "ftp-patator" in low or "ssh-patator" in low:
            return "bruteforce"

        if "web" in low or "sql" in low or "xss" in low or "infiltration" in low:
            return "web_attack"

        if "bot" in low or "malware" in low:
            return "malware_beaconing"

        known_internal = {
            "normal", "scan", "bruteforce", "flood", "web_probe", "web_attack",
            "dns_anomaly", "lateral_movement", "data_exfiltration",
            "malware_beaconing", "suspicious_outbound", "unknown_anomaly",
        }
        if low in known_internal:
            return low

        return "unknown_anomaly"

    eval_df["eval_label"] = eval_df["eval_label_raw"].apply(normalize_eval_label)

    # Бинарная разметка: 0 — нормальная активность, 1 — атака/аномалия.
    # Важно: BENIGN из CIC-IDS2017 теперь корректно превращается в normal, а не в атаку.
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
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    if classifier_info.get("status") != "trained" or not hasattr(classifier_info.get("model"), "feature_importances_"):
        ax.text(0.5, 0.5, "Важность признаков недоступна:\nмодель не обучена", ha="center", va="center")
        ax.set_axis_off()
        return fig

    imp = pd.Series(classifier_info["model"].feature_importances_, index=classifier_info["feature_cols"])
    imp = imp.sort_values().tail(12)

    labels = [FEATURE_RU.get(str(v), str(v)) for v in imp.index]
    colors = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(imp))]

    bars = ax.barh(labels, imp.values, color=colors, edgecolor="none")
    ax.set_title("Вклад признаков модели")
    ax.set_xlabel("Значимость признака")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    max_value = float(imp.max()) if len(imp) else 0.0
    if max_value > 0:
        for bar, value in zip(bars, imp.values):
            ax.text(
                value + max_value * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                fontsize=9,
            )
        ax.set_xlim(0, max_value * 1.18)

    fig.tight_layout()
    return fig


def render_instruction_page():
    st.subheader("Краткая инструкция по работе с прототипом")
    st.markdown(
        """
        **Назначение.** Прототип выполняет импорт журналов событий и сетевой телеметрии, приводит данные к единой схеме, агрегирует записи во временные окна, рассчитывает признаки и выявляет аномалии.

        **Порядок работы:**
        1. Загрузите один или несколько файлов журналов/телеметрии в левой панели.
        2. Выберите период анализа и окно агрегации. Для больших файлов рекомендуется начинать с окна 5 минут.
        3. Настройте пороги чувствительности и политики интерпретации событий.
        4. Нажмите **«Запустить анализ»** внизу панели параметров.
        5. На вкладке **«Аномалии»** просмотрите карточки выявленных аномалий, уровень риска, объяснение и рекомендации.
        6. При необходимости скачайте выбранную карточку, все аномалии, таблицу оценки или сводный отчет.

        **Важно.** Изменение чек-боксов и порогов само по себе не запускает повторную обработку. Новый расчет выполняется только после нажатия кнопки **«Запустить анализ»**.
        """
    )
    st.info(
        "Окно анализа — это временной интервал, за который события объединяются в один набор признаков. "
        "Например, 286 тыс. исходных записей могут превратиться в несколько тысяч окон анализа."
    )


st.title("Прототип интеллектуальной системы обнаружения аномалий")
st.caption("Анализ событий и сетевой телеметрии: импорт разнородных источников, нормализация, признаки, профили угроз, модель, показатель аномальности и карточки аномалий.")

profiles = load_profiles(DEFAULT_PROFILES)
assets = load_assets(DEFAULT_ASSETS)
rule_sets = {
    "windows": _load_rule_file(WINDOWS_RULES),
    "linux": _load_rule_file(LINUX_RULES),
    "network": _load_rule_file(NETWORK_RULES),
}
policy_defaults, rule_threshold_defaults = _collect_rule_defaults(rule_sets)

last_events_all = st.session_state.get("analysis_events_all")

with st.sidebar:
    st.header("Параметры анализа")
    with st.form("analysis_run_form"):
        uploaded = st.file_uploader(
            "Загрузить источники событий и телеметрии",
            type=None,
            accept_multiple_files=True,
            help="Можно загрузить несколько файлов одновременно. Файлы без расширения также принимаются: формат определяется по содержимому.",
        )
        st.caption(
            "Если файлы не выбраны, используется демонстрационный набор. "
            "Базовые форматы: " + ", ".join(SUPPORTED_EXTENSIONS)
        )

        st.subheader("Период анализа")
        use_full_period = st.checkbox(
            "Использовать весь период журнала",
            value=True,
            help="Для нового файла рекомендуется сначала выполнить расчет по всему периоду. После первого запуска можно сузить диапазон времени.",
        )
        selected_range = None
        if not use_full_period and last_events_all is not None and len(last_events_all):
            min_ts = last_events_all["timestamp"].min().to_pydatetime()
            max_ts = last_events_all["timestamp"].max().to_pydatetime()
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
        elif not use_full_period:
            st.info("Диапазон времени станет доступен после первого запуска анализа.")

        st.subheader("Окно агрегации")
        window_minutes = st.selectbox(
            "Период группировки событий",
            [1, 5, 15, 30, 60],
            index=1,
            help="Окно определяет, за какой интервал события объединяются в один набор признаков. Например, сканирование считается не по одной строке, а по активности источника за 5 минут.",
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
            "Латеральное перемещение: межсегментных соединений",
            3, 100, 10, 1,
            help="Срабатывает при нетипичных обращениях между VLAN/сегментами."
        )
        lateral_service_hits = st.slider(
            "Латеральное перемещение: обращений к админским сервисам",
            1, 50, 5, 1,
            help="SMB/RDP/SSH/WinRM между сегментами."
        )
        exfil_bytes = st.slider(
            "Вывод данных: байт наружу",
            500_000, 50_000_000, 5_000_000, 500_000,
            help="Порог большого исходящего объема данных во внешний сегмент."
        )
        beacon_events = st.slider(
            "Командный канал: малых внешних соединений",
            3, 100, 8, 1,
            help="Количество повторяющихся малых внешних соединений на нетипичные порты."
        )

        st.divider()
        st.subheader("Политики интерпретации событий")
        st.caption("Чек-боксы задают политику среды: что считать подозрительным, разрешенным или исключаемым из анализа.")

        used_policy_keys = set()
        policy_values = {}
        policy_values.update(_render_policy_group(
            "Windows Event Log / Sysmon",
            policy_defaults,
            [
                "ignore_single_wrong_password", "strict_unknown_user_control", "detect_success_after_failures",
                "allow_rdp_logons", "allow_smb_rpc_access", "detect_remote_service_creation",
                "exclude_web_ports_from_scan", "analyze_wfp_blocked_connections", "enable_noisy_wfp_packet_mode",
                "detect_web_server_shell", "detect_cmd_network_activity", "monitor_mass_file_read",
                "detect_suspicious_archives", "enable_dns_tunneling_detection", "ignore_internal_dns_domains",
                "enable_beaconing_periodicity", "exclude_known_cloud_services", "enable_flood_detection",
                "show_unknown_anomalies",
            ],
            used_policy_keys,
        ))
        policy_values.update(_render_policy_group(
            "Linux auth.log / auditd / syslog",
            policy_defaults,
            [
                "strict_invalid_user_control", "use_ssh_password_failure_buffer", "allow_outbound_ssh_internal",
                "allow_inbound_ssh_logins", "analyze_linux_firewall_blocks", "detect_shell_network_connections",
                "monitor_linux_mass_file_read", "detect_linux_suspicious_archives", "monitor_linux_file_transfer_logs",
                "enable_linux_dns_tunneling_detection", "monitor_dns_txt_queries", "enable_linux_beaconing_periodicity",
                "ignore_linux_update_repositories", "detect_kernel_syn_flood_warnings", "detect_conntrack_overflow",
                "enable_linux_flood_detection", "ignore_system_daemon_noise", "ignore_os_update_noise",
                "enable_linux_web_uri_analysis", "detect_linux_web_server_shell",
            ],
            used_policy_keys,
        ))
        policy_values.update(_render_policy_group(
            "Сетевая телеметрия / SIEM / NTA / IDS / Proxy",
            policy_defaults,
            [
                "detect_network_auth_bruteforce", "detect_password_spraying", "detect_port_scan", "detect_host_scan",
                "detect_denied_scan", "use_ids_scan_alerts", "detect_network_flood", "detect_distributed_flood",
                "detect_syn_flood", "use_ids_flood_alerts", "detect_web_attack_uri", "detect_web_probe_by_errors",
                "use_ids_web_alerts", "detect_rare_external_ports", "detect_new_external_destinations",
                "use_proxy_security_categories", "detect_large_external_upload", "detect_many_external_transfers",
                "detect_dns_txt_exfiltration", "detect_dns_tunneling", "detect_dns_domain_burst",
                "detect_dns_txt_abuse", "detect_dga_like_dns", "detect_beaconing_periodicity",
                "detect_small_repeated_connections", "use_ids_c2_alerts", "detect_internal_admin_service_access",
                "detect_internal_spread", "use_ids_lateral_alerts", "apply_network_allowlist",
                "ignore_trusted_cloud_services",
            ],
            used_policy_keys,
        ))

        run_analysis = st.form_submit_button("Запустить анализ", type="primary")

    st.caption("Запуск: streamlit run app.py")
    if st.session_state.get("analysis_state"):
        state = st.session_state["analysis_state"]
        st.subheader("Последний расчет")
        st.metric("Записей обработано", len(state["events"]))
        st.metric("Окон анализа", len(state["features"]))

if run_analysis:
    try:
        with st.spinner("Выполняется импорт, нормализация, расчет признаков и выявление аномалий..."):
            perf = {}
            t0 = time.perf_counter()
            raw_events = read_many_logs(uploaded, default_path=DEFAULT_EVENTS)
            events_all = normalize_events(raw_events)
            perf["normalization_sec"] = time.perf_counter() - t0

            if selected_range:
                start_ts, end_ts = selected_range
                events = events_all[(events_all["timestamp"] >= pd.Timestamp(start_ts)) & (events_all["timestamp"] <= pd.Timestamp(end_ts))].copy()
            else:
                events = events_all.copy()

            thresholds = dict(rule_threshold_defaults)
            thresholds.update({
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
            })
            thresholds.update(policy_values)

            t1 = time.perf_counter()
            features = build_features(events, window_minutes=window_minutes, thresholds=thresholds)
            perf["feature_sec"] = time.perf_counter() - t1

            t2 = time.perf_counter()
            classifier_info = train_classifier_from_weak_labels(features)
            perf["train_sec"] = time.perf_counter() - t2

            t3 = time.perf_counter()
            anomalies = detect_incidents(events, features, profiles, assets, classifier_info, thresholds)
            anomalies = add_risk_matrix_zone(anomalies)
            anomalies = prepare_anomalies_for_ui(anomalies)
            perf["detect_sec"] = time.perf_counter() - t3
            perf["total_sec"] = sum(perf.values())

            st.session_state["analysis_events_all"] = events_all
            st.session_state["analysis_state"] = {
                "raw_events": raw_events,
                "events_all": events_all,
                "events": events,
                "features": features,
                "classifier_info": classifier_info,
                "anomalies": anomalies,
                "perf": perf,
                "thresholds": thresholds,
                "anomaly_threshold": anomaly_threshold,
                "min_risk_to_show": min_risk_to_show,
            }
    except Exception as exc:
        st.error(f"Не удалось выполнить анализ: {exc}")
        st.stop()

if not st.session_state.get("analysis_state"):
    render_instruction_page()
    st.stop()

state = st.session_state["analysis_state"]
raw_events = state["raw_events"]
events_all = state["events_all"]
events = state["events"]
features = state["features"]
classifier_info = state["classifier_info"]
anomalies = state["anomalies"]
perf = state["perf"]
thresholds = state["thresholds"]
anomaly_threshold = state["anomaly_threshold"]
min_risk_to_show = state["min_risk_to_show"]

risk_order = {"Низкий": 1, "Умеренный": 2, "Средний": 3, "Высокий": 4}
visible_anomalies = anomalies[
    anomalies["risk_matrix_zone"].map(risk_order).fillna(0) >= risk_order[min_risk_to_show]
].copy()

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

col1.metric("Записей обработано", f"{len(events):,}".replace(",", " "))
col2.metric("Окон анализа", f"{len(features):,}".replace(",", " "))
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
        f"окон анализа — {train_rows}. Это предварительная разметка, сформированная правилами и профилями угроз."
    )
else:
    st.warning(
        "Модель машинного обучения не обучена: недостаточно классов в слабой разметке. "
        "Правила и показатель аномальности продолжают работать."
    )

tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Инструкция", "Обзор", "Окна анализа", "Аномалии", "Узлы", "Разработчик", "Отчет"])

with tab0:
    render_instruction_page()

with tab1:
    st.subheader("Общая картина")
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Выявленные аномалии по типам")
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

    st.markdown("#### Последние обработанные записи")
    st.caption("Показано только краткое превью. Полные нормализованные журналы не выводятся в браузер, так как это аналитический слой над источниками событий.")
    st.dataframe(safe_for_streamlit(events.sort_values("timestamp", ascending=False).head(20)), use_container_width=True)

with tab2:
    st.subheader("Окна анализа и признаки")
    st.caption("Одно окно анализа — это агрегированный временной интервал/источник. По нему рассчитываются признаки для правил, слабой разметки, модели и показателя аномальности.")
    preview_rows = min(len(features), 5000)
    st.dataframe(safe_for_streamlit(features.head(preview_rows)), use_container_width=True, height=520)
    if len(features) > preview_rows:
        st.caption(f"Показаны первые {preview_rows} из {len(features)} окон анализа.")
    st.download_button("Скачать окна анализа и признаки CSV", data=features.to_csv(index=False).encode("utf-8-sig"), file_name="analysis_windows_features.csv", mime="text/csv")

with tab3:
    st.subheader("Карточки выявленных аномалий")
    if visible_anomalies.empty:
        st.info("Аномалии по выбранному уровню риска не найдены.")
    else:
        ctrl_left, ctrl_right = st.columns([2.4, 1])
        risk_filter_options = ["Все уровни", "Низкий", "Умеренный", "Средний", "Высокий"]
        with ctrl_right:
            risk_filter = st.selectbox("Фильтр по уровню риска", risk_filter_options, index=0)

        filtered_anomalies = visible_anomalies.copy()
        if risk_filter != "Все уровни" and "risk_matrix_zone" in filtered_anomalies.columns:
            filtered_anomalies = filtered_anomalies[filtered_anomalies["risk_matrix_zone"] == risk_filter]

        filtered_anomalies = filtered_anomalies.sort_values("_anomaly_sort_num" if "_anomaly_sort_num" in filtered_anomalies.columns else "anomaly_id")
        options = ["Все"] + filtered_anomalies["anomaly_id"].tolist()
        with ctrl_left:
            selected_id = st.selectbox("Режим просмотра", options=options, index=0)

        if filtered_anomalies.empty:
            st.info("Аномалии по выбранному фильтру не найдены.")
        elif selected_id == "Все":
            st.markdown("#### Все выявленные аномалии")
            st.dataframe(safe_for_streamlit(display_anomaly_table(filtered_anomalies)), use_container_width=True, height=360)

            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "Скачать все аномалии CSV",
                    data=display_anomaly_table(filtered_anomalies).to_csv(index=False).encode("utf-8-sig"),
                    file_name="detected_anomalies.csv",
                    mime="text/csv",
                )
            with d2:
                st.download_button(
                    "Скачать отчет по всем аномалиям Markdown",
                    data=build_ui_report_markdown(filtered_anomalies).encode("utf-8"),
                    file_name="detected_anomalies_report.md",
                    mime="text/markdown",
                )
        else:
            row = filtered_anomalies[filtered_anomalies["anomaly_id"] == selected_id].iloc[0]
            st.markdown("---")
            st.markdown(f"### Карточка аномалии: {row['anomaly_id']}")
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Тип аномалии:** {row['scenario_ru']}")
            c2.markdown(f"**Уровень риска:** {risk_badge(row.get('risk_matrix_zone', row.get('risk_level', '')))}", unsafe_allow_html=True)
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

            selected_df = filtered_anomalies[filtered_anomalies["anomaly_id"] == selected_id]
            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "Скачать выбранную карточку Markdown",
                    data=build_ui_report_markdown(selected_df).encode("utf-8"),
                    file_name=f"{selected_id}_anomaly_card.md",
                    mime="text/markdown",
                )
            with d2:
                st.download_button(
                    "Скачать все аномалии CSV",
                    data=display_anomaly_table(filtered_anomalies).to_csv(index=False).encode("utf-8-sig"),
                    file_name="detected_anomalies.csv",
                    mime="text/csv",
                )

with tab4:
    st.subheader("Узлы и топология")
    st.caption("Инвентарь узлов используется для учета роли, зоны, критичности сервиса и контекста уязвимостей. От состава топологии зависит приоритет реагирования.")
    st.dataframe(safe_for_streamlit(display_assets_table(assets)), use_container_width=True)
    if len(anomalies):
        st.markdown("#### Аномалии по узлам назначения")
        asset_risk = anomalies.groupby(["dst_ip", "service", "risk_matrix_zone"]).size().reset_index(name="anomaly_count")
        asset_risk = asset_risk.rename(columns={
            "dst_ip": "Узел назначения",
            "service": "Сервис",
            "risk_matrix_zone": "Уровень риска",
            "anomaly_count": "Количество аномалий",
        })
        st.dataframe(safe_for_streamlit(asset_risk), use_container_width=True)

with tab5:
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
        p2.metric("Записей в секунду", f"{events_per_sec:.1f}")
        p3.metric("Время на запись", f"{ms_per_event:.3f} мс")
        p4.metric("Оценка сетевой пропускной способности", f"{approx_mbps:.2f} Мбит/с")
        st.caption("Показатели рассчитаны для обработки журналов событий, а не для inline-фильтрации пакетов в сетевом разрыве.")

        st.markdown("#### Превью нормализованных записей")
        st.caption("Показаны первые 500 записей. Полная таблица не выводится, чтобы не перегружать браузер.")
        st.dataframe(safe_for_streamlit(events.head(500)), use_container_width=True, height=240)

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
        st.download_button(
            "Скачать таблицу оценки CSV",
            data=dev_table.to_csv(index=False).encode("utf-8-sig"),
            file_name="model_evaluation_table.csv",
            mime="text/csv",
        )
    else:
        st.info("Нет данных или зависимостей для оценки качества.")

with tab6:
    st.subheader("Сводный отчет")
    report_md = build_ui_report_markdown(visible_anomalies)
    st.markdown(report_md)
    st.download_button("Скачать отчет Markdown", data=report_md.encode("utf-8"), file_name="anomaly_report.md", mime="text/markdown")
    st.download_button("Скачать аномалии CSV", data=display_anomaly_table(visible_anomalies).to_csv(index=False).encode("utf-8-sig"), file_name="anomalies.csv", mime="text/csv")
