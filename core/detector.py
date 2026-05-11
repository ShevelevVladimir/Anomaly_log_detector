
from __future__ import annotations

import ipaddress
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


COLUMN_ALIASES = {
    "timestamp": ["timestamp", "time", "date", "@timestamp", "event_time", "datetime"],
    "source": ["source", "log_source", "device", "sensor", "source_type"],
    "event_type": ["event_type", "type", "category", "event.category", "kind"],
    "src_ip": ["src_ip", "source_ip", "client_ip", "src", "ip_src", "remote_addr"],
    "dst_ip": ["dst_ip", "destination_ip", "server_ip", "dst", "ip_dst", "host_ip"],
    "src_port": ["src_port", "source_port", "sport"],
    "dst_port": ["dst_port", "destination_port", "port", "dport"],
    "proto": ["proto", "protocol", "network_protocol"],
    "outcome": ["outcome", "status", "action", "result", "event.outcome"],
    "user": ["user", "username", "account", "login", "user.name"],
    "domain": ["domain", "query", "dns_query", "hostname"],
    "uri": ["uri", "url", "path", "request", "http_request"],
    "bytes": ["bytes", "bytes_out", "sent_bytes", "size"],
    "duration": ["duration", "elapsed", "response_time"],
    "rule_id": ["rule_id", "sid", "signature_id", "alert_id"],
    "signature": ["signature", "alert", "rule_name", "message"],
    "method": ["method", "http_method"],
    "status_code": ["status_code", "http_status", "code"],
    "user_agent": ["user_agent", "ua"],
    "true_label": ["true_label", "label", "attack_type", "ground_truth", "class"],
    # Поля для Windows Event Log / Sysmon / расширенных событий
    "event_id": ["event_id", "win_event_id", "id", "event.code"],
    "process_name": ["process_name", "process", "image", "process.name"],
    "command_line": ["command_line", "cmdline", "process.command_line"],
}

PORT_SERVICE = {
    20: "FTP",
    21: "FTP",
    22: "SSH",
    25: "SMTP",
    53: "DNS",
    80: "HTTP",
    110: "POP3",
    123: "NTP",
    135: "RPC",
    139: "NetBIOS",
    143: "IMAP",
    389: "LDAP",
    443: "HTTPS",
    445: "SMB",
    3389: "RDP",
    5432: "PostgreSQL",
    5900: "VNC",
    8080: "HTTP-alt",
    8443: "HTTPS-alt",
}

SCENARIO_RU = {
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

MITRE_MAP = {
    "scan": ("Reconnaissance", "T1046 Network Service Discovery"),
    "bruteforce": ("Credential Access", "T1110 Brute Force"),
    "flood": ("Impact", "T1499 Endpoint Denial of Service"),
    "web_probe": ("Initial Access", "T1190 Exploit Public-Facing Application"),
    "web_attack": ("Initial Access", "T1190 Exploit Public-Facing Application"),
    "dns_anomaly": ("Command and Control", "T1071.004 DNS"),
    "lateral_movement": ("Lateral Movement", "T1021 Remote Services"),
    "data_exfiltration": ("Exfiltration", "T1041 Exfiltration Over C2 Channel"),
    "malware_beaconing": ("Command and Control", "T1071 Application Layer Protocol"),
    "suspicious_outbound": ("Command and Control / Exfiltration", "T1071 Application Layer Protocol"),
    "unknown_anomaly": ("Discovery", "T1082 System Information Discovery"),
    "normal": ("-", "-"),
}


def load_profiles(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_assets(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["ip", "role", "criticality", "service", "software", "vulnerability"])


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    renames = {}
    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lower_map:
                renames[lower_map[alias.lower()]] = target
                break
    return df.rename(columns=renames)


def _is_private_ip(value: str) -> bool:
    try:
        return ipaddress.ip_address(str(value)).is_private
    except Exception:
        return False


def _safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def _infer_event_type(row) -> str:
    current = str(row.get("event_type", "")).lower()
    if current in {"net_conn", "auth", "dns", "http", "ids_alert", "system"}:
        return current

    source = str(row.get("source", "")).lower()
    uri = str(row.get("uri", ""))
    domain = str(row.get("domain", ""))
    user = str(row.get("user", ""))
    signature = str(row.get("signature", ""))
    status_code = _safe_int(row.get("status_code", 0), 0)

    if "ids" in source or signature or row.get("rule_id", "") not in ["", None, np.nan]:
        return "ids_alert"
    if user and ("auth" in current or "login" in current or "vpn" in source):
        return "auth"
    if domain and ("dns" in source or not uri):
        return "dns"
    if uri or status_code:
        return "http"
    return "net_conn"


def _normalize_outcome(value) -> str:
    text = str(value).strip().lower()
    if text in {"allow", "allowed", "accept", "accepted", "success", "succeeded", "ok", "200", "201", "204"}:
        return "success"
    if text in {"deny", "denied", "drop", "dropped", "blocked", "reject", "rejected", "fail", "failed", "failure"}:
        return "failure"
    if text.startswith("2"):
        return "success"
    if text.startswith(("4", "5")):
        return "failure"
    return text if text else "unknown"


def normalize_events(raw: pd.DataFrame) -> pd.DataFrame:
    df = _standardize_columns(raw)

    for col in [
        "timestamp", "source", "event_type", "src_ip", "dst_ip", "src_port",
        "dst_port", "proto", "outcome", "user", "domain", "uri", "bytes",
        "duration", "rule_id", "signature", "method", "status_code",
        "user_agent", "true_label",
        # Расширенные поля под Windows/Sysmon/процессы
        "event_id", "process_name", "command_line"
    ]:
        if col not in df.columns:
            df[col] = ""

    if df["timestamp"].astype(str).str.strip().eq("").all():
        start = pd.Timestamp.now().floor("min") - pd.Timedelta(minutes=len(df))
        df["timestamp"] = [start + pd.Timedelta(seconds=i * 5) for i in range(len(df))]
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if df["timestamp"].isna().any():
            fallback = pd.Timestamp.now().floor("min")
            df["timestamp"] = df["timestamp"].fillna(fallback)

    df["dst_port"] = df["dst_port"].apply(lambda x: _safe_int(x, 0))
    df["src_port"] = df["src_port"].apply(lambda x: _safe_int(x, 0))
    df["status_code"] = df["status_code"].apply(lambda x: _safe_int(x, 0))
    df["bytes"] = df["bytes"].apply(lambda x: _safe_int(x, 0))
    df["duration"] = df["duration"].apply(lambda x: float(x) if str(x).replace(".", "", 1).isdigit() else 0.0)

    df["outcome"] = df["outcome"].apply(_normalize_outcome)
    df["event_type"] = df.apply(_infer_event_type, axis=1)
    df["proto"] = df["proto"].replace("", "tcp").fillna("tcp").astype(str).str.lower()
    df["service"] = df["dst_port"].map(PORT_SERVICE).fillna(df["dst_port"].apply(lambda x: f"port-{x}" if x else "unknown"))
    df["is_external_dst"] = ~df["dst_ip"].apply(_is_private_ip)

    df = df.drop_duplicates(
        subset=["timestamp", "source", "event_type", "src_ip", "dst_ip", "dst_port", "proto", "outcome", "uri", "user"],
        keep="first",
    )
    return df.sort_values("timestamp").reset_index(drop=True)


def _http_error_rate(series: pd.Series) -> float:
    codes = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    if len(codes) == 0:
        return 0.0
    http = codes[codes > 0]
    if len(http) == 0:
        return 0.0
    return float(((http >= 400) & (http <= 599)).mean())

WEB_ATTACK_PATTERNS = [
    # SQL injection
    "union select",
    "select%20",
    "' or '1'='1",
    "' or 1=1",
    "\" or \"1\"=\"1",
    "information_schema",
    "sleep(",
    "benchmark(",
    "--",

    # XSS
    "<script",
    "%3cscript",
    "javascript:",
    "onerror=",
    "onload=",
    "alert(",
    "document.cookie",

    # path traversal / LFI / sensitive files
    "../",
    "..%2f",
    "%2e%2e%2f",
    "/etc/passwd",
    "boot.ini",
    ".env",
    "config.php",
    "backup.zip",
    "wp-config.php",
]


def _contains_web_attack_pattern(uri: str) -> bool:
    """Проверяет URI на типовые признаки SQLi/XSS/path traversal."""
    text = str(uri).lower()
    return any(pattern in text for pattern in WEB_ATTACK_PATTERNS)


def _is_suspicious_domain(domain: str) -> bool:
    """Базовая эвристика для DNS-аномалий: длинные и нетипичные поддомены."""
    value = str(domain).lower().strip()

    if not value:
        return False

    first_label = value.split(".")[0]
    digit_count = sum(ch.isdigit() for ch in first_label)
    hyphen_count = first_label.count("-")

    if len(first_label) >= 35:
        return True

    if len(first_label) >= 20 and digit_count >= 6:
        return True

    if len(first_label) >= 20 and hyphen_count >= 3:
        return True

    return False


def _segment_of(ip: str) -> str:
    """Определяет сегмент стендовой сети по IP."""
    value = str(ip)

    if value.startswith("172.16.1."):
        return "DMZ"

    if value.startswith("192.168.1."):
        return "VLAN1-SERVER"

    if value.startswith("192.168.10."):
        return "VLAN10-USER"

    if value.startswith("192.168.20."):
        return "VLAN20-ADMIN"

    if value.startswith("10.10.51."):
        return "EXTERNAL"

    return "UNKNOWN"

def _majority_label(values):
    """Возвращает метку атаки для окна: если есть атакующая метка, она важнее normal."""
    vals = [str(v).strip() for v in values if str(v).strip() and str(v).strip().lower() != "nan"]
    if not vals:
        return ""
    non_normal = [v for v in vals if v != "normal"]
    if non_normal:
        return pd.Series(non_normal).mode().iloc[0]
    return "normal"


def _weak_label(row, thresholds):
    """
    Слабая разметка: предварительное присвоение класса активности
    на основе экспертных условий и профилей угроз.
    """
    # 1. Подбор учетных данных.
    if row["auth_fail_count"] >= thresholds.get("brute_fails", 10) or (
        row["auth_fail_count"] >= max(4, thresholds.get("brute_fails", 10) // 2)
        and row["unique_users"] >= 4
    ):
        return "bruteforce"

    # 2. Сканирование портов или узлов.
    if (
        row["unique_dst_ports"] >= thresholds.get("scan_ports", 18)
        or row["unique_dst_ips"] >= thresholds.get("scan_hosts", 20)
    ) and row["denied_rate"] >= 0.35:
        return "scan"

    # 3. Подозрительная активность веб-сервиса:
    # web probing + SQLi/XSS/path traversal признаки.
    if (
        row.get("web_attack_pattern_count", 0) >= 1
        or (
            row["http_request_count"] >= 15
            and row["http_error_rate"] >= thresholds.get("web_errors", 0.45)
        )
    ):
        return "web_attack"

    # 4. DNS-аномалия.
    if (
        row.get("dns_query_count", 0) >= thresholds.get("dns_queries", 25)
        and (
            row.get("unique_domains", 0) >= thresholds.get("unique_domains", 20)
            or row.get("suspicious_domain_count", 0) >= 3
        )
    ):
        return "dns_anomaly"

    # 5. Латеральное перемещение.
    if (
        row.get("cross_segment_count", 0) >= thresholds.get("lateral_connections", 10)
        and row.get("lateral_service_count", 0) >= thresholds.get("lateral_service_hits", 5)
    ):
        return "lateral_movement"

    # 6. Подозрительный вывод данных.
    if (
        row.get("external_bytes_sum", 0) >= thresholds.get("exfil_bytes", 5_000_000)
        and row.get("external_connection_count", 0) >= 1
    ):
        return "data_exfiltration"

    # 7. Beaconing / C2-подобная активность.
    if (
        row.get("small_external_connection_count", 0) >= thresholds.get("beacon_events", 8)
        and row.get("external_connection_count", 0) >= thresholds.get("beacon_events", 8)
    ):
        return "malware_beaconing"

    # 8. Общая подозрительная исходящая активность.
    if (
        row.get("bytes_sum", 0) >= max(5_000_000, row.get("bytes_threshold", 5_000_000))
        and row.get("external_dst_count", 0) >= 1
    ):
        return "suspicious_outbound"

    return "normal"


def build_features(events: pd.DataFrame, window_minutes: int, thresholds: Dict[str, Any]) -> pd.DataFrame:
    df = events.copy()
    df["window_start"] = df["timestamp"].dt.floor(f"{window_minutes}min")
    df["is_failure"] = df["outcome"].isin(["failure", "deny", "denied", "blocked", "dropped"]).astype(int)
    df["is_auth_fail"] = ((df["event_type"] == "auth") & (df["outcome"] == "failure")).astype(int)
    df["is_http"] = (df["event_type"] == "http").astype(int)
    df["is_ids"] = (df["event_type"] == "ids_alert").astype(int)
    
    # Расширенные признаки для дополнительных классов аномалий.
    df["is_dns"] = (df["event_type"] == "dns").astype(int)
    df["is_web_attack_pattern"] = df["uri"].apply(_contains_web_attack_pattern).astype(int)
    df["is_suspicious_domain"] = df["domain"].apply(_is_suspicious_domain).astype(int)

    df["src_segment"] = df["src_ip"].apply(_segment_of)
    df["dst_segment"] = df["dst_ip"].apply(_segment_of)

    # Межсегментная активность внутри корпоративной сети.
    df["is_cross_segment_internal"] = (
        (df["src_segment"] != df["dst_segment"]) &
        (df["src_segment"].isin(["VLAN1-SERVER", "VLAN10-USER", "VLAN20-ADMIN"])) &
        (df["dst_segment"].isin(["VLAN1-SERVER", "VLAN10-USER", "VLAN20-ADMIN"]))
    ).astype(int)

    # Сервисы, часто связанные с латеральным перемещением.
    df["is_lateral_service"] = df["dst_port"].isin([22, 135, 139, 445, 3389, 5985, 5986]).astype(int)

    # Внешние соединения.
    df["is_external_connection"] = df["dst_segment"].eq("EXTERNAL").astype(int)

    # Малые повторяющиеся внешние соединения на нетипичные порты — возможный beaconing.
    df["is_small_external_connection"] = (
        (df["is_external_connection"] == 1) &
        (df["bytes"] > 0) &
        (df["bytes"] <= 10000) &
        (~df["dst_port"].isin([53, 80, 443, 123]))
    ).astype(int)

    # Объем данных во внешний сегмент.
    df["external_bytes"] = np.where(df["is_external_connection"] == 1, df["bytes"], 0)
    
    group_cols = ["window_start", "src_ip"]
    features = df.groupby(group_cols).agg(
        time_start=("timestamp", "min"),
        time_end=("timestamp", "max"),
        event_count=("timestamp", "count"),
        unique_dst_ips=("dst_ip", "nunique"),
        unique_dst_ports=("dst_port", "nunique"),
        denied_count=("is_failure", "sum"),
        auth_fail_count=("is_auth_fail", "sum"),
        unique_users=("user", lambda x: x.replace("", np.nan).dropna().nunique()),
        http_request_count=("is_http", "sum"),
        ids_alert_count=("is_ids", "sum"),
        bytes_sum=("bytes", "sum"),
        external_dst_count=("is_external_dst", "sum"),
        dns_query_count=("is_dns", "sum"),
        unique_domains=("domain", lambda x: x.replace("", np.nan).dropna().nunique()),
        suspicious_domain_count=("is_suspicious_domain", "sum"),
        web_attack_pattern_count=("is_web_attack_pattern", "sum"),
        cross_segment_count=("is_cross_segment_internal", "sum"),
        lateral_service_count=("is_lateral_service", "sum"),
        external_connection_count=("is_external_connection", "sum"),
        external_bytes_sum=("external_bytes", "sum"),
        small_external_connection_count=("is_small_external_connection", "sum"),
        top_dst_ip=("dst_ip", lambda x: x.mode().iloc[0] if len(x.mode()) else ""),
        top_dst_port=("dst_port", lambda x: int(x.mode().iloc[0]) if len(x.mode()) else 0),
        top_service=("service", lambda x: x.mode().iloc[0] if len(x.mode()) else "unknown"),
        uri_examples=("uri", lambda x: ", ".join([v for v in x.astype(str).unique()[:3] if v])),
    ).reset_index()

    features["denied_rate"] = features["denied_count"] / features["event_count"].clip(lower=1)

    if "true_label" in df.columns:
        true_labels = df.groupby(group_cols)["true_label"].apply(_majority_label).reset_index(name="true_label")
        features = features.merge(true_labels, on=group_cols, how="left")
    else:
        features["true_label"] = ""

    http_rates = []
    for _, frow in features.iterrows():
        sub = df[(df["window_start"] == frow["window_start"]) & (df["src_ip"] == frow["src_ip"])]
        http_rates.append(_http_error_rate(sub["status_code"]))
    features["http_error_rate"] = http_rates

    byte_threshold = max(5000000, float(features["bytes_sum"].quantile(0.95)) if len(features) else 5000000)
    features["bytes_threshold"] = byte_threshold

    features["weak_label"] = features.apply(lambda r: _weak_label(r, thresholds), axis=1)

    # Target flood: много событий к одному сервису от одного или многих источников.
    target = df.groupby(["window_start", "dst_ip", "dst_port"]).agg(
        target_event_count=("timestamp", "count"),
        target_unique_src=("src_ip", "nunique"),
    ).reset_index()
    flood_targets = target[
        (target["target_event_count"] >= thresholds["flood_events"]) |
        ((target["target_event_count"] >= thresholds["flood_events"] // 2) & (target["target_unique_src"] >= 8))
    ]
    features["flood_target"] = False
    for _, trow in flood_targets.iterrows():
        mask = (features["window_start"] == trow["window_start"]) & (features["top_dst_ip"] == trow["dst_ip"])
        features.loc[mask, "weak_label"] = "flood"
        features.loc[mask, "flood_target"] = True

    features = _calculate_anomaly_score(features)
    return features


def _calculate_anomaly_score(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    numeric_cols = [
        "event_count", "unique_dst_ips", "unique_dst_ports", "denied_rate",
        "auth_fail_count", "unique_users", "http_request_count", "http_error_rate",
        "ids_alert_count", "bytes_sum", "external_dst_count",
        "dns_query_count", "unique_domains", "suspicious_domain_count",
        "web_attack_pattern_count", "cross_segment_count", "lateral_service_count",
        "external_connection_count", "external_bytes_sum", "small_external_connection_count",
    ]

    if len(features) >= 8 and SKLEARN_AVAILABLE:
        x = features[numeric_cols].fillna(0).astype(float)
        scaler = StandardScaler()
        xs = scaler.fit_transform(x)
        model = IsolationForest(contamination="auto", random_state=42)
        model.fit(xs)
        raw = -model.score_samples(xs)
        mn, mx = raw.min(), raw.max()
        features["anomaly_score"] = (raw - mn) / (mx - mn + 1e-9)
    else:
        scores = []
        for col in numeric_cols:
            s = features[col].fillna(0).astype(float)
            max_val = s.quantile(0.98) if len(s) else 1
            scores.append((s / (max_val + 1e-9)).clip(0, 1))
        features["anomaly_score"] = pd.concat(scores, axis=1).mean(axis=1)

    return features


def train_classifier_from_weak_labels(features: pd.DataFrame) -> Dict[str, Any]:
    feature_cols = [
        "event_count",
        "unique_dst_ips",
        "unique_dst_ports",
        "denied_rate",
        "auth_fail_count",
        "unique_users",
        "http_request_count",
        "http_error_rate",
        "ids_alert_count",
        "bytes_sum",
        "external_dst_count",
        "anomaly_score",
        "dns_query_count",
        "unique_domains",
        "suspicious_domain_count",
        "web_attack_pattern_count",
        "cross_segment_count",
        "lateral_service_count",
        "external_connection_count",
        "external_bytes_sum",
        "small_external_connection_count",
    ]
    labels = features["weak_label"].fillna("normal")
    class_count = labels.nunique()

    if not SKLEARN_AVAILABLE or class_count < 2 or len(features) < 10:
        return {"status": "not_trained", "reason": "Недостаточно данных или недоступен scikit-learn."}

    x = features[feature_cols].fillna(0).astype(float)
    y = labels.astype(str)

    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        random_state=42,
        class_weight="balanced_subsample",
    )
    # Отдельная проверочная выборка нужна, чтобы метрики не считались
    # на тех же данных, на которых модель обучалась.
    can_split = len(x) >= 20 and y.nunique() > 1 and y.value_counts().min() >= 2

    eval_info = {}

    if can_split:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.35,
            random_state=42,
            stratify=y,
        )

        eval_clf = RandomForestClassifier(
            n_estimators=120,
            random_state=42,
            class_weight="balanced",
        )

        eval_clf.fit(x_train, y_train)

        eval_pred = eval_clf.predict(x_test)
        eval_proba = eval_clf.predict_proba(x_test)
        eval_conf = eval_proba.max(axis=1)

        eval_info = {
            "available": True,
            "x_test_index": x_test.index.tolist(),
            "y_test": y_test.tolist(),
            "eval_prediction": eval_pred.tolist(),
            "eval_confidence": eval_conf.tolist(),
            "eval_classes": eval_clf.classes_.tolist(),
            "eval_proba": eval_proba.tolist(),
        }
    else:
        eval_info = {
            "available": False,
            "reason": "Недостаточно данных или классов для отдельной тестовой выборки.",
        }

    # Финальная модель для работы интерфейса обучается на всех доступных данных.
    # Это нужно, чтобы приложение могло классифицировать все загруженные события.
    clf.fit(x, y)

    pred = clf.predict(x)
    proba = clf.predict_proba(x)
    conf = proba.max(axis=1)

    result = features.copy()
    result["ml_prediction"] = pred
    result["ml_confidence"] = conf

    return {
        "status": "trained",
        "model": clf,
        "feature_cols": feature_cols,
        "classes": clf.classes_.tolist(),
        "features": result,
        "eval_info": eval_info,

        # Для отображения в интерфейсе
        "class_count": int(y.nunique()),
        "train_rows": int(len(features)),
    }


def _risk_level(score: float) -> str:
    if score >= 85:
        return "Высокий"
    if score >= 55:
        return "Средний"
    return "Низкий"


def _recommendation(scenario: str) -> str:
    return {
        "scan": "Проверить источник сканирования, сопоставить с разрешенными административными задачами и при необходимости ограничить доступ на периметре.",
        "bruteforce": "Проверить журналы аутентификации, заблокировать источник при подтверждении атаки и усилить контроль учетных записей.",
        "flood": "Проверить нагрузку на сервис, источники запросов, правила фильтрации и состояние доступности сервиса.",
        "web_probe": "Проверить обращения к веб-ресурсу, наличие запросов к административным путям и актуальность обновлений веб-приложения.",
        "web_attack": "Проверить журналы reverse proxy/WAF/IDS, URI-запросы, наличие SQLi/XSS/path traversal признаков и актуальность веб-приложения.",
        "dns_anomaly": "Проверить DNS-запросы узла, длину доменных имен, повторяемость обращений и возможность DNS-туннелирования.",
        "lateral_movement": "Проверить межсегментные подключения, обращения к SMB/RDP/SSH/WinRM, учетную запись источника и допустимость такого направления.",
        "data_exfiltration": "Проверить объем исходящего трафика, внешний адрес назначения, процессы на узле-источнике и наличие признаков вывода данных.",
        "malware_beaconing": "Проверить регулярные внешние соединения, домен/IP назначения, процессы на узле и возможный командный канал.",
        "suspicious_outbound": "Проверить хост-источник, процессы и сетевые соединения, так как активность может указывать на командный канал или вывод данных.",
        "unknown_anomaly": "Проверить цепочку событий вручную, так как активность существенно отличается от локального профиля поведения.",
        "normal": "Дополнительные действия не требуются.",
    }.get(scenario, "Проверить исходные события и контекст актива.")


def _vuln_context(dst_ip, dst_port, assets: pd.DataFrame) -> str:
    if assets.empty:
        return ""
    subset = assets[(assets["ip"].astype(str) == str(dst_ip))]
    if subset.empty:
        return ""
    row = subset.iloc[0]
    vuln = str(row.get("vulnerability", "")).strip()
    software = str(row.get("software", "")).strip()
    if vuln:
        return f"Актив {dst_ip} ({software}) связан с уязвимостью/записью: {vuln}."
    return ""


def _asset_criticality(dst_ip, assets: pd.DataFrame) -> str:
    if assets.empty:
        return "Средняя"
    subset = assets[assets["ip"].astype(str) == str(dst_ip)]
    if subset.empty:
        return "Средняя"
    return str(subset.iloc[0].get("criticality", "Средняя"))


def detect_incidents(
    events: pd.DataFrame,
    features: pd.DataFrame,
    profiles: Dict[str, Any],
    assets: pd.DataFrame,
    classifier_info: Dict[str, Any],
    thresholds: Dict[str, Any],
) -> pd.DataFrame:
    df = features.copy()

    feature_cols = classifier_info.get("feature_cols", [])
    if classifier_info.get("status") == "trained":
        clf = classifier_info["model"]
        x = df[feature_cols].fillna(0).astype(float)
        proba = clf.predict_proba(x)
        pred = clf.predict(x)
        df["ml_prediction"] = pred
        df["ml_confidence"] = proba.max(axis=1)
    else:
        df["ml_prediction"] = df["weak_label"]
        df["ml_confidence"] = np.where(df["weak_label"] == "normal", 0.35, 0.70)

    selected = df[
        (df["weak_label"] != "normal") |
        (df["ml_prediction"] != "normal") |
        (df["anomaly_score"] >= thresholds["anomaly_threshold"])
    ].copy()

    rows = []
    for i, row in selected.reset_index(drop=True).iterrows():
        label = row["weak_label"]
        if label == "normal" and row["ml_prediction"] != "normal":
            label = row["ml_prediction"]
        if label == "normal":
            label = "unknown_anomaly"

        confidence = float(max(row.get("ml_confidence", 0.0), row.get("anomaly_score", 0.0)))

        risk_score = 25
        if label != "unknown_anomaly":
            risk_score += 25
        risk_score += float(row["anomaly_score"]) * 25
        risk_score += float(row.get("ml_confidence", 0.0)) * 15

        criticality = _asset_criticality(row["top_dst_ip"], assets)
        if criticality.lower().startswith("выс"):
            risk_score += 15
        elif criticality.lower().startswith("сред"):
            risk_score += 7

        vuln = _vuln_context(row["top_dst_ip"], row["top_dst_port"], assets)
        if vuln:
            risk_score += 15

        risk_score = min(100, risk_score)
        risk_level = _risk_level(risk_score)

        explanation = []
        if row["weak_label"] != "normal":
            explanation.append(f"Сработал профиль угрозы: {SCENARIO_RU.get(row['weak_label'], row['weak_label'])}.")
        if row["ml_prediction"] != "normal":
            explanation.append(f"Модель отнесла активность к классу: {SCENARIO_RU.get(row['ml_prediction'], row['ml_prediction'])}.")
        if row["anomaly_score"] >= thresholds["anomaly_threshold"]:
            explanation.append(f"Показатель аномальности превышает порог: {row['anomaly_score']:.2f}.")
        if row["unique_dst_ports"] > 1:
            explanation.append(f"Уникальных портов назначения: {int(row['unique_dst_ports'])}.")
        if row["unique_dst_ips"] > 1:
            explanation.append(f"Уникальных адресов назначения: {int(row['unique_dst_ips'])}.")
        if row["denied_rate"] > 0:
            explanation.append(f"Доля отказов/блокировок: {row['denied_rate']:.2f}.")
        if row["auth_fail_count"] > 0:
            explanation.append(f"Неудачных попыток входа: {int(row['auth_fail_count'])}.")
        if row["http_error_rate"] > 0:
            explanation.append(f"Доля HTTP-ошибок: {row['http_error_rate']:.2f}.")
        if row["bytes_sum"] > 0:
            explanation.append(f"Объем переданных данных: {int(row['bytes_sum'])} байт.")
        if vuln:
            explanation.append(vuln)

        tactic, technique = MITRE_MAP.get(label, MITRE_MAP["unknown_anomaly"])

        rows.append({
            "incident_id": f"INC-{i+1:04d}",
            "scenario": label,
            "scenario_ru": SCENARIO_RU.get(label, label),
            "risk_score": round(risk_score, 1),
            "risk_level": risk_level,
            "confidence": round(confidence, 3),
            "anomaly_score": round(float(row["anomaly_score"]), 3),
            "src_ip": row["src_ip"],
            "dst_ip": row["top_dst_ip"],
            "service": row["top_service"],
            "dst_port": int(row["top_dst_port"]),
            "time_start": str(row["time_start"]),
            "time_end": str(row["time_end"]),
            "weak_label": row["weak_label"],
            "ml_prediction": row["ml_prediction"],
            "ml_confidence": round(float(row["ml_confidence"]), 3),
            "mitre_tactic": tactic,
            "mitre_technique": technique,
            "vulnerability_context": vuln,
            "recommendation": _recommendation(label),
            "explanation_json": json.dumps(explanation, ensure_ascii=False),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "incident_id", "scenario", "scenario_ru", "risk_score", "risk_level",
            "confidence", "anomaly_score", "src_ip", "dst_ip", "service",
            "dst_port", "time_start", "time_end", "weak_label", "ml_prediction",
            "ml_confidence", "mitre_tactic", "mitre_technique", "vulnerability_context",
            "recommendation", "explanation_json"
        ])

    return pd.DataFrame(rows).sort_values(["risk_score", "confidence"], ascending=False).reset_index(drop=True)


def build_report_markdown(incidents: pd.DataFrame) -> str:
    lines = [
        "# Отчет по результатам обнаружения аномалий",
        "",
        f"Всего обнаруженных аномалий: **{len(incidents)}**",
        "",
    ]
    if incidents.empty:
        lines.append("Аномалии по выбранным условиям не обнаружены.")
        return "\n".join(lines)

    for _, row in incidents.iterrows():
        lines.extend([
            f"## {row['incident_id']} — {row['scenario_ru']}",
            f"- Уровень риска: **{row['risk_level']}**",
            f"- Оценка уверенности: **{row['confidence']:.2f}**",
            f"- Показатель аномальности: **{row['anomaly_score']:.2f}**",
            f"- Период: {row['time_start']} — {row['time_end']}",
            f"- Источник: `{row['src_ip']}`",
            f"- Назначение: `{row['dst_ip']}`",
            f"- Сервис: `{row['service']}`",
            f"- MITRE ATT&CK: {row['mitre_tactic']} / {row['mitre_technique']}",
            "",
            "Объяснение:",
        ])
        for item in json.loads(row["explanation_json"]):
            lines.append(f"- {item}")
        lines.extend([
            "",
            f"Рекомендация: {row['recommendation']}",
            "",
        ])
        if row.get("vulnerability_context"):
            lines.extend([f"Контекст уязвимостей: {row['vulnerability_context']}", ""])

    return "\n".join(lines)
