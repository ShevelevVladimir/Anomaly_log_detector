
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
    "dst_port": ["dst_port", "destination_port", "port", "dport", "destination port"],
    "proto": ["proto", "protocol", "network_protocol"],
    "outcome": ["outcome", "status", "action", "result", "event.outcome"],
    "user": ["user", "username", "account", "login", "user.name"],
    "domain": ["domain", "query", "dns_query", "hostname"],
    "uri": ["uri", "url", "path", "request", "http_request"],
    "bytes": ["bytes", "bytes_out", "sent_bytes", "size", "total length of fwd packets", "total length of bwd packets", "subflow fwd bytes", "subflow bwd bytes"],
    "duration": ["duration", "elapsed", "response_time", "flow duration"],
    "packets": ["packets", "packet_count", "total packets", "total fwd packets", "total backward packets"],
    "flow_bytes_per_sec": ["flow bytes/s", "bytes/s", "bps", "net_in", "net_out"],
    "flow_packets_per_sec": ["flow packets/s", "packets/s", "pps"],
    "syn_flag_count": ["syn flag count", "syn_count"],
    "rst_flag_count": ["rst flag count", "rst_count"],
    "ack_flag_count": ["ack flag count", "ack_count"],
    "average_packet_size": ["average packet size", "avg packet size", "packet length mean"],
    "rule_id": ["rule_id", "sid", "signature_id", "alert_id"],
    "signature": ["signature", "alert", "rule_name", "message"],
    "method": ["method", "http_method"],
    "status_code": ["status_code", "http_status", "code"],
    "user_agent": ["user_agent", "ua"],
    "true_label": ["true_label", "label", " label", "attack_type", "ground_truth", "class"],
    # Поля для Windows Event Log / Sysmon / Linux auditd / сетевой телеметрии
    "event_id": ["event_id", "win_event_id", "id", "event.code", "eventid", "event_id_int"],
    "process_name": ["process_name", "process", "image", "process.name", "Image", "NewProcessName", "exe", "comm", "application"],
    "parent_process_name": ["parent_process_name", "parent_process", "ParentImage", "CreatorProcessName", "process.parent.name"],
    "command_line": ["command_line", "cmdline", "process.command_line", "CommandLine", "a0", "args"],
    "logon_type": ["LogonType", "logon_type", "logon.type"],
    "sub_status": ["SubStatus", "Substatus", "sub_status"],
    "status": ["Status", "status", "rcode"],
    "share_name": ["ShareName", "share_name"],
    "relative_target_name": ["RelativeTargetName", "relative_target_name"],
    "object_name": ["ObjectName", "object_name", "path", "file_path"],
    "accesses": ["Accesses", "AccessMask", "accesses"],
    "service_name": ["ServiceName", "service_name"],
    "service_file_name": ["ServiceFileName", "ImagePath", "service_file_name"],
    "target_filename": ["TargetFilename", "target_filename", "file_name", "filename"],
    "query_type": ["query_type", "qtype", "dns_type"],
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
    "audit_tampering": "изменение или очистка журналов аудита",
    "account_change": "изменение учетных записей",
    "privilege_change": "изменение привилегий",
    "persistence": "создание службы или закрепление",
    "privileged_logon": "привилегированный вход",
    "suspicious_process": "подозрительный запуск процесса",
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
    "audit_tampering": ("Defense Evasion", "T1070 Indicator Removal"),
    "account_change": ("Persistence / Privilege Escalation", "T1098 Account Manipulation"),
    "privilege_change": ("Privilege Escalation", "T1098 Account Manipulation"),
    "persistence": ("Persistence", "T1543.003 Windows Service"),
    "privileged_logon": ("Privilege Escalation", "T1078 Valid Accounts"),
    "suspicious_process": ("Execution", "T1059 Command and Scripting Interpreter"),
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
    # Внешние выгрузки часто содержат пробелы в заголовках (CIC-IDS2017: " Flow Duration").
    # Сначала очищаем имена столбцов, затем применяем карту синонимов.
    df.columns = [str(c).strip() for c in df.columns]
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


def _is_external_ip(value: str) -> bool:
    """
    Возвращает True только для реального внешнего IP-адреса.

    Важно: технические заглушки вроде flow_target, dataset_source, unknown
    не должны считаться внешними адресами. Иначе flow-датасеты без IP
    ошибочно превращаются в подозрительную исходящую активность.
    """
    try:
        ip = ipaddress.ip_address(str(value).strip())
        return not (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_unspecified
            or ip.is_reserved
        )
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

    if str(row.get("data_kind", "")).lower() == "flow":
        return "net_conn"
    if str(row.get("data_kind", "")).lower() == "metric":
        return "system"
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



def _find_numeric_column(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """Возвращает первую найденную числовую колонку из списка синонимов."""
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for name in candidates:
        col = lower_map.get(name.lower().strip())
        if col is not None:
            return pd.to_numeric(df[col], errors="coerce").fillna(0)
    return pd.Series([0] * len(df), index=df.index, dtype="float64")


def _infer_data_kind(df: pd.DataFrame) -> str:
    """Определяет смысл данных: события, метрики или сетевые потоки.

    Это важнее расширения файла: CSV может быть журналом, NetFlow, Zabbix-метриками
    или размеченным датасетом CIC-IDS2017.
    """
    cols = {str(c).lower().strip() for c in df.columns}
    flow_markers = {
        "flow duration", "flow bytes/s", "flow packets/s", "total fwd packets",
        "total backward packets", "destination port", "syn flag count", "packet length mean",
        "fwd packets/s", "bwd packets/s", "subflow fwd bytes", "subflow bwd bytes",
    }
    metric_markers = {
        "cpu", "cpu_usage", "cpu utilization", "memory", "memory_usage", "ram",
        "disk", "disk_usage", "net_in", "net_out", "network in", "network out",
        "value", "item", "item_name", "host", "zabbix", "trigger", "severity",
    }
    if len(cols & flow_markers) >= 2:
        return "flow"
    if len(cols & metric_markers) >= 2:
        return "metric"
    return "event"

def normalize_events(raw: pd.DataFrame) -> pd.DataFrame:
    df = _standardize_columns(raw)

    for col in [
        "timestamp", "source", "event_type", "src_ip", "dst_ip", "src_port",
        "dst_port", "proto", "outcome", "user", "domain", "uri", "bytes",
        "duration", "rule_id", "signature", "method", "status_code",
        "user_agent", "true_label", "packets", "flow_bytes_per_sec", "flow_packets_per_sec",
        "syn_flag_count", "rst_flag_count", "ack_flag_count", "average_packet_size", "data_kind",
        # Расширенные поля под Windows/Sysmon/процессы
        "event_id", "process_name", "parent_process_name", "command_line", "logon_type",
        "sub_status", "status", "share_name", "relative_target_name", "object_name",
        "accesses", "service_name", "service_file_name", "target_filename", "query_type"
    ]:
        if col not in df.columns:
            df[col] = ""

    inferred_kind = _infer_data_kind(df)
    if "data_kind" not in df.columns or df["data_kind"].astype(str).str.strip().eq("").all():
        df["data_kind"] = inferred_kind
    else:
        df["data_kind"] = df["data_kind"].replace("", inferred_kind).fillna(inferred_kind)

    # Для потоковых/метрических CSV без IP-адресов создаем технический источник.
    # Это не выдуманный сетевой адрес, а группировочный ключ, чтобы алгоритм не схлопывал данные некорректно.
    if df["src_ip"].astype(str).str.strip().eq("").all() and inferred_kind in {"flow", "metric"}:
        df["src_ip"] = "dataset_source"
    if df["dst_ip"].astype(str).str.strip().eq("").all() and inferred_kind == "flow":
        df["dst_ip"] = "flow_target"

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
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)

    # Универсальные сетевые/телеметрические признаки. Для CIC/NetFlow часть колонок
    # приходит уже готовой, для обычных логов они остаются нулевыми.
    df["packets"] = pd.to_numeric(df["packets"], errors="coerce").fillna(0)
    if df["packets"].sum() == 0:
        df["packets"] = (
            _find_numeric_column(df, ["Total Fwd Packets", "Subflow Fwd Packets"]) +
            _find_numeric_column(df, ["Total Backward Packets", "Subflow Bwd Packets"])
        )

    extra_bytes = (
        _find_numeric_column(df, ["Total Length of Fwd Packets", "Subflow Fwd Bytes"]) +
        _find_numeric_column(df, ["Total Length of Bwd Packets", "Subflow Bwd Bytes"])
    )
    df["bytes"] = pd.to_numeric(df["bytes"], errors="coerce").fillna(0)
    df["bytes"] = np.where(df["bytes"] > 0, df["bytes"], extra_bytes)

    df["flow_bytes_per_sec"] = pd.to_numeric(df["flow_bytes_per_sec"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df["flow_packets_per_sec"] = pd.to_numeric(df["flow_packets_per_sec"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    if df["flow_bytes_per_sec"].sum() == 0:
        df["flow_bytes_per_sec"] = _find_numeric_column(df, ["Flow Bytes/s", "bytes/s", "net_in", "net_out"])
    if df["flow_packets_per_sec"].sum() == 0:
        df["flow_packets_per_sec"] = _find_numeric_column(df, ["Flow Packets/s", "packets/s"])

    for col in ["syn_flag_count", "rst_flag_count", "ack_flag_count", "average_packet_size"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if df["syn_flag_count"].sum() == 0:
        df["syn_flag_count"] = _find_numeric_column(df, ["SYN Flag Count"])
    if df["rst_flag_count"].sum() == 0:
        df["rst_flag_count"] = _find_numeric_column(df, ["RST Flag Count"])
    if df["ack_flag_count"].sum() == 0:
        df["ack_flag_count"] = _find_numeric_column(df, ["ACK Flag Count"])
    if df["average_packet_size"].sum() == 0:
        df["average_packet_size"] = _find_numeric_column(df, ["Average Packet Size", "Packet Length Mean"])

    df["outcome"] = df["outcome"].apply(_normalize_outcome)
    df["event_type"] = df.apply(_infer_event_type, axis=1)
    df["proto"] = df["proto"].replace("", "tcp").fillna("tcp").astype(str).str.lower()
    df["service"] = df["dst_port"].map(PORT_SERVICE).fillna(df["dst_port"].apply(lambda x: f"port-{x}" if x else "unknown"))
    df["is_external_dst"] = df["dst_ip"].apply(_is_external_ip)

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



def _label_to_scenario(value: Any) -> str:
    """Преобразует разметку учебных датасетов в сценарий для контрольной проверки.

    В производственной выгрузке этого поля обычно нет, поэтому логика не влияет
    на реальные данные предприятия. В CIC/UNSW/CSE-CIC оно нужно для честной
    экспериментальной части и расчета качества.
    """
    label_text = str(value).strip().lower()
    if not label_text or label_text in {"benign", "normal", "normal activity", "0", "nan"}:
        return "normal"
    if "ddos" in label_text or "dos" in label_text or "flood" in label_text or "heartbleed" in label_text:
        return "flood"
    if "portscan" in label_text or "scan" in label_text or "recon" in label_text:
        return "scan"
    if "brute" in label_text or "ftp-patator" in label_text or "ssh-patator" in label_text:
        return "bruteforce"
    if "web" in label_text or "sql" in label_text or "xss" in label_text or "infiltration" in label_text:
        return "web_attack"
    if "bot" in label_text or "malware" in label_text:
        return "malware_beaconing"
    return "unknown_anomaly"



SUSPICIOUS_PROCESS_PATTERNS = [
    "powershell.exe", "cmd.exe", "wscript.exe", "cscript.exe", "rundll32.exe",
    "regsvr32.exe", "certutil.exe", "bitsadmin.exe", "mshta.exe", "wmic.exe",
    "psexec.exe", "vssadmin.exe", "wevtutil.exe", "bcdedit.exe", "net.exe", "net1.exe",
    "bash", "sh", "dash", "nc", "netcat", "ncat", "socat", "curl", "wget",
    "python", "python3", "perl", "php", "ruby", "lua", "telnet", "ssh", "scp", "rsync",
    "tar", "gzip", "zip", "7z", "rar"
]

WEB_SERVER_PROCESS_PATTERNS = [
    "w3wp.exe", "tomcat.exe", "java.exe", "nginx.exe", "httpd.exe", "apache.exe",
    "php-cgi.exe", "www-data", "apache", "nginx", "tomcat"
]

ARCHIVE_EXTENSIONS = [".zip", ".rar", ".7z", ".tar", ".gz", ".tgz"]
SUSPICIOUS_ARCHIVE_PATHS = ["\\windows\\temp\\", "\\users\\public\\", "\\programdata\\", "\\temp\\", "/tmp/", "/var/tmp/", "/home/", "/srv/", "/var/www/"]
COMMON_OUTBOUND_PORTS = {53, 80, 123, 443, 587, 993, 995}
LATERAL_SERVICE_PORTS = {22, 135, 139, 445, 3389, 5985, 5986}


def _policy_enabled(thresholds: Dict[str, Any], key: str, default: bool = True) -> bool:
    value = thresholds.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "да", "истина"}
    return bool(value)


def _contains_any(text: Any, patterns: list[str] | tuple[str, ...]) -> bool:
    value = str(text or "").lower()
    return any(str(p).lower() in value for p in patterns)


def _series_text(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series([""] * len(df), index=df.index, dtype="object")
    for col in columns:
        if col in df.columns:
            result = result.astype(str) + " " + df[col].fillna("").astype(str)
    return result.str.lower()


def _event_id_series(df: pd.DataFrame) -> pd.Series:
    if "event_id" not in df.columns:
        return pd.Series([0] * len(df), index=df.index, dtype="int64")
    return pd.to_numeric(df["event_id"], errors="coerce").fillna(0).astype(int)


def _int_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([0] * len(df), index=df.index, dtype="int64")
    return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)


def _str_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return df[col].fillna("").astype(str)

def _weak_label(row, thresholds):
    """
    Слабая разметка: предварительное присвоение класса активности
    на основе экспертных условий, журналов ОС, сетевой телеметрии и профилей угроз.
    """
    # 0. Критичные события ОС: очистка аудита, создание служб, изменение учеток/групп.
    if row.get("win_audit_tampering_count", 0) >= 1:
        return "audit_tampering"

    if _policy_enabled(thresholds, "detect_remote_service_creation", True) and row.get("win_service_install_count", 0) >= 1:
        return "persistence"

    if row.get("win_group_change_count", 0) >= 1:
        return "privilege_change"

    if row.get("win_account_change_count", 0) >= 1:
        return "account_change"

    # 1. Подбор учетных данных: Windows, Linux, VPN/proxy/web auth.
    wrong_password_threshold = thresholds.get("brute_fails", 10) if _policy_enabled(thresholds, "ignore_single_wrong_password", True) else 1
    unknown_user_threshold = thresholds.get("unknown_user_fails", 5)
    if _policy_enabled(thresholds, "strict_unknown_user_control", False) or _policy_enabled(thresholds, "strict_invalid_user_control", False):
        unknown_user_threshold = min(unknown_user_threshold, thresholds.get("unknown_user_fails_strict", 1))

    if (
        row.get("auth_fail_count", 0) >= thresholds.get("brute_fails", 10)
        or row.get("win_logon_failed_count", 0) >= thresholds.get("brute_fails", 10)
        or row.get("win_wrong_password_count", 0) >= wrong_password_threshold
        or row.get("win_unknown_user_count", 0) >= unknown_user_threshold
        or row.get("win_account_locked_count", 0) >= 1
        or row.get("linux_ssh_failed_count", 0) >= thresholds.get("brute_fails", 10)
        or row.get("linux_ssh_invalid_user_count", 0) >= thresholds.get("invalid_user_fails", 3)
        or row.get("network_auth_fail_count", 0) >= thresholds.get("brute_fails", 10)
        or (
            row.get("auth_fail_count", 0) >= max(4, thresholds.get("brute_fails", 10) // 2)
            and row.get("unique_users", 0) >= 4
        )
    ):
        return "bruteforce"

    if _policy_enabled(thresholds, "detect_success_after_failures", True) and (
        row.get("win_success_after_failures", 0) >= 1 or row.get("linux_ssh_success_after_failures", 0) >= 1
    ):
        return "bruteforce"

    # 2. Латеральное перемещение: RDP/SMB/RPC/SSH/службы/админские порты.
    if not _policy_enabled(thresholds, "allow_rdp_logons", False) and row.get("win_rdp_logon_count", 0) >= 1:
        return "lateral_movement"

    if not _policy_enabled(thresholds, "allow_smb_rpc_access", True) and (
        row.get("win_network_logon_count", 0) >= 1 or row.get("win_smb_access_count", 0) >= 1
    ):
        return "lateral_movement"

    if (
        row.get("win_smb_access_count", 0) >= thresholds.get("smb_access_events", 15)
        or row.get("win_explicit_credentials_count", 0) >= 1
        or row.get("linux_ssh_outbound_count", 0) >= (999999 if _policy_enabled(thresholds, "allow_outbound_ssh_internal", True) else 1)
        or (
            row.get("cross_segment_count", 0) >= thresholds.get("lateral_connections", 10)
            and row.get("lateral_service_count", 0) >= thresholds.get("lateral_service_hits", 5)
        )
        or row.get("admin_service_connection_count", 0) >= thresholds.get("lateral_service_hits", 5)
        or row.get("unique_internal_dst_ips", 0) >= thresholds.get("lateral_unique_hosts", 10)
    ):
        return "lateral_movement"

    # 3. Привилегированные входы и подозрительные процессы.
    if row.get("win_privileged_logon_count", 0) >= 1:
        return "privileged_logon"

    if row.get("win_suspicious_process_count", 0) >= 1 or row.get("linux_suspicious_process_count", 0) >= 1:
        return "suspicious_process"

    # 4. Подозрительная активность веб-сервиса.
    if (
        row.get("web_attack_pattern_count", 0) >= 1
        or row.get("win_webshell_process_count", 0) >= 1
        or row.get("linux_webshell_process_count", 0) >= 1
        or (
            row.get("http_request_count", 0) >= 15
            and row.get("http_error_rate", 0) >= thresholds.get("web_errors", 0.45)
        )
    ):
        return "web_attack"

    # 5. Сканирование портов/узлов.
    if (
        _policy_enabled(thresholds, "detect_port_scan", True)
        and row.get("unique_dst_ports", 0) >= thresholds.get("scan_ports", 18)
        and row.get("denied_rate", 0) >= 0.15
    ) or (
        _policy_enabled(thresholds, "detect_host_scan", True)
        and row.get("unique_dst_ips", 0) >= thresholds.get("scan_hosts", 20)
    ) or (
        _policy_enabled(thresholds, "analyze_wfp_blocked_connections", True)
        and row.get("win_wfp_blocked_count", 0) >= thresholds.get("wfp_blocked_connections", 20)
    ) or (
        _policy_enabled(thresholds, "analyze_linux_firewall_blocks", True)
        and row.get("linux_firewall_drop_count", 0) >= thresholds.get("linux_firewall_blocked_events", 20)
    ):
        return "scan"

    # 6. DNS-аномалии и возможный DNS tunneling.
    if (
        row.get("dns_query_count", 0) >= thresholds.get("dns_queries", 25)
        and (
            row.get("unique_domains", 0) >= thresholds.get("unique_domains", 20)
            or row.get("suspicious_domain_count", 0) >= 3
        )
    ) or (
        _policy_enabled(thresholds, "enable_dns_tunneling_detection", True)
        and row.get("long_dns_query_count", 0) >= 1
    ) or (
        _policy_enabled(thresholds, "monitor_dns_txt_queries", True)
        and row.get("dns_txt_query_count", 0) >= thresholds.get("dns_txt_queries", 25)
    ):
        return "dns_anomaly"

    # 7. Вывод данных.
    if (
        row.get("external_bytes_sum", 0) >= thresholds.get("exfil_bytes", 5_000_000)
        and row.get("external_connection_count", 0) >= 1
    ) or (
        _policy_enabled(thresholds, "monitor_mass_file_read", True)
        and row.get("win_file_read_count", 0) >= thresholds.get("mass_file_read_events", 50)
    ) or (
        _policy_enabled(thresholds, "monitor_linux_mass_file_read", True)
        and row.get("linux_file_read_count", 0) >= thresholds.get("mass_file_read_events", 50)
    ) or (
        _policy_enabled(thresholds, "detect_suspicious_archives", True)
        and (row.get("win_archive_create_count", 0) >= 1 or row.get("linux_archive_create_count", 0) >= 1)
    ):
        return "data_exfiltration"

    # 8. Beaconing / C2-подобная активность.
    if (
        row.get("small_external_connection_count", 0) >= thresholds.get("beacon_events", 8)
        and row.get("external_connection_count", 0) >= thresholds.get("beacon_events", 8)
    ) or (
        _policy_enabled(thresholds, "enable_beaconing_periodicity", True)
        and row.get("external_connection_count", 0) >= thresholds.get("beacon_min_events", thresholds.get("beacon_events", 8))
        and row.get("repeated_destination_count", 0) >= thresholds.get("beacon_min_events", thresholds.get("beacon_events", 8))
    ):
        return "malware_beaconing"

    # 9. Флуд / DDoS / резкий сетевой всплеск.
    if (_policy_enabled(thresholds, "enable_flood_detection", True) or _policy_enabled(thresholds, "detect_network_flood", True)) and (
        row.get("linux_syn_flood_warning_count", 0) >= 1
        or row.get("linux_conntrack_full_count", 0) >= 1
        or row.get("win_network_event_count", 0) >= thresholds.get("flood_connections", thresholds.get("flood_events", 120))
        or row.get("target_event_count", 0) >= thresholds.get("flood_events", 120)
    ):
        return "flood"

    # 10. Общая подозрительная исходящая активность.
    if (
        row.get("linux_shell_network_count", 0) >= (1 if _policy_enabled(thresholds, "detect_shell_network_connections", True) else 999999)
        or row.get("win_suspicious_process_network_count", 0) >= (1 if _policy_enabled(thresholds, "detect_cmd_network_activity", True) else 999999)
        or row.get("rare_dst_port_count", 0) >= thresholds.get("rare_outbound_connections", 5)
        or (
            row.get("bytes_sum", 0) >= max(5_000_000, row.get("bytes_threshold", 5_000_000))
            and row.get("external_dst_count", 0) >= 1
        )
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
    # Используем валидный внешний IP, а не только стендовый сегмент EXTERNAL.
    # Это важно для реальных firewall/NetFlow/Proxy-выгрузок и не ломает CSV без IP.
    df["is_external_connection"] = df["is_external_dst"].astype(int)

    # Малые повторяющиеся внешние соединения на нетипичные порты — возможный beaconing.
    df["is_small_external_connection"] = (
        (df["is_external_connection"] == 1) &
        (df["bytes"] > 0) &
        (df["bytes"] <= 10000) &
        (~df["dst_port"].isin([53, 80, 443, 123]))
    ).astype(int)

    # Объем данных во внешний сегмент.
    df["external_bytes"] = np.where(df["is_external_connection"] == 1, df["bytes"], 0)

    # Признаки Windows Security / System / Sysmon, Linux auth/syslog/auditd и сетевой телеметрии.
    df["event_id_int"] = _event_id_series(df)
    df["logon_type_int"] = _int_col(df, "logon_type")
    raw_text = _series_text(df, [
        "raw_message", "signature", "uri", "domain", "process_name", "parent_process_name",
        "command_line", "sub_status", "status", "accesses", "object_name", "target_filename",
        "service_name", "service_file_name", "query_type", "outcome", "source"
    ])
    process_text = _series_text(df, ["process_name", "command_line"])
    parent_text = _series_text(df, ["parent_process_name"])
    domain_len = _str_col(df, "domain").str.len()

    # Windows-аутентификация и действия на узле.
    df["win_logon_failed"] = df["event_id_int"].isin([4625, 4771, 4776]).astype(int)
    df["win_wrong_password"] = ((df["event_id_int"].eq(4625)) & raw_text.str.contains("0xc000006a", na=False)).astype(int)
    df["win_unknown_user"] = ((df["event_id_int"].eq(4625)) & raw_text.str.contains("0xc0000064|unknown user|invalid user", regex=True, na=False)).astype(int)
    df["win_account_locked"] = df["event_id_int"].isin([4740]).astype(int)
    df["win_logon_success"] = df["event_id_int"].isin([4624]).astype(int)
    df["win_rdp_logon"] = ((df["event_id_int"].eq(4624)) & (df["logon_type_int"].eq(10))).astype(int)
    df["win_network_logon"] = ((df["event_id_int"].eq(4624)) & (df["logon_type_int"].eq(3))).astype(int)
    df["win_explicit_credentials"] = df["event_id_int"].isin([4648]).astype(int)
    df["win_privileged_logon"] = df["event_id_int"].isin([4672]).astype(int)
    df["win_smb_access"] = df["event_id_int"].isin([5140, 5145]).astype(int)
    df["win_file_read"] = ((df["event_id_int"].isin([4663, 5145])) & raw_text.str.contains("readdata|read data|%%4416", regex=True, na=False)).astype(int)
    df["win_service_install"] = df["event_id_int"].isin([4697, 7045]).astype(int)
    df["win_audit_tampering"] = df["event_id_int"].isin([1102, 4616, 4719]).astype(int)
    df["win_account_change"] = df["event_id_int"].isin([4720, 4722, 4723, 4724, 4725, 4726, 4738]).astype(int)
    df["win_group_change"] = df["event_id_int"].isin([4728, 4729, 4732, 4733, 4756, 4757]).astype(int)
    df["win_wfp_allowed"] = df["event_id_int"].isin([5156]).astype(int)
    df["win_wfp_blocked"] = df["event_id_int"].isin([5157, 5152]).astype(int)
    is_sysmon = _str_col(df, "source").str.lower().str.contains("sysmon", na=False) | raw_text.str.contains("sysmon", na=False)
    df["sysmon_network_event"] = ((is_sysmon) & (df["event_id_int"].eq(3))).astype(int)
    df["sysmon_dns_event"] = ((is_sysmon) & (df["event_id_int"].eq(22))).astype(int)
    df["win_network_event"] = (df["event_id_int"].isin([5156, 5157, 5152]).astype(int) | df["sysmon_network_event"]).astype(int)
    df["win_suspicious_process"] = process_text.apply(lambda x: int(any(p in x for p in SUSPICIOUS_PROCESS_PATTERNS)))
    df["win_webshell_process"] = ((parent_text.apply(lambda x: any(p in x for p in WEB_SERVER_PROCESS_PATTERNS))) & (df["win_suspicious_process"].eq(1))).astype(int)
    df["win_suspicious_process_network"] = ((df["win_suspicious_process"].eq(1)) & (df["is_external_dst"].astype(int).eq(1)) & (df["event_id_int"].isin([3, 5156]))).astype(int)
    df["win_archive_create"] = raw_text.apply(lambda x: int(any(ext in x for ext in ARCHIVE_EXTENSIONS) and any(path in x for path in SUSPICIOUS_ARCHIVE_PATHS)))

    # Linux auth.log / secure / syslog / auditd.
    df["linux_ssh_failed"] = raw_text.str.contains("failed password|authentication failure", regex=True, na=False).astype(int)
    df["linux_ssh_invalid_user"] = raw_text.str.contains("invalid user|failed password for invalid user", regex=True, na=False).astype(int)
    df["linux_ssh_accepted"] = raw_text.str.contains("accepted password|accepted publickey", regex=True, na=False).astype(int)
    df["linux_firewall_drop"] = raw_text.str.contains("iptables drop|iptables reject|ufw block| drop | reject |dpt=", regex=True, na=False).astype(int)
    df["linux_syn_flood_warning"] = raw_text.str.contains("possible syn flooding|sending cookies|syn flood", regex=True, na=False).astype(int)
    df["linux_conntrack_full"] = raw_text.str.contains("nf_conntrack.*table full", regex=True, na=False).astype(int)
    df["linux_ssh_outbound"] = raw_text.str.contains(r"execve.*(?:/usr/bin/ssh|/bin/ssh|/usr/bin/scp|/bin/scp|rsync)", regex=True, na=False).astype(int)
    df["linux_shell_network"] = raw_text.str.contains(r"connect.*(?:/bin/bash|/usr/bin/bash|/bin/sh|/usr/bin/sh|/bin/dash)|(?:/bin/bash|/usr/bin/bash|/bin/sh|/usr/bin/sh|/bin/dash).*connect", regex=True, na=False).astype(int)
    df["linux_suspicious_process"] = raw_text.apply(lambda x: int(any(p in x for p in SUSPICIOUS_PROCESS_PATTERNS)))
    df["linux_webshell_process"] = (raw_text.apply(lambda x: any(p in x for p in WEB_SERVER_PROCESS_PATTERNS)) & df["linux_suspicious_process"].eq(1)).astype(int)
    df["linux_file_read"] = raw_text.str.contains(r"syscall=(?:open|openat)|type=syscall.*open|ok download|download", regex=True, na=False).astype(int)
    df["linux_archive_create"] = raw_text.apply(lambda x: int(any(ext in x for ext in ARCHIVE_EXTENSIONS) or any(f" {tool}" in x for tool in [" tar", " gzip", " zip", " 7z", " rar"])))
    df["dns_txt_query"] = (raw_text.str.contains(" txt ", regex=False, na=False) | _str_col(df, "query_type").str.upper().eq("TXT")).astype(int)
    df["long_dns_query"] = ((domain_len >= float(thresholds.get("dns_query_length", 60))) | raw_text.str.contains(r"[a-z0-9]{50,}\.", regex=True, na=False)).astype(int)
    df["is_dns"] = ((df["is_dns"].astype(int) == 1) | (df["sysmon_dns_event"].astype(int) == 1)).astype(int)

    # Универсальные сетевые признаки, если события пришли из firewall/NTA/NDR/SIEM/PCAP.
    excluded_ports = {80, 443} if _policy_enabled(thresholds, "exclude_web_ports_from_scan", True) else set()
    df["scan_dst_port_for_count"] = np.where(df["dst_port"].isin(list(excluded_ports)), 0, df["dst_port"])
    df["rare_outbound_port"] = ((df["is_external_dst"].astype(int).eq(1)) & (~df["dst_port"].isin(COMMON_OUTBOUND_PORTS)) & (df["dst_port"] > 0)).astype(int)
    df["admin_service_connection"] = ((df["dst_port"].isin(LATERAL_SERVICE_PORTS)) & (df["dst_ip"].apply(_is_private_ip))).astype(int)
    df["network_auth_failure"] = raw_text.str.contains("authentication failed|login failed|auth failed|denied|failure", regex=True, na=False).astype(int)
    df["repeated_destination_key"] = df["dst_ip"].astype(str) + ":" + df["dst_port"].astype(str)
    
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
        packets_sum=("packets", "sum"),
        flow_bytes_per_sec_max=("flow_bytes_per_sec", "max"),
        flow_packets_per_sec_max=("flow_packets_per_sec", "max"),
        flow_bytes_per_sec_mean=("flow_bytes_per_sec", "mean"),
        flow_packets_per_sec_mean=("flow_packets_per_sec", "mean"),
        syn_flag_sum=("syn_flag_count", "sum"),
        rst_flag_sum=("rst_flag_count", "sum"),
        ack_flag_sum=("ack_flag_count", "sum"),
        average_packet_size_mean=("average_packet_size", "mean"),
        flow_count=("data_kind", lambda x: (x.astype(str).str.lower() == "flow").sum()),
        metric_count=("data_kind", lambda x: (x.astype(str).str.lower() == "metric").sum()),
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
        # Windows / Linux / network policy features
        win_logon_failed_count=("win_logon_failed", "sum"),
        win_wrong_password_count=("win_wrong_password", "sum"),
        win_unknown_user_count=("win_unknown_user", "sum"),
        win_account_locked_count=("win_account_locked", "sum"),
        win_logon_success_count=("win_logon_success", "sum"),
        win_rdp_logon_count=("win_rdp_logon", "sum"),
        win_network_logon_count=("win_network_logon", "sum"),
        win_explicit_credentials_count=("win_explicit_credentials", "sum"),
        win_privileged_logon_count=("win_privileged_logon", "sum"),
        win_smb_access_count=("win_smb_access", "sum"),
        win_file_read_count=("win_file_read", "sum"),
        win_service_install_count=("win_service_install", "sum"),
        win_audit_tampering_count=("win_audit_tampering", "sum"),
        win_account_change_count=("win_account_change", "sum"),
        win_group_change_count=("win_group_change", "sum"),
        win_wfp_allowed_count=("win_wfp_allowed", "sum"),
        win_wfp_blocked_count=("win_wfp_blocked", "sum"),
        win_network_event_count=("win_network_event", "sum"),
        win_suspicious_process_count=("win_suspicious_process", "sum"),
        win_webshell_process_count=("win_webshell_process", "sum"),
        win_suspicious_process_network_count=("win_suspicious_process_network", "sum"),
        win_archive_create_count=("win_archive_create", "sum"),
        linux_ssh_failed_count=("linux_ssh_failed", "sum"),
        linux_ssh_invalid_user_count=("linux_ssh_invalid_user", "sum"),
        linux_ssh_accepted_count=("linux_ssh_accepted", "sum"),
        linux_firewall_drop_count=("linux_firewall_drop", "sum"),
        linux_syn_flood_warning_count=("linux_syn_flood_warning", "sum"),
        linux_conntrack_full_count=("linux_conntrack_full", "sum"),
        linux_ssh_outbound_count=("linux_ssh_outbound", "sum"),
        linux_shell_network_count=("linux_shell_network", "sum"),
        linux_suspicious_process_count=("linux_suspicious_process", "sum"),
        linux_webshell_process_count=("linux_webshell_process", "sum"),
        linux_file_read_count=("linux_file_read", "sum"),
        linux_archive_create_count=("linux_archive_create", "sum"),
        dns_txt_query_count=("dns_txt_query", "sum"),
        long_dns_query_count=("long_dns_query", "sum"),
        rare_dst_port_count=("rare_outbound_port", "sum"),
        admin_service_connection_count=("admin_service_connection", "sum"),
        network_auth_fail_count=("network_auth_failure", "sum"),
        unique_internal_dst_ips=("dst_ip", lambda x: x[x.apply(_is_private_ip)].nunique()),
        repeated_destination_count=("repeated_destination_key", lambda x: int(x.value_counts().max()) if len(x) else 0),
        top_event_ids=("event_id_int", lambda x: ", ".join(map(str, [v for v in x.value_counts().head(5).index.tolist() if int(v) != 0]))),
        top_dst_ip=("dst_ip", lambda x: x.mode().iloc[0] if len(x.mode()) else ""),
        top_dst_port=("dst_port", lambda x: int(x.mode().iloc[0]) if len(x.mode()) else 0),
        top_service=("service", lambda x: x.mode().iloc[0] if len(x.mode()) else "unknown"),
        uri_examples=("uri", lambda x: ", ".join([v for v in x.astype(str).unique()[:3] if v])),
    ).reset_index()

    features["denied_rate"] = features["denied_count"] / features["event_count"].clip(lower=1)
    features["win_success_after_failures"] = ((features["win_logon_failed_count"] >= thresholds.get("successful_login_after_failures", 5)) & (features["win_logon_success_count"] > 0)).astype(int)
    features["linux_ssh_success_after_failures"] = ((features["linux_ssh_failed_count"] >= thresholds.get("successful_login_after_failures", 5)) & (features["linux_ssh_accepted_count"] > 0)).astype(int)

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

    # Базовые пороги по самому набору данных: подходят для неизвестных выгрузок
    # предприятия, где заранее не известны ни источник, ни масштаб значений.
    for col in ["event_count", "bytes_sum", "packets_sum", "flow_bytes_per_sec_max", "flow_packets_per_sec_max"]:
        q95 = float(features[col].quantile(0.95)) if len(features) else 0.0
        q99 = float(features[col].quantile(0.99)) if len(features) else 0.0
        features[f"{col}_q95"] = q95
        features[f"{col}_q99"] = q99

    features["weak_label"] = features.apply(lambda r: _weak_label(r, thresholds), axis=1)

    # Универсальная эвристика для flow/NetFlow/CIC/NTA/NDR-таблиц: не привязана к Label,
    # а смотрит на всплески потоков, пакетов, байтов и PPS/BPS.
    flow_flood_mask = (
        (features["flow_count"] > 0) &
        (
            ((features["packets_sum_q99"] > 0) & (features["packets_sum"] > features["packets_sum_q99"])) |
            ((features["flow_packets_per_sec_max_q99"] > 0) & (features["flow_packets_per_sec_max"] > features["flow_packets_per_sec_max_q99"])) |
            ((features["flow_bytes_per_sec_max_q99"] > 0) & (features["flow_bytes_per_sec_max"] > features["flow_bytes_per_sec_max_q99"])) |
            ((features["bytes_sum_q99"] > 0) & (features["bytes_sum"] > features["bytes_sum_q99"]))
        )
    )
    features.loc[flow_flood_mask & features["weak_label"].eq("normal"), "weak_label"] = "flood"

    # В контрольных размеченных датасетах (CIC-IDS2017 и аналоги) Label используется
    # не как производственный источник знания, а как режим валидации: программа показывает,
    # какие строки/окна являются известной атакой, чтобы можно было считать метрики качества.
    if "true_label" in features.columns:
        benchmark_scenario = features["true_label"].apply(_label_to_scenario)
        mask = benchmark_scenario.ne("normal") & features["weak_label"].eq("normal")
        features.loc[mask, "weak_label"] = benchmark_scenario[mask]

    # Метрики мониторинга (например, Zabbix): фиксируем резкие пики как неизвестную/ресурсную аномалию.
    metric_spike_mask = (
        (features["metric_count"] > 0) &
        ((features["event_count"] >= features["event_count_q99"].clip(lower=10)) | (features["bytes_sum"] >= features["bytes_sum_q99"].clip(lower=1)))
    )
    features.loc[metric_spike_mask & features["weak_label"].eq("normal"), "weak_label"] = "unknown_anomaly"

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
        "ids_alert_count", "bytes_sum", "packets_sum", "flow_bytes_per_sec_max", "flow_packets_per_sec_max",
        "flow_bytes_per_sec_mean", "flow_packets_per_sec_mean", "syn_flag_sum", "rst_flag_sum",
        "ack_flag_sum", "average_packet_size_mean", "flow_count", "metric_count", "external_dst_count",
        "dns_query_count", "unique_domains", "suspicious_domain_count",
        "web_attack_pattern_count", "cross_segment_count", "lateral_service_count",
        "external_connection_count", "external_bytes_sum", "small_external_connection_count",
        "win_logon_failed_count",
        "win_wrong_password_count",
        "win_unknown_user_count",
        "win_account_locked_count",
        "win_logon_success_count",
        "win_success_after_failures",
        "win_rdp_logon_count",
        "win_network_logon_count",
        "win_explicit_credentials_count",
        "win_privileged_logon_count",
        "win_smb_access_count",
        "win_file_read_count",
        "win_service_install_count",
        "win_audit_tampering_count",
        "win_account_change_count",
        "win_group_change_count",
        "win_wfp_allowed_count",
        "win_wfp_blocked_count",
        "win_network_event_count",
        "win_suspicious_process_count",
        "win_webshell_process_count",
        "win_suspicious_process_network_count",
        "win_archive_create_count",
        "linux_ssh_failed_count",
        "linux_ssh_invalid_user_count",
        "linux_ssh_accepted_count",
        "linux_ssh_success_after_failures",
        "linux_firewall_drop_count",
        "linux_syn_flood_warning_count",
        "linux_conntrack_full_count",
        "linux_ssh_outbound_count",
        "linux_shell_network_count",
        "linux_suspicious_process_count",
        "linux_webshell_process_count",
        "linux_file_read_count",
        "linux_archive_create_count",
        "dns_txt_query_count",
        "long_dns_query_count",
        "rare_dst_port_count",
        "admin_service_connection_count",
        "network_auth_fail_count",
        "unique_internal_dst_ips",
        "repeated_destination_count",
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
        "packets_sum",
        "flow_bytes_per_sec_max",
        "flow_packets_per_sec_max",
        "flow_bytes_per_sec_mean",
        "flow_packets_per_sec_mean",
        "syn_flag_sum",
        "rst_flag_sum",
        "ack_flag_sum",
        "average_packet_size_mean",
        "flow_count",
        "metric_count",
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
        "win_logon_failed_count",
        "win_wrong_password_count",
        "win_unknown_user_count",
        "win_account_locked_count",
        "win_logon_success_count",
        "win_success_after_failures",
        "win_rdp_logon_count",
        "win_network_logon_count",
        "win_explicit_credentials_count",
        "win_privileged_logon_count",
        "win_smb_access_count",
        "win_file_read_count",
        "win_service_install_count",
        "win_audit_tampering_count",
        "win_account_change_count",
        "win_group_change_count",
        "win_wfp_allowed_count",
        "win_wfp_blocked_count",
        "win_network_event_count",
        "win_suspicious_process_count",
        "win_webshell_process_count",
        "win_suspicious_process_network_count",
        "win_archive_create_count",
        "linux_ssh_failed_count",
        "linux_ssh_invalid_user_count",
        "linux_ssh_accepted_count",
        "linux_ssh_success_after_failures",
        "linux_firewall_drop_count",
        "linux_syn_flood_warning_count",
        "linux_conntrack_full_count",
        "linux_ssh_outbound_count",
        "linux_shell_network_count",
        "linux_suspicious_process_count",
        "linux_webshell_process_count",
        "linux_file_read_count",
        "linux_archive_create_count",
        "dns_txt_query_count",
        "long_dns_query_count",
        "rare_dst_port_count",
        "admin_service_connection_count",
        "network_auth_fail_count",
        "unique_internal_dst_ips",
        "repeated_destination_count",
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
        "audit_tampering": "Проверить очистку журнала, изменение политики аудита или системного времени; сохранить оставшиеся журналы и проверить признаки сокрытия следов.",
        "account_change": "Проверить создание, удаление, включение или изменение учетной записи, инициатора операции и наличие согласованной административной заявки.",
        "privilege_change": "Проверить изменение состава групп, добавление пользователей в привилегированные группы и необходимость немедленного отката.",
        "persistence": "Проверить созданную службу, путь исполняемого файла, учетную запись запуска и возможность закрепления злоумышленника.",
        "privileged_logon": "Проверить привилегированный вход, источник подключения, учетную запись и соответствие административным работам.",
        "suspicious_process": "Проверить командную строку, родительский процесс, пользователя запуска и сетевые соединения узла.",
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

    show_unknown = _policy_enabled(thresholds, "show_unknown_anomalies", True)
    anomaly_mask = df["anomaly_score"] >= thresholds["anomaly_threshold"]
    selected = df[
        (df["weak_label"] != "normal") |
        (df["ml_prediction"] != "normal") |
        (show_unknown & anomaly_mask)
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
        if row.get("packets_sum", 0) > 0:
            explanation.append(f"Пакетов/потоковых единиц за окно: {int(row['packets_sum'])}.")
        if row.get("flow_packets_per_sec_max", 0) > 0:
            explanation.append(f"Пиковая интенсивность потока: {float(row['flow_packets_per_sec_max']):.2f} пак/с.")
        if row.get("top_event_ids", ""):
            explanation.append(f"Ключевые Event ID/коды событий за окно: {row.get('top_event_ids')}.")
        for col, label_text in [
            ("win_logon_failed_count", "Windows: неудачных входов"),
            ("win_rdp_logon_count", "Windows: RDP-входов"),
            ("win_smb_access_count", "Windows: SMB-доступов"),
            ("win_service_install_count", "Windows: созданий служб"),
            ("win_audit_tampering_count", "Windows: изменений/очисток аудита"),
            ("win_suspicious_process_count", "Windows: подозрительных процессов"),
            ("linux_ssh_failed_count", "Linux: неудачных SSH-входов"),
            ("linux_firewall_drop_count", "Linux/firewall: блокировок"),
            ("linux_syn_flood_warning_count", "Linux: предупреждений о SYN flood"),
            ("dns_txt_query_count", "DNS TXT-запросов"),
            ("long_dns_query_count", "длинных DNS-запросов"),
        ]:
            if row.get(col, 0) > 0:
                explanation.append(f"{label_text}: {int(row.get(col, 0))}.")
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
