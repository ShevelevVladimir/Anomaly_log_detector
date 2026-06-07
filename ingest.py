from __future__ import annotations

"""
Модуль предварительного импорта данных безопасности.

Назначение:
1) принять один или несколько файлов разных форматов;
2) определить формат по расширению и/или содержимому;
3) преобразовать записи к единому промежуточному DataFrame;
4) передать этот DataFrame в normalize_events() основного детектора.

Модуль намеренно не заменяет нормализацию. Он решает более раннюю задачу:
"сырые источники -> единое промежуточное представление".
"""

import csv
import gzip
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

TEXT_EXTENSIONS = {".log", ".txt", ".cef", ".leef", ".w3c", ".out", ".syslog"}
JSON_EXTENSIONS = {".json"}
JSONL_EXTENSIONS = {".jsonl", ".ndjson"}
TABLE_EXTENSIONS = {".csv", ".tsv"}
XML_EXTENSIONS = {".xml"}
WINDOWS_EVENT_EXTENSIONS = {".evtx"}
PCAP_EXTENSIONS = {".pcap", ".pcapng"}
SPREADSHEET_EXTENSIONS = {".xlsx", ".xls"}
ARCHIVE_EXTENSIONS = {".zip", ".gz", ".tgz", ".tar", ".tar.gz"}

SUPPORTED_EXTENSIONS = sorted(
    TEXT_EXTENSIONS
    | JSON_EXTENSIONS
    | JSONL_EXTENSIONS
    | TABLE_EXTENSIONS
    | XML_EXTENSIONS
    | WINDOWS_EVENT_EXTENSIONS
    | PCAP_EXTENSIONS
    | SPREADSHEET_EXTENSIONS
    | ARCHIVE_EXTENSIONS
)

_COMMON_COLUMNS = [
    "timestamp", "source", "event_type", "src_ip", "dst_ip", "src_port", "dst_port",
    "proto", "outcome", "user", "domain", "uri", "bytes", "duration", "rule_id",
    "signature", "method", "status_code", "user_agent", "event_id", "process_name",
    "command_line", "raw_message", "source_file", "parser_format",
]

_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_TS_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?|"
    r"\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})"
)
_KEY_VALUE_RE = re.compile(r"(?P<key>[A-Za-z0-9_.-]+)=(?P<value>\"[^\"]*\"|'[^']*'|\S+)")


def _uploaded_name(obj: Any) -> str:
    return getattr(obj, "name", None) or getattr(obj, "filename", None) or str(obj)


def _uploaded_bytes(obj: Any) -> bytes:
    if isinstance(obj, (str, Path)):
        return Path(obj).read_bytes()
    if isinstance(obj, bytes):
        return obj
    if hasattr(obj, "getvalue"):
        return obj.getvalue()
    if hasattr(obj, "read"):
        pos = None
        try:
            pos = obj.tell()
        except Exception:
            pass
        data = obj.read()
        if pos is not None:
            try:
                obj.seek(pos)
            except Exception:
                pass
        return data
    raise ValueError(f"Неподдерживаемый объект файла: {type(obj)}")


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _full_suffix(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".tar.gz"):
        return ".tar.gz"
    return Path(lower).suffix


def _ensure_common_columns(df: pd.DataFrame, source_file: str, parser_format: str) -> pd.DataFrame:
    if df is None or df.empty:
        df = pd.DataFrame([{}])
    df = df.copy()
    if "source_file" not in df.columns:
        df["source_file"] = source_file
    if "parser_format" not in df.columns:
        df["parser_format"] = parser_format
    if "source" not in df.columns:
        df["source"] = parser_format
    for col in _COMMON_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def _flatten_json_records(obj: Any) -> pd.DataFrame:
    if isinstance(obj, list):
        records = obj
    elif isinstance(obj, dict):
        for key in ("events", "records", "data", "hits"):
            if isinstance(obj.get(key), list):
                records = obj[key]
                break
        else:
            records = [obj]
    else:
        records = [{"raw_message": str(obj)}]
    return pd.json_normalize(records)


def _read_json(data: bytes, name: str) -> pd.DataFrame:
    text = _decode(data).strip()
    obj = json.loads(text)
    return _ensure_common_columns(_flatten_json_records(obj), name, "json")


def _read_jsonl(data: bytes, name: str) -> pd.DataFrame:
    rows = []
    for line in _decode(data).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            rows.append({"raw_message": line})
    return _ensure_common_columns(pd.json_normalize(rows), name, "jsonl")


def _read_csv_like(data: bytes, name: str, sep: str | None = None) -> pd.DataFrame:
    text = _decode(data)
    if sep is None:
        sample = text[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            sep = dialect.delimiter
        except Exception:
            sep = "\t" if "\t" in sample else ","
    df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
    return _ensure_common_columns(df, name, "table")


def _parse_key_values(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for m in _KEY_VALUE_RE.finditer(line):
        val = m.group("value").strip().strip('"\'')
        result[m.group("key").lower()] = val
    return result


def _row_from_text_line(line: str, name: str, parser_format: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "raw_message": line,
        "source_file": name,
        "parser_format": parser_format,
        "source": parser_format,
    }

    ts = _TS_RE.search(line)
    if ts:
        row["timestamp"] = ts.group("ts")

    ips = _IP_RE.findall(line)
    if len(ips) >= 1:
        row["src_ip"] = ips[0]
    if len(ips) >= 2:
        row["dst_ip"] = ips[1]

    kv = _parse_key_values(line)
    aliases = {
        "src_ip": ["src", "src_ip", "source", "source_ip", "client", "client_ip", "saddr"],
        "dst_ip": ["dst", "dst_ip", "destination", "destination_ip", "daddr"],
        "src_port": ["sport", "src_port", "source_port"],
        "dst_port": ["dport", "dst_port", "destination_port", "port"],
        "proto": ["proto", "protocol"],
        "outcome": ["action", "result", "outcome", "status"],
        "user": ["user", "username", "account", "login"],
        "bytes": ["bytes", "len", "size", "sent_bytes"],
        "uri": ["uri", "url", "request", "path"],
        "signature": ["msg", "message", "alert", "signature"],
        "rule_id": ["sid", "rule_id", "id"],
    }
    for target, keys in aliases.items():
        for key in keys:
            if key in kv:
                row[target] = kv[key]
                break

    l = line.lower()
    if any(x in l for x in ("deny", "denied", "drop", "blocked", "reject", "fail", "failure")):
        row.setdefault("outcome", "failure")
    elif any(x in l for x in ("allow", "pass", "accept", "success", "allowed")):
        row.setdefault("outcome", "success")

    if "cef:" in l:
        row["event_type"] = "ids_alert" if "alert" in l else "net_conn"
        row["source"] = "cef"
    elif "leef:" in l:
        row["event_type"] = "ids_alert" if "alert" in l else "net_conn"
        row["source"] = "leef"
    elif any(x in l for x in ("suricata", "snort", "alert", "signature", "sid=")):
        row["event_type"] = "ids_alert"
    elif any(x in l for x in ("login", "auth", "sshd", "failed password")):
        row["event_type"] = "auth"
    elif any(x in l for x in ("get ", "post ", "http/", " status=")):
        row["event_type"] = "http"
    else:
        row["event_type"] = "net_conn"
    return row


def _read_text(data: bytes, name: str, parser_format: str = "text_log") -> pd.DataFrame:
    lines = [line.strip() for line in _decode(data).splitlines() if line.strip()]
    rows = [_row_from_text_line(line, name, parser_format) for line in lines]
    return _ensure_common_columns(pd.DataFrame(rows), name, parser_format)


def _read_w3c(data: bytes, name: str) -> pd.DataFrame:
    text = _decode(data)
    fields = None
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#Fields:"):
            fields = line.replace("#Fields:", "").strip().split()
            continue
        if line.startswith("#"):
            continue
        if fields:
            values = line.split()
            row = dict(zip(fields, values))
            if "date" in row and "time" in row:
                row["timestamp"] = row.get("date", "") + " " + row.get("time", "")
            row["raw_message"] = line
            rows.append(row)
        else:
            rows.append(_row_from_text_line(line, name, "w3c"))
    return _ensure_common_columns(pd.DataFrame(rows), name, "w3c")


def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _read_xml(data: bytes, name: str) -> pd.DataFrame:
    root = ET.fromstring(_decode(data).lstrip())
    rows = []
    # Windows Event XML: <Event><System>...</System><EventData>...</EventData></Event>
    events = root.findall(".//{*}Event") or ([root] if _strip_ns(root.tag).lower() == "event" else [])
    if events:
        for ev in events:
            row: dict[str, Any] = {}
            for el in ev.iter():
                tag = _strip_ns(el.tag)
                text = (el.text or "").strip()
                if tag == "TimeCreated":
                    row["timestamp"] = el.attrib.get("SystemTime", "")
                elif tag == "EventID":
                    row["event_id"] = text
                elif tag == "Provider":
                    row["source"] = el.attrib.get("Name", text)
                elif tag == "Computer":
                    row["host"] = text
                elif tag == "Data":
                    key = el.attrib.get("Name")
                    if key:
                        row[key] = text
                elif text and tag not in row:
                    row[tag] = text
            row["raw_message"] = ET.tostring(ev, encoding="unicode")[:5000]
            row["event_type"] = "system"
            rows.append(row)
    else:
        for item in list(root):
            row = { _strip_ns(child.tag): (child.text or "").strip() for child in item }
            row["raw_message"] = ET.tostring(item, encoding="unicode")[:5000]
            rows.append(row)
    return _ensure_common_columns(pd.DataFrame(rows), name, "xml")


def _xml_text_fast(xml: str, tag: str) -> str:
    m = re.search(fr"<{tag}\b[^>]*>(.*?)</{tag}>", xml, flags=re.S | re.I)
    if not m:
        return ""
    return re.sub(r"<[^>]+>", "", m.group(1)).strip()


def _xml_attr_fast(xml: str, tag: str, attr: str) -> str:
    m = re.search(fr"<{tag}\b[^>]*\s{attr}=['\"]([^'\"]+)['\"]", xml, flags=re.S | re.I)
    return m.group(1).strip() if m else ""


def _evtx_data_field(xml: str, *names: str) -> str:
    for name in names:
        m = re.search(fr"<Data\b[^>]*\sName=['\"]{re.escape(name)}['\"][^>]*>(.*?)</Data>", xml, flags=re.S | re.I)
        if m:
            return re.sub(r"<[^>]+>", "", m.group(1)).strip()
    return ""


def _parse_evtx_xml_record(xml: str, name: str) -> dict[str, Any]:
    """Быстрый разбор одной XML-записи Windows Event Log.

    Здесь намеренно используется легкий regex-разбор ключевых полей, а не полный
    ElementTree для каждой записи: EVTX может содержать десятки тысяч событий,
    и для прототипа важнее стабильно извлечь признаки перед нормализацией.
    """
    event_id = _xml_text_fast(xml, "EventID")
    provider = _xml_attr_fast(xml, "Provider", "Name") or "windows_evtx"
    timestamp = _xml_attr_fast(xml, "TimeCreated", "SystemTime")
    computer = _xml_text_fast(xml, "Computer")

    user = _evtx_data_field(xml, "TargetUserName", "SubjectUserName", "AccountName", "UserName", "User")
    src_ip = _evtx_data_field(xml, "IpAddress", "ClientAddress", "SourceAddress", "SrcIp")
    process_name = _evtx_data_field(xml, "ProcessName", "NewProcessName", "Application")
    command_line = _evtx_data_field(xml, "CommandLine", "ProcessCommandLine")
    uri = _evtx_data_field(xml, "TargetFilename", "ObjectName")
    domain = _evtx_data_field(xml, "WorkstationName", "TargetDomainName", "SubjectDomainName") or computer

    row: dict[str, Any] = {
        "timestamp": timestamp,
        "source_file": name,
        "parser_format": "evtx",
        "source": provider,
        "event_type": "system",
        "event_id": event_id,
        "user": user,
        "src_ip": src_ip,
        "domain": domain,
        "process_name": process_name,
        "command_line": command_line,
        "uri": uri,
        "raw_message": xml[:5000],
    }

    if event_id in {"4624", "4625", "4634", "4648", "4672", "4771", "4776"}:
        row["event_type"] = "auth"
    elif event_id in {"4688", "4689", "7045"}:
        row["event_type"] = "process"
    elif event_id in {"5156", "5157", "5152", "5154"}:
        row["event_type"] = "net_conn"
    elif event_id in {"1102", "4719", "4732", "4738", "4720", "4726"}:
        row["event_type"] = "security_change"

    if event_id in {"4625", "4771", "4776", "5157", "5152"}:
        row["outcome"] = "failure"
    elif event_id in {"4624", "4634", "4688", "5156"}:
        row["outcome"] = "success"

    return row

def _extract_event_xml_fragments(text: str) -> list[str]:
    """wevtutil возвращает последовательность <Event>...</Event> без общего корня."""
    text = re.sub(r"<\?xml[^>]*\?>", "", text, flags=re.I).strip()
    return re.findall(r"<Event\b.*?</Event>", text, flags=re.S | re.I)


def _read_evtx_via_wevtutil(data: bytes, name: str) -> pd.DataFrame | None:
    """Нативный путь для Windows: системный wevtutil часто читает EVTX надежнее,
    чем сторонняя библиотека, особенно при нестандартных UTF-16 строках.
    Возвращает None, если запуск невозможен или событий не получено.
    """
    if not sys.platform.startswith("win"):
        return None

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".evtx") as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        cmd = ["wevtutil", "qe", tmp_path, "/lf:true", "/f:xml"]
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
        raw = proc.stdout or b""
        if not raw:
            return None

        # Вывод wevtutil в разных окружениях может быть UTF-8 или UTF-16LE.
        for enc in ("utf-8", "utf-16-le", "cp1251"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode("utf-8", errors="replace")

        fragments = _extract_event_xml_fragments(text)
        if not fragments:
            return None
        rows = [_parse_evtx_xml_record(xml, name) for xml in fragments]
        return _ensure_common_columns(pd.DataFrame(rows), name, "evtx")
    except Exception:
        return None
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


def _read_evtx(data: bytes, name: str) -> pd.DataFrame:
    # 1) На Windows сначала используем штатный парсер ОС.
    native = _read_evtx_via_wevtutil(data, name)
    if native is not None and not native.empty:
        return native

    # 2) Универсальный fallback через python-evtx.
    try:
        from Evtx.Evtx import Evtx  # type: ignore
    except Exception as exc:
        return _ensure_common_columns(pd.DataFrame([{
            "source": "windows_evtx",
            "event_type": "system",
            "raw_message": f"EVTX-файл обнаружен, но библиотека python-evtx не установлена: {exc}",
        }]), name, "evtx_unparsed")

    rows = []
    skipped = 0
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".evtx") as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        with Evtx(tmp_path) as log:
            for record in log.records():
                try:
                    xml = record.xml()
                    rows.append(_parse_evtx_xml_record(xml, name))
                except Exception:
                    # Не роняем весь импорт из-за одной поврежденной/нестандартной записи.
                    skipped += 1
                    continue
    except Exception as exc:
        if not rows:
            return _ensure_common_columns(pd.DataFrame([{
                "source": "read_error",
                "event_type": "system",
                "raw_message": f"Не удалось прочитать EVTX-файл: {exc}",
            }]), name, "read_error")
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    if not rows:
        rows = [{
            "source": "read_error",
            "event_type": "system",
            "raw_message": "EVTX-файл открыт, но записи не извлечены.",
        }]
    if skipped:
        rows.append({
            "source": "windows_evtx",
            "event_type": "system",
            "raw_message": f"Служебная запись импорта: пропущено поврежденных/нестандартных EVTX-записей: {skipped}",
        })
    return _ensure_common_columns(pd.DataFrame(rows), name, "evtx")

def _read_pcap(data: bytes, name: str) -> pd.DataFrame:
    try:
        from scapy.all import PcapReader, IP, TCP, UDP  # type: ignore
    except Exception as exc:
        return _ensure_common_columns(pd.DataFrame([{
            "source": "pcap",
            "event_type": "net_conn",
            "raw_message": f"PCAP-файл обнаружен, но пакет scapy не установлен: {exc}",
        }]), name, "pcap_unparsed")

    rows = []
    with PcapReader(io.BytesIO(data)) as packets:
        for pkt in packets:
            if IP not in pkt:
                continue
            proto = "ip"
            src_port = ""
            dst_port = ""
            if TCP in pkt:
                proto = "tcp"
                src_port = int(pkt[TCP].sport)
                dst_port = int(pkt[TCP].dport)
            elif UDP in pkt:
                proto = "udp"
                src_port = int(pkt[UDP].sport)
                dst_port = int(pkt[UDP].dport)
            rows.append({
                "timestamp": pd.to_datetime(float(pkt.time), unit="s", errors="coerce"),
                "source": "pcap",
                "event_type": "net_conn",
                "src_ip": pkt[IP].src,
                "dst_ip": pkt[IP].dst,
                "src_port": src_port,
                "dst_port": dst_port,
                "proto": proto,
                "bytes": len(pkt),
                "raw_message": pkt.summary(),
            })
    return _ensure_common_columns(pd.DataFrame(rows), name, "pcap")


def _read_spreadsheet(data: bytes, name: str) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(data))
    return _ensure_common_columns(df, name, "spreadsheet")


def _detect_format(data: bytes, name: str) -> str:
    ext = _full_suffix(name)
    if ext in JSON_EXTENSIONS:
        return "json"
    if ext in JSONL_EXTENSIONS:
        return "jsonl"
    if ext in TABLE_EXTENSIONS:
        return "tsv" if ext == ".tsv" else "csv"
    if ext in TEXT_EXTENSIONS:
        return "w3c" if ext == ".w3c" else "text"
    if ext in XML_EXTENSIONS:
        return "xml"
    if ext in WINDOWS_EVENT_EXTENSIONS:
        return "evtx"
    if ext in PCAP_EXTENSIONS:
        return "pcap"
    if ext in SPREADSHEET_EXTENSIONS:
        return "spreadsheet"
    if ext in ARCHIVE_EXTENSIONS:
        return "archive"

    # Автоопределение для файлов без расширения или с нестандартным именем.
    head = _decode(data[:8192]).lstrip()
    if head.startswith("{") or head.startswith("["):
        return "json"
    first_lines = [x.strip() for x in head.splitlines()[:5] if x.strip()]
    if first_lines and all(x.startswith("{") and x.endswith("}") for x in first_lines):
        return "jsonl"
    if head.startswith("<"):
        return "xml"
    if head.startswith("#Fields:") or "#Fields:" in head[:2048]:
        return "w3c"
    if "\t" in head:
        return "tsv"
    if head.count(";") > 5 or head.count(",") > 5:
        return "csv"
    return "text"


def _read_archive(data: bytes, name: str) -> pd.DataFrame:
    rows = []
    lower = name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for item in zf.infolist():
                if item.is_dir():
                    continue
                rows.append(read_single_log_bytes(zf.read(item), f"{name}/{item.filename}"))
    elif lower.endswith((".tar", ".tar.gz", ".tgz")):
        mode = "r:gz" if lower.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(fileobj=io.BytesIO(data), mode=mode) as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                f = tf.extractfile(member)
                if f:
                    rows.append(read_single_log_bytes(f.read(), f"{name}/{member.name}"))
    elif lower.endswith(".gz"):
        inner_name = name[:-3]
        rows.append(read_single_log_bytes(gzip.decompress(data), inner_name))
    if not rows:
        return _ensure_common_columns(pd.DataFrame(), name, "archive_empty")
    return pd.concat(rows, ignore_index=True, sort=False)


def read_single_log_bytes(data: bytes, name: str) -> pd.DataFrame:
    fmt = _detect_format(data, name)
    if fmt == "archive":
        return _read_archive(data, name)
    if fmt == "json":
        return _read_json(data, name)
    if fmt == "jsonl":
        return _read_jsonl(data, name)
    if fmt == "csv":
        return _read_csv_like(data, name, sep=None)
    if fmt == "tsv":
        return _read_csv_like(data, name, sep="\t")
    if fmt == "w3c":
        return _read_w3c(data, name)
    if fmt == "xml":
        return _read_xml(data, name)
    if fmt == "evtx":
        return _read_evtx(data, name)
    if fmt == "pcap":
        return _read_pcap(data, name)
    if fmt == "spreadsheet":
        return _read_spreadsheet(data, name)
    return _read_text(data, name, parser_format="text_log")


def read_many_logs(uploaded_files: Iterable[Any] | Any | None, default_path: str | Path | None = None) -> pd.DataFrame:
    """
    Читает много сырых файлов и объединяет их в один DataFrame.

    Если uploaded_files пустой, используется default_path. Это сохраняет демонстрационный режим.
    """
    if uploaded_files is None or uploaded_files == []:
        if default_path is None:
            return pd.DataFrame(columns=_COMMON_COLUMNS)
        return read_single_log_bytes(Path(default_path).read_bytes(), str(default_path))

    if not isinstance(uploaded_files, (list, tuple, set)):
        uploaded_files = [uploaded_files]

    frames = []
    errors = []
    for file_obj in uploaded_files:
        name = _uploaded_name(file_obj)
        try:
            frames.append(read_single_log_bytes(_uploaded_bytes(file_obj), name))
        except Exception as exc:
            errors.append({
                "source_file": name,
                "parser_format": "read_error",
                "source": "read_error",
                "event_type": "system",
                "raw_message": f"Не удалось прочитать файл: {exc}",
            })
    if errors:
        frames.append(_ensure_common_columns(pd.DataFrame(errors), "errors", "read_error"))
    if not frames:
        return pd.DataFrame(columns=_COMMON_COLUMNS)
    return pd.concat(frames, ignore_index=True, sort=False)
