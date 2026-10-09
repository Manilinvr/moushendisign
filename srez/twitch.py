"""Данные с Twitch: число фолловеров, аватар, чат VOD.

Используется публичный GraphQL-эндпоинт веб-версии Twitch — тот же, через
который сайт сам грузит эти данные. Ключи и регистрация приложения не нужны.
Если Twitch недоступен, значения можно вписать в проект вручную
(followers = ..., avatar = "...", chat = "...").
"""
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GQL_URL = "https://gql.twitch.tv/gql"
CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"  # публичный client-id сайта twitch.tv
COMMENTS_HASH = "b70a3591ff0f4e0313d126c6a1502d79a1c02baebb288227c582044aa76adf6a"


def _post(payload, timeout=20):
    req = urllib.request.Request(
        GQL_URL, data=json.dumps(payload).encode(),
        headers={"Client-ID": CLIENT_ID, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def user_info(login: str) -> dict:
    q = ("query($login:String!){user(login:$login){login displayName "
         "profileImageURL(width:300) followers{totalCount}}}")
    data = _post({"query": q, "variables": {"login": login}})
    u = (data.get("data") or {}).get("user")
    if not u:
        raise LookupError(f"Канал {login} не найден на Twitch")
    return {
        "login": u["login"],
        "name": u["displayName"],
        "followers": u["followers"]["totalCount"],
        "avatar_url": u["profileImageURL"],
        "fetched": int(time.time()),
    }


def download(url: str, out: Path) -> Path:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        out.write_bytes(r.read())
    return out


def _chat_range(vod_id: str, t0: float, t1: float) -> list[dict]:
    """Сообщения с t0 до t1: Twitch отдаёт чат страницами, начиная с любой секунды."""
    msgs, cursor = [], None
    while True:
        variables = {"videoID": vod_id}
        if cursor:
            variables["cursor"] = cursor
        else:
            variables["contentOffsetSeconds"] = int(t0)
        data = _post({
            "operationName": "VideoCommentsByOffsetOrCursor",
            "variables": variables,
            "extensions": {"persistedQuery": {"version": 1, "sha256Hash": COMMENTS_HASH}},
        })
        if isinstance(data, list):
            data = data[0]
        if data.get("errors"):
            raise RuntimeError(f"Twitch отказал в выдаче чата: {data['errors'][0].get('message')}")
        comments = (((data.get("data") or {}).get("video") or {}).get("comments")) or {}
        edges = comments.get("edges") or []
        for e in edges:
            node = e["node"]
            t = float(node["contentOffsetSeconds"])
            if t >= t1:
                return msgs
            if t >= t0:
                frags = (node.get("message") or {}).get("fragments", [])
                msgs.append({"t": t, "text": "".join(fr.get("text", "") for fr in frags)})
        if not edges or not comments.get("pageInfo", {}).get("hasNextPage"):
            return msgs
        cursor = edges[-1].get("cursor")
        if not cursor:
            return msgs


def download_chat(vod_id: str, out: Path, length: float | None = None, workers: int = 8) -> Path:
    """Весь чат VOD в JSONL: {"t": секунда от начала, "text": сообщение}.

    Если известна длина VOD, чат качается параллельно несколькими кусками.
    """
    if length:
        step = length / workers
        ranges = [(i * step, (i + 1) * step if i < workers - 1 else float("inf")) for i in range(workers)]
    else:
        ranges = [(0.0, float("inf"))]
    with ThreadPoolExecutor(len(ranges)) as ex:
        parts = list(ex.map(lambda r: _chat_range(vod_id, *r), ranges))
    tmp = out.with_suffix(".part")
    with tmp.open("w", encoding="utf-8") as f:
        for part in parts:
            for m in part:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
    tmp.replace(out)
    return out
