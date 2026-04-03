from __future__ import annotations

import re
from typing import List
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(url_or_id: str) -> str:
    value = url_or_id.strip()
    if "youtube.com" not in value and "youtu.be" not in value:
        return value

    parsed = urlparse(value)
    if parsed.netloc.endswith("youtu.be"):
        return parsed.path.lstrip("/").split("/")[0]

    if parsed.path == "/watch":
        return parse_qs(parsed.query).get("v", [""])[0]

    match = re.search(r"/(shorts|embed)/([^/?]+)", parsed.path)
    if match:
        return match.group(2)

    return ""


def fetch_transcript(video_id: str, languages: List[str] | None = None) -> List[dict]:
    preferred_languages = languages or ["en"]
    return YouTubeTranscriptApi.get_transcript(video_id, languages=preferred_languages)
