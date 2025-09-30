"""Quickstart implementation using Stream Python AI SDK with Deepgram STT."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import webbrowser
from typing import Any, Optional
from urllib.parse import urlencode
from uuid import uuid4

import json
import time
from pathlib import Path

from dotenv import load_dotenv
from getstream import Stream
from getstream.models import UserRequest
from getstream.plugins.deepgram.stt import DeepgramSTT
from getstream.video import rtc
from getstream.video.rtc.track_util import PcmData

LOGGER = logging.getLogger(__name__)


def _ensure_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {var_name}")
    return value


def _build_join_url(base_url: str, call_id: str, api_key: str, user_token: str) -> str:
    base = base_url.rstrip("/")
    query = urlencode({"api_key": api_key, "token": user_token, "skip_lobby": "true"})
    return f"{base}"


def _display_join_url(url: str) -> None:
    LOGGER.info("Launching call UI at %s", url)
    try:
        if not webbrowser.open(url):
            LOGGER.warning("Automatic browser open was declined by the system.")
            print(f"Open this URL manually: {url}")
    except Exception:
        LOGGER.exception("Failed to open browser automatically")
        print(f"Open this URL manually: {url}")


def _user_label(user: Any) -> str:
    if user is None:
        return "remote-user"
    if isinstance(user, dict):
        return user.get("id") or user.get("user_id") or "remote-user"
    for attr in ("id", "user_id"):
        if hasattr(user, attr):
            value = getattr(user, attr)
            if value:
                return str(value)
    if hasattr(user, "user"):
        inner = getattr(user, "user")
        if isinstance(inner, dict):
            return inner.get("id") or inner.get("user_id") or "remote-user"
    return "remote-user"


# ----------  ----------
class JsonlSink:
    """
    <TRANSCRIPT_DIR or transcripts>/<YYYY-MM-DD>-<call_id>.jsonl
    
    """
    def __init__(self, call_id: str, directory: Optional[str] = None) -> None:
        date_str = time.strftime("%Y-%m-%d")
        base_dir = Path(directory or os.getenv("TRANSCRIPT_DIR", "transcripts"))
        self.path = base_dir / f"{date_str}-{call_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _append(path: Path, obj: dict) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    async def emit(self, obj: dict) -> None:
       
        await asyncio.to_thread(self._append, self.path, obj)
# -----------------------------------------------------------------


async def _run_call(client: Stream, *, base_url: str) -> None:
    user_id = f"user-{uuid4()}"
    bot_user_id = f"deepgram-stt-bot-{uuid4()}"
    stt: Optional[DeepgramSTT] = None

    try:
        client.upsert_users(
            UserRequest(id=user_id, name="Deepgram Quickstart User"),
            UserRequest(id=bot_user_id, name="Deepgram STT Bot"),
        )
        user_token = client.create_token(user_id, expiration=3600)

        call_id = 'abc123'
        call = client.video.call("default", call_id)
        call.get_or_create(data={"created_by_id": bot_user_id})

   
        sink = JsonlSink(call_id)

        join_url = _build_join_url(base_url, call_id, client.api_key, user_token)
        _display_join_url(join_url)

        stt = DeepgramSTT(interim_results=True)

        @stt.on("partial_transcript")
        async def handle_partial(text: str, user: Optional[Any], metadata: dict) -> None:
            LOGGER.debug("Partial transcript (%s): %s", _user_label(user), text)
            await sink.emit({
                "type": "partial",
                "ts": time.time(),
                "call_id": call_id,
                "user": _user_label(user),
                "text": text,
                "meta": metadata,
            })

        @stt.on("transcript")
        async def handle_final(text: str, user: Optional[Any], metadata: dict) -> None:
            LOGGER.info("Final transcript (%s): %s", _user_label(user), text)
            await sink.emit({
                "type": "final",
                "ts": time.time(),
                "call_id": call_id,
                "user": _user_label(user),
                "text": text,
                "meta": metadata,
            })

        @stt.on("error")
        async def handle_error(error: Exception) -> None:
            LOGGER.error("Deepgram STT error: %s", error)
            await sink.emit({
                "type": "error",
                "ts": time.time(),
                "call_id": call_id,
                "error": str(error),
            })

        async with await rtc.join(call, bot_user_id) as connection:
            @connection.on("audio")
            async def on_audio(pcm: PcmData, user: Any) -> None:
                await stt.process_audio(pcm, user)

            LOGGER.info("Waiting for audio... Press Ctrl+C to stop the session.")
            try:
                await connection.wait()
            except KeyboardInterrupt:
                LOGGER.info("Interrupted by user, leaving the call.")
    finally:
        if stt is not None:
            try:
                await stt.close()
            except Exception:
                LOGGER.exception("Failed to close Deepgram STT cleanly")
        try:
            client.delete_users([user_id, bot_user_id])
        except Exception:
            LOGGER.warning("Failed to delete temporary Stream users", exc_info=True)


async def async_main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    load_dotenv()
    _ensure_env("STREAM_API_KEY")
    _ensure_env("STREAM_API_SECRET")
    base_url = _ensure_env("EXAMPLE_BASE_URL")
    _ensure_env("DEEPGRAM_API_KEY")

    client = Stream.from_env()

    try:
        await _run_call(client, base_url=base_url)
    except Exception:
        LOGGER.exception("Failed to run Deepgram quickstart session")
        raise


def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(async_main())


if __name__ == "__main__":
    main()
