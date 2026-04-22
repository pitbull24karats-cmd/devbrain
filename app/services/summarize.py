from __future__ import annotations
import asyncio
import httpx
from app.core.config import settings

# Share the same semaphore as embed to serialize all Ollama calls
from app.services.embed import _ollama_sem

_SUMMARIZE_PROMPT = (settings.prompts_dir / "summarize.txt").read_text(encoding="utf-8")
_ABSTRACT_PROMPT = (settings.prompts_dir / "abstract.txt").read_text(encoding="utf-8")


async def _generate(prompt: str) -> str:
    async with _ollama_sem:
        async with httpx.AsyncClient(
            base_url=settings.ollama_base_url, timeout=1800.0
        ) as client:
            resp = await client.post(
                "/api/generate",
                json={"model": settings.llm_model, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            return resp.json()["response"].strip()


async def summarize_content(content: str, content_type: str = "file") -> dict:
    prompt = _SUMMARIZE_PROMPT.format(content_type=content_type, content=content)
    raw = await _generate(prompt)
    return _parse_sections(raw)


async def abstract_summary(summary: str) -> dict:
    prompt = _ABSTRACT_PROMPT.format(summary=summary)
    raw = await _generate(prompt)
    return _parse_sections(raw)


def _parse_sections(text: str) -> dict:
    sections: dict[str, list[str]] = {}
    current = "summary"
    sections[current] = []
    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("## summary") or lower.startswith("**summary"):
            current = "summary"
        elif lower.startswith("## insight") or lower.startswith("**insight"):
            current = "insights"
        elif lower.startswith("## reusable") or lower.startswith("**reusable"):
            current = "reusable"
        elif lower.startswith("## abstract") or lower.startswith("**abstract"):
            current = "abstract"
        else:
            sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}
