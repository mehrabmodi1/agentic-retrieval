from __future__ import annotations

import anthropic


def get_llm_client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic()


async def generate_text(client: anthropic.AsyncAnthropic, model: str, prompt: str) -> str:
    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
