from datetime import datetime
from pathlib import Path
import json
from together import Together
from config import settings

client = Together(api_key=settings.together_api_key)

def chat_completion(messages, **extra):
    """
    Wrapper that:
      * sends a non-stream request (simpler logging)
      * returns content, metadata, and writes raw JSON to disk
    """
    response = client.chat.completions.create(
        model=settings.model_name,
        messages=messages,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        stream=False,
        **extra,
    )
    
    # Persist raw for reproducibility
    ts = datetime.utcnow().isoformat()
    Path(settings.log_dir).mkdir(parents=True, exist_ok=True)
    raw_path = Path(settings.log_dir) / f"{ts}.json"
    raw_path.write_text(response.model_dump_json(indent=2, exclude_none=True))

    content = response.choices[0].message.content
    usage = response.usage.model_dump()
    return content, usage
