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
    
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "chat_logs.jsonl"

    # include timestamp in the record if you like
    record = response.model_dump(exclude_none=True)
    record["timestamp"] = datetime.utcnow().isoformat()

    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, indent=None) + "\n")

    content = response.choices[0].message.content
    usage = response.usage.model_dump()
    
    return content, usage
