# Step 45: SSE Streaming Chat Consumer & Markdown Renderer

## 1. Process & Purpose
Consume the backend `POST /chat` Server-Sent Events stream inside Streamlit, rendering generated tokens dynamically into the chat message container and displaying cover images and show tags upon stream completion.

---

## 2. Frontend SSE Stream Consumer

```python
# src/ui/chat_stream.py
import json
import httpx
import streamlit as st

def stream_chat_response(api_base_url: str, messages: list):
    payload = {"messages": messages}
    full_response = ""
    retrieved_context = []

    with httpx.stream("POST", f"{api_base_url}/chat", json=payload, timeout=60.0) as response:
        for line in response.iter_lines():
            if not line:
                continue
            if line.startswith("event: context"):
                continue
            elif line.startswith("event: token"):
                continue
            elif line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    # Check if it's context JSON
                    if data_str.startswith("[{"):
                        retrieved_context = json.loads(data_str)
                    else:
                        full_response += data_str
                        yield data_str, retrieved_context
                except Exception:
                    full_response += data_str
                    yield data_str, retrieved_context
```

---

## 3. Deliverables & Acceptance Criteria
- [x] Streamlit `st.write_stream` consumer dynamically appending tokens.
- [x] Context payload captured during initial event.
