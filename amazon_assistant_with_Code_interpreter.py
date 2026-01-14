from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from code_interpreter_system_prompt import SYSTEM_PROMPT, ASSISTANT_PROMPT

# Load environment variables early.
load_dotenv()

MODEL_DEFAULT = os.getenv("OPENAI_ASSISTANT_MODEL", "gpt-4o-mini")



@st.cache_resource
def get_openai_client() -> OpenAI:
    """
    Create a cached OpenAI client.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        st.error("OPENAI_API_KEY is not set. Please add it to your .env file.")
        st.stop()
    return OpenAI(api_key=api_key) 


def ensure_state() -> None:
    """Initialize Streamlit session state keys."""
    defaults = {
        "messages": [],
        "thread_id": None,
        "assistant_id": None,
        "assistant_prompt": ASSISTANT_PROMPT,
        "uploaded_files": {},  # name -> {id, size}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def create_or_update_assistant(
    client: OpenAI, instructions: str, model: str
) -> str:
    """
    Create a new assistant when instructions change, otherwise reuse.
    If the prompt changes, the conversation thread is reset to avoid stale context.
    """
    prompt_changed = instructions.strip() != (st.session_state.get("assistant_prompt") or "").strip()
    if st.session_state.get("assistant_id") is None or prompt_changed:
        assistant = client.beta.assistants.create(
            name="Analysis Copilot",
            instructions=instructions.strip(),
            model=model,
            tools=[{"type": "code_interpreter"}],
        )
        st.session_state.assistant_id = assistant.id
        st.session_state.assistant_prompt = instructions.strip()
        # reset thread when prompt changes so context matches the new brief
        st.session_state.thread_id = None
    return st.session_state.assistant_id


def ensure_thread(client: OpenAI) -> str:
    if st.session_state.get("thread_id") is None:
        thread = client.beta.threads.create()
        st.session_state.thread_id = thread.id
    return st.session_state.thread_id


def upload_files_to_assistants(
    client: OpenAI,
    uploads: List[st.runtime.uploaded_file_manager.UploadedFile],
    progress: Optional[Any] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Upload CSV files to the Assistants API if not already uploaded.
    Returns the updated mapping of filename -> metadata.
    """
    total = len(uploads)
    completed = 0

    def _update_progress(label: str) -> None:
        if progress and total:
            progress.progress(completed / total, text=label)

    for f in uploads:
        if f.name in st.session_state.uploaded_files:
            completed += 1
            _update_progress(f"Already uploaded: {f.name} ({completed}/{total})")
            continue
        file_bytes = f.getvalue()
        uploaded = client.files.create(
            file=(f.name, io.BytesIO(file_bytes)),
            purpose="assistants",
        )
        st.session_state.uploaded_files[f.name] = {
            "id": uploaded.id,
            "name": f.name,
            "size": len(file_bytes),
        }
        completed += 1
        _update_progress(f"Uploaded: {f.name} ({completed}/{total})")

    if progress and total:
        progress.progress(1.0, text="Uploads complete")
    return st.session_state.uploaded_files


def render_file_list(file_map: Dict[str, Dict[str, str]]) -> None:
    if not file_map:
        st.caption("No files uploaded yet.")
        return
    for meta in file_map.values():
        st.caption(f"📎 {meta['name']} ({meta['size']} bytes)")


def stream_assistant_run(
    client: OpenAI,
    thread_id: str,
    assistant_id: str,
    instructions: str,
    placeholder,
) -> Optional[str]:
    """
    Stream the assistant run and render incremental text in the provided placeholder.
    """
    accumulated = ""
    tool_logs: List[str] = []

    try:
        with client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant_id,
            instructions=instructions,
        ) as stream:
            for event in stream:
                event_type = getattr(event, "type", None) or getattr(event, "event", None)

                if event_type and "message.delta" in event_type:
                    delta = event.data.delta
                    for content in getattr(delta, "content", []):
                        if getattr(content, "type", None) == "text" and getattr(content, "text", None):
                            text_chunk = content.text.value or ""
                            accumulated += text_chunk
                            placeholder.markdown(accumulated + " ▌")

                elif event_type and "run.step.delta" in event_type:
                    delta = event.data.delta
                    for output in getattr(delta, "outputs", []):
                        if getattr(output, "type", None) == "logs":
                            tool_logs.append(output.logs or "")
                            log_block = "\n".join(tool_logs)
                            placeholder.markdown(f"{accumulated}\n\n```\n{log_block}\n```\n ▌")

            placeholder.markdown(accumulated)
            return accumulated
    except Exception as exc:  # noqa: BLE001
        placeholder.error(f"Streaming failed: {exc}")
        return None


def main() -> None:
    st.set_page_config(page_title="Assistant with Code Interpreter", layout="wide")
    ensure_state()
    if st.session_state.assistant_prompt is None:
        st.session_state.assistant_prompt = SYSTEM_PROMPT
    client = get_openai_client()

    st.title("Amazon Assistant with Code Interpreter")
    st.write(
        "Upload CSVs and chat with an OpenAI Assistant that can run code to analyze your data."
    )
    thread_label = st.session_state.thread_id or "Not created yet"
    st.info(f"Thread ID: `{thread_label}`", icon="🧵")

    with st.sidebar:
        st.subheader("Assistant settings")
        st.caption("System prompt is configured in code.")
        st.caption(f"Model: {MODEL_DEFAULT}")

        uploaded_files = st.file_uploader(
            "Upload CSV files for analysis",
            type=["csv"],
            accept_multiple_files=True,
            help="Files are securely sent to the Assistant API and available to code interpreter.",
        )
        if uploaded_files:
            progress_placeholder = st.empty()
            progress_bar = progress_placeholder.progress(0.0, text="Uploading files...")
            upload_files_to_assistants(client, uploaded_files, progress=progress_bar)
            progress_placeholder.empty()
        render_file_list(st.session_state.uploaded_files)

        if st.button("Reset conversation"):
            st.session_state.messages = []
            st.session_state.thread_id = None
            st.toast("Conversation reset.")

    # Render previous turns
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a question or request an analysis")
    if user_input:
        assistant_id = create_or_update_assistant(client, SYSTEM_PROMPT, MODEL_DEFAULT)
        thread_id = ensure_thread(client)

        # Record user message locally
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        attachment_payload = [
            {"file_id": meta["id"], "tools": [{"type": "code_interpreter"}]}
            for meta in st.session_state.uploaded_files.values()
        ]

        # Send the message to the thread
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_input,
            attachments=attachment_payload if attachment_payload else None,
        )

        # Stream the assistant reply
        with st.chat_message("assistant"):
            placeholder = st.empty()
            streamed = stream_assistant_run(
                client=client,
                thread_id=thread_id,
                assistant_id=assistant_id,
                instructions=SYSTEM_PROMPT,
                placeholder=placeholder,
            )

        if streamed:
            st.session_state.messages.append({"role": "assistant", "content": streamed})


if __name__ == "__main__":
    main()
