import gradio as gr
from services.rag import NepaliRAG

# Single shared RAG instance
rag = NepaliRAG()

# ── Upload handler ──────────────────────────────────────────────────────────
def handle_upload(file):
    """Called when the user uploads a PDF."""
    if file is None:
        return "No file received.", gr.update(interactive=False)
    status = rag.load_document(file.name)
    return status, gr.update(interactive=True)


# ── Chat handler ─────────────────────────────────────────────────────────────
def chat(message, history, system_prompt):
    """RAG query handler."""
    if not message.strip():
        return ""
    return rag.query(message, system_prompt)


# ── UI ────────────────────────────────────────────────────────────────────────
custom_css = """
body, .gradio-container {
    background: #0f1117 !important;
    font-family: 'Inter', sans-serif;
}
#upload-card {
    background: linear-gradient(145deg, #1a1d2e, #12151f);
    border: 1px solid #2a2d3e;
    border-radius: 16px;
    padding: 24px;
}
#app-title {
    text-align: center;
    background: linear-gradient(135deg, #6c63ff, #e040fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 8px;
}
#app-subtitle {
    text-align: center;
    color: #8b8fa8;
    font-size: 1rem;
    margin-bottom: 32px;
}
.message-wrap .message.bot {
    background-color: #1e2130 !important;
    border-radius: 12px 12px 12px 0px !important;
}
.message-wrap .message.user {
    background-color: #6c63ff !important;
    border-radius: 12px 12px 0px 12px !important;
}
#status-box textarea {
    background: #1e2130 !important;
    color: #a0f0a0 !important;
    border: 1px solid #2a2d3e !important;
    border-radius: 10px !important;
}
#pdf-upload {
    border: 2px dashed #6c63ff !important;
    border-radius: 12px !important;
}
"""

with gr.Blocks() as demo:
    gr.HTML('<div id="app-title">Hamro RAG Assistant</div>')
    gr.HTML('<div id="app-subtitle">Upload a PDF, then chat about it in English or Nepali</div>')

    with gr.Row():
        with gr.Column(scale=1, min_width=300, elem_id="upload-card"):
            gr.Markdown("### Document")
            pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"], elem_id="pdf-upload")
            upload_btn = gr.Button("Index Document", variant="primary")
            initial_status = f"Ready: {rag.current_doc}" if rag.current_doc else "Waiting for document..."
            status_box = gr.Textbox(label="Status", value=initial_status, interactive=False, elem_id="status-box")
            initial_msg_interactive = rag.db is not None
            
            with gr.Accordion("Settings", open=True):
                chat_mode = gr.Radio(
                    label="Chat Mode",
                    choices=["RAG (PDF Context)", "General Chat"],
                    value="RAG (PDF Context)"
                )
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value="You are a Nepali-English expert assistant. Respond in Devanagari script for Nepali queries and in English for English queries.",
                    lines=5
                )

        with gr.Column(scale=3):
            chatbot_ui = gr.Chatbot(height=520, show_label=False)
            msg = gr.Textbox(placeholder="Ask a question...", interactive=True, show_label=False)
            
            with gr.Row():
                submit = gr.Button("Send", variant="primary")
                stop = gr.Button("Stop", variant="stop")
                clear = gr.Button("Clear")

            def respond(message, chat_history, sys_p, mode):
                # 1. Add user message to history immediately
                chat_history.append({"role": "user", "content": message})
                yield "", chat_history
                
                # 2. Add 'Thinking' placeholder
                chat_history.append({"role": "assistant", "content": "Thinking..."})
                yield "", chat_history
                
                # 3. Clear placeholder and start streaming
                chat_history[-1]["content"] = ""
                # Pass the mode to the query function
                for chunk in rag.query(message, sys_p, mode):
                    chat_history[-1]["content"] += chunk
                    yield "", chat_history

            sub_msg = msg.submit(respond, [msg, chatbot_ui, system_prompt, chat_mode], [msg, chatbot_ui])
            sub_btn = submit.click(respond, [msg, chatbot_ui, system_prompt, chat_mode], [msg, chatbot_ui])
            
           
            sub_msg.then(lambda: gr.update(interactive=True), None, [msg])
            sub_btn.then(lambda: gr.update(interactive=True), None, [msg])
            
            # The Stop button cancels the 'respond' generator
            stop.click(fn=None, inputs=None, outputs=None, cancels=[sub_msg, sub_btn])
            
            msg.submit(fn=None, inputs=None, outputs=None, cancels=[sub_msg, sub_btn])
            
            clear.click(lambda: [], None, chatbot_ui, queue=False)

    upload_btn.click(
        fn=handle_upload,
        inputs=[pdf_upload],
        outputs=[status_box, msg],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        theme=gr.themes.Base(primary_hue="violet"),
        css=custom_css
    )