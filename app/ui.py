from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])


@router.get("/ui", response_class=HTMLResponse)
async def ui_page() -> str:
    """
    UI chat đơn giản kiểu ChatGPT, gọi trực tiếp /api/v1/chat.
    """
    return """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8" />
        <title>Wiki Chatbot RAG</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <style>
            *,
            *::before,
            *::after {
                box-sizing: border-box;
            }

            body {
                margin: 0;
                min-height: 100vh;
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: #050816;
                color: #e5e7eb;
                display: flex;
                justify-content: center;
                align-items: stretch;
                padding: 16px;
            }

            .wrapper {
                width: 100%;
                max-width: 900px;
                background: #020617;
                border-radius: 16px;
                border: 1px solid #1f2937;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }

            .header {
                padding: 12px 16px;
                border-bottom: 1px solid #1f2937;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .title {
                font-size: 16px;
                font-weight: 600;
            }

            .subtitle {
                font-size: 12px;
                color: #9ca3af;
            }

            .chat {
                flex: 1;
                padding: 12px 16px;
                overflow-y: auto;
                background: radial-gradient(circle at top, #020617, #020617 60%, #000 100%);
            }

            .msg {
                max-width: 80%;
                margin-bottom: 10px;
                padding: 8px 10px;
                border-radius: 12px;
                font-size: 14px;
                line-height: 1.4;
                white-space: pre-wrap;
            }

            .msg.user {
                margin-left: auto;
                background: #0ea5e9;
                color: #0b1120;
            }

            .msg.bot {
                margin-right: auto;
                background: #111827;
                border: 1px solid #1f2937;
            }

            .msg-label {
                font-size: 11px;
                opacity: 0.7;
                margin-bottom: 2px;
            }

            .input-bar {
                border-top: 1px solid #1f2937;
                padding: 8px 10px;
                display: flex;
                gap: 8px;
                align-items: flex-end;
                background: #020617;
            }

            textarea {
                flex: 1;
                resize: none;
                border-radius: 12px;
                border: 1px solid #374151;
                padding: 8px 10px;
                background: #020617;
                color: #e5e7eb;
                font-size: 14px;
                max-height: 120px;
            }

            textarea:focus {
                outline: none;
                border-color: #0ea5e9;
            }

            button {
                border: none;
                border-radius: 999px;
                padding: 8px 14px;
                font-size: 14px;
                font-weight: 500;
                cursor: pointer;
                background: #0ea5e9;
                color: #0b1120;
            }

            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }

            .status {
                font-size: 11px;
                color: #9ca3af;
                padding: 4px 16px 8px;
            }

            .small {
                font-size: 11px;
                color: #6b7280;
            }
        </style>
    </head>
    <body>
        <div class="wrapper">
            <div class="header">
                <div>
                    <div class="title">Wiki Chatbot RAG</div>
                </div>
            </div>
            <div id="chat" class="chat">
            </div>
            <div class="status" id="status">Sẵn sàng.</div>
            <div class="input-bar">
                <textarea id="question" rows="1" placeholder="Nhập câu hỏi và nhấn Enter..."></textarea>
                <button id="send">Gửi</button>
            </div>
        </div>

        <script>
            const chatEl = document.getElementById("chat");
            const questionEl = document.getElementById("question");
            const sendBtn = document.getElementById("send");
            const statusEl = document.getElementById("status");

            function appendMsg(role, text) {
                const div = document.createElement("div");
                div.className = "msg " + (role === "user" ? "user" : "bot");
                const label = document.createElement("div");
                label.className = "msg-label";
                label.textContent = role === "user" ? "Bạn" : "Bot";
                const body = document.createElement("div");
                body.textContent = text;
                div.appendChild(label);
                div.appendChild(body);
                chatEl.appendChild(div);
                chatEl.scrollTop = chatEl.scrollHeight;
            }

            async function sendQuestion() {
                const q = questionEl.value.trim();
                if (!q) return;

                appendMsg("user", q);
                questionEl.value = "";

                sendBtn.disabled = true;
                statusEl.textContent = "Đang suy nghĩ...";

                try {
                    const res = await fetch("/api/v1/chat", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ question: q, document_ids: null, verbose: false }),
                    });

                    if (!res.ok) {
                        const t = await res.text();
                        throw new Error("Lỗi " + res.status + ": " + t);
                    }

                    const data = await res.json();
                    const answer = data.answer || "[Không nhận được câu trả lời]";
                    appendMsg("bot", answer);
                } catch (e) {
                    console.error(e);
                    appendMsg("bot", "Có lỗi khi gọi /api/v1/chat: " + (e.message || e));
                } finally {
                    sendBtn.disabled = false;
                    statusEl.textContent = "Sẵn sàng.";
                }
            }

            sendBtn.addEventListener("click", (e) => {
                e.preventDefault();
                sendQuestion();
            });

            questionEl.addEventListener("keydown", (e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendQuestion();
                }
            });
        </script>
    </body>
    </html>
    """

