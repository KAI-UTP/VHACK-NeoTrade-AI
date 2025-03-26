// Modal and Chat Logic

document.addEventListener("DOMContentLoaded", () => {
    // 🔐 Login Modal Logic
    const modal = document.getElementById("loginModal");
    const openBtn = document.getElementById("openModalBtn");
    const closeBtn = document.getElementById("closeModalBtn");

    openBtn?.addEventListener("click", (e) => {
        e.preventDefault();
        modal.style.display = "block";
    });

    closeBtn?.addEventListener("click", () => {
        modal.style.display = "none";
    });

    window.addEventListener("click", (e) => {
        if (e.target === modal) {
            modal.style.display = "none";
        }
    });

    // 💬 Chat Modal Toggle Logic
    const chatToggle = document.getElementById("chatToggle");
    const chatBox = document.getElementById("chatBox");
    const closeChat = document.getElementById("closeChat");
    const sendChat = document.getElementById("sendChat");
    const chatInput = document.getElementById("chatInput");
    const chatBody = document.getElementById("chatBody");

    chatToggle?.addEventListener("click", () => {
        chatBox.style.display = "flex";
        chatToggle.style.display = "none";
    });

    closeChat?.addEventListener("click", () => {
        chatBox.style.display = "none";
        chatToggle.style.display = "block";
    });

    // 💬 Chat Sending Logic
    sendChat?.addEventListener("click", async () => {
        const msg = chatInput.value.trim();
        if (msg !== "") {
            const userMsg = document.createElement("p");
            userMsg.textContent = msg;
            userMsg.style.textAlign = "right";
            userMsg.style.backgroundColor = "#d1e7dd";
            userMsg.style.padding = "8px";
            userMsg.style.borderRadius = "8px";
            userMsg.style.marginBottom = "10px";
            chatBody.appendChild(userMsg);
            chatInput.value = "";
            chatBody.scrollTop = chatBody.scrollHeight;

            try {
                const res = await fetch("/chat", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ message: msg })
                });

                const data = await res.json();

                const reply = document.createElement("p");
                reply.textContent = data.reply;
                reply.className = "bot-msg";
                chatBody.appendChild(reply);
                chatBody.scrollTop = chatBody.scrollHeight;

            } catch (error) {
                const errorMsg = document.createElement("p");
                errorMsg.textContent = "Error contacting server.";
                errorMsg.className = "bot-msg";
                chatBody.appendChild(errorMsg);
            }
        }
    });

    // 💬 Tchat (Tutorial Chat) Logic
    const tchatInput = document.getElementById("tchatInput");
    const sendTchat = document.getElementById("sendTchat");
    const tchatBody = document.getElementById("tchatBody");
    const tchatBox = document.querySelector(".tutorial-right .chat-box");

    sendTchat?.addEventListener("click", async () => {
        const msg = tchatInput.value.trim();
        if (msg !== "") {
            const userMsg = document.createElement("p");
            userMsg.textContent = msg;
            userMsg.style.textAlign = "right";
            userMsg.style.backgroundColor = "#d1e7dd";
            userMsg.style.padding = "8px";
            userMsg.style.borderRadius = "8px";
            userMsg.style.marginBottom = "10px";
            tchatBody.appendChild(userMsg);
            tchatInput.value = "";
            tchatBody.scrollTop = tchatBody.scrollHeight;

            try {
                const res = await fetch("/tchat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ message: msg })
                });

                const data = await res.json();

                const reply = document.createElement("p");
                reply.textContent = data.reply;
                reply.className = "bot-msg";
                tchatBody.appendChild(reply);
                tchatBody.scrollTop = tchatBody.scrollHeight;

            } catch (error) {
                const errorMsg = document.createElement("p");
                errorMsg.textContent = "Error contacting Tchat.";
                errorMsg.className = "bot-msg";
                tchatBody.appendChild(errorMsg);
            }
        }
    });

    // 🟦 Make chat box draggable
    function makeChatDraggable(chatBox) {
        const header = chatBox.querySelector('.chat-header');
        let isDragging = false;
        let offsetX, offsetY;

        header?.addEventListener('mousedown', function (e) {
            isDragging = true;
            offsetX = e.clientX - chatBox.getBoundingClientRect().left;
            offsetY = e.clientY - chatBox.getBoundingClientRect().top;
            chatBox.style.position = 'fixed';
        });

        document.addEventListener('mousemove', function (e) {
            if (isDragging) {
                chatBox.style.left = (e.clientX - offsetX) + 'px';
                chatBox.style.top = (e.clientY - offsetY) + 'px';
                chatBox.style.right = 'auto';
            }
        });

        document.addEventListener('mouseup', function () {
            isDragging = false;
        });
    }

    if (chatBox) makeChatDraggable(chatBox);
    if (tchatBox) makeChatDraggable(tchatBox);

});

function fetchSignalPanel() {
    fetch('/latest_signal')
        .then(res => res.json())
        .then(data => {
            document.getElementById("signal-status").innerHTML = formatSignal(data.signal);
            document.getElementById("entry-price").textContent = data.entry_price;
            document.getElementById("stop-loss").textContent = data.stop_loss;
            document.getElementById("take-profit").textContent = data.take_profit;
            document.getElementById("profit-loss").textContent = data.pnl;
            document.getElementById("last-update").textContent = data.last_updated;
        });
}

function formatSignal(signal) {
    if (signal === "BUY") return `<span class="dot green"></span> BUY`;
    if (signal === "SELL") return `<span class="dot red"></span> SELL`;
    return `<span class="dot yellow"></span> HOLD`;
}

// Trigger every 15 minutes
setInterval(fetchSignalPanel, 15 * 60 * 1000);
fetchSignalPanel(); // Call immediately on load too

