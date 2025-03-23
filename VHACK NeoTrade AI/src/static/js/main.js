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
});
