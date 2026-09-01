async function sendMessage() {
    const inputField = document.getElementById('user-input');
    const message = inputField.value.trim();
    if (!message) return;

    appendMessage(message, 'user-message');
    inputField.value = '';

    const typingIndicator = document.getElementById('typing');
    typingIndicator.style.display = 'block';

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();
        typingIndicator.style.display = 'none';

        if(response.ok) {
            appendMessage(data.reply, 'ai-message');
        } else {
            throw new Error(data.detail || "Error en el servidor central.");
        }
    } catch (error) {
        typingIndicator.style.display = 'none';
        appendMessage("Falla de conexión con la red de Teletraan Services. Reintentando...", 'ai-message');
        console.error(error);
    }
}

document.getElementById('user-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') sendMessage();
});

function appendMessage(text, className) {
    const chatBox = document.getElementById('chat-box');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${className}`;
    msgDiv.innerText = text;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}