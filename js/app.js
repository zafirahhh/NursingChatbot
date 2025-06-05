// KKH Nursing Chatbot - Basic Chat Interactivity

document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.getElementById('chat-window');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');

    // Avatars
    const avatars = {
        user: '👩',
        bot: '🤖'
    };

    // Load chat history from localStorage
    function loadHistory() {
        const history = JSON.parse(localStorage.getItem('kkh-chat-history') || '[]');
        history.forEach(msg => appendMessage(msg.sender, msg.text, false));
    }

    // Save message to localStorage
    function saveMessage(sender, text) {
        const history = JSON.parse(localStorage.getItem('kkh-chat-history') || '[]');
        history.push({ sender, text });
        localStorage.setItem('kkh-chat-history', JSON.stringify(history));
    }

    // Append message to chat window
    function appendMessage(sender, text, save = true) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        const avatarSpan = document.createElement('span');
        avatarSpan.className = 'avatar';
        avatarSpan.textContent = avatars[sender];
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = text;
        messageDiv.appendChild(avatarSpan);
        messageDiv.appendChild(contentDiv);
        chatWindow.appendChild(messageDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
        if (save) saveMessage(sender, text);
    }

    // Typing indicator
    function showTyping() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot typing';
        const avatarSpan = document.createElement('span');
        avatarSpan.className = 'avatar';
        avatarSpan.textContent = avatars.bot;
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = 'Typing...';
        typingDiv.appendChild(avatarSpan);
        typingDiv.appendChild(contentDiv);
        typingDiv.id = 'typing-indicator';
        chatWindow.appendChild(typingDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
    function removeTyping() {
        const typingDiv = document.getElementById('typing-indicator');
        if (typingDiv) typingDiv.remove();
    }

    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const userText = userInput.value.trim();
        if (!userText) return;
        appendMessage('user', userText);
        userInput.value = '';
        showTyping();
        setTimeout(() => {
            removeTyping();
            appendMessage('bot', 'Thank you for your question. A virtual nurse will respond shortly.');
        }, 900);
    });

    // Accessibility: focus chat window on load
    chatWindow.focus();
    loadHistory();
});
