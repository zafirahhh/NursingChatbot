// KKH Nursing Chatbot - Basic Chat Interactivity

document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.getElementById('chat-window');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const clearChatBtn = document.getElementById('clear-chat');
    const micBtn = document.getElementById('mic-btn');
    const chatSessionsList = document.getElementById('chat-sessions');
    const promptList = document.getElementById('prompt-list');
    const addSessionBtn = document.getElementById('add-session');
    const addPromptBtn = document.getElementById('add-prompt');

    // Avatars
    const avatars = {
        user: '👩',
        bot: '🤖'
    };

    // --- Sessions & Prompts Integration ---
    let sessions = JSON.parse(localStorage.getItem('kkh-sessions') || '[{"name":"General","id":"general"}]');
    let prompts = JSON.parse(localStorage.getItem('kkh-prompts') || '[{"name":"Default Prompt","text":""}]');
    let activeSessionId = localStorage.getItem('kkh-active-session') || 'general';
    let activePromptIdx = 0;

    function renderSessions() {
        chatSessionsList.innerHTML = '';
        sessions.forEach((session, idx) => {
            const li = document.createElement('li');
            li.className = 'chat-session' + (session.id === activeSessionId ? ' active' : '');
            li.textContent = session.name;
            li.tabIndex = 0;
            li.onclick = () => switchSession(session.id);
            // Add rename and delete buttons except for 'General'
            if (session.id !== 'general') {
                const renameBtn = document.createElement('button');
                renameBtn.textContent = '✏️';
                renameBtn.title = 'Rename Session';
                renameBtn.className = 'session-action-btn';
                renameBtn.onclick = (e) => {
                    e.stopPropagation();
                    const newName = prompt('Rename session:', session.name);
                    if (newName && newName.trim()) {
                        session.name = newName.trim();
                        saveSessions();
                        renderSessions();
                    }
                };
                li.appendChild(renameBtn);
                const delBtn = document.createElement('button');
                delBtn.textContent = '🗑️';
                delBtn.title = 'Delete Session';
                delBtn.className = 'session-action-btn';
                delBtn.onclick = (e) => {
                    e.stopPropagation();
                    if (confirm('Delete this session?')) {
                        sessions.splice(idx, 1);
                        localStorage.removeItem('kkh-chat-history-' + session.id);
                        if (activeSessionId === session.id) activeSessionId = 'general';
                        saveSessions();
                        renderSessions();
                        switchSession(activeSessionId);
                    }
                };
                li.appendChild(delBtn);
            }
            chatSessionsList.appendChild(li);
        });
        // Add "+ new session" button at the end
        const addLi = document.createElement('li');
        addLi.className = 'chat-session add-new';
        addLi.textContent = '+ new session';
        addLi.tabIndex = 0;
        addLi.onclick = () => {
            const name = prompt('Session name?');
            if (!name) return;
            const id = name.toLowerCase().replace(/\s+/g, '-') + '-' + Date.now();
            sessions.push({ name, id });
            saveSessions();
            renderSessions();
            switchSession(id);
        };
        chatSessionsList.appendChild(addLi);
    }
    function renderPrompts() {
        promptList.innerHTML = '';
        prompts.forEach((prompt, idx) => {
            const li = document.createElement('li');
            li.className = 'prompt-item' + (idx === activePromptIdx ? ' active' : '');
            li.textContent = prompt.name;
            li.tabIndex = 0;
            li.onclick = () => { activePromptIdx = idx; renderPrompts(); };
            // Add delete button except for default
            if (idx !== 0) {
                const delBtn = document.createElement('button');
                delBtn.textContent = '🗑️';
                delBtn.title = 'Delete Prompt';
                delBtn.className = 'prompt-action-btn';
                delBtn.onclick = (e) => {
                    e.stopPropagation();
                    prompts.splice(idx, 1);
                    if (activePromptIdx >= prompts.length) activePromptIdx = 0;
                    savePrompts();
                    renderPrompts();
                };
                li.appendChild(delBtn);
            }
            promptList.appendChild(li);
        });
        // Add "+ new prompt" button at the end
        const addLi = document.createElement('li');
        addLi.className = 'prompt-item add-new';
        addLi.textContent = '+ new prompt';
        addLi.tabIndex = 0;
        addLi.onclick = () => {
            // Use last user message as prompt name and last bot reply as prompt text
            const key = 'kkh-chat-history-' + activeSessionId;
            const history = JSON.parse(localStorage.getItem(key) || '[]');
            if (history.length < 2) {
                alert('Ask a question first to generate a prompt.');
                return;
            }
            let lastUser = null, lastBot = null;
            for (let i = history.length - 1; i >= 0; i--) {
                if (!lastBot && history[i].sender === 'bot') lastBot = history[i].text;
                if (!lastUser && history[i].sender === 'user') lastUser = history[i].text;
                if (lastUser && lastBot) break;
            }
            if (!lastUser || !lastBot) {
                alert('Need both a user question and a bot answer to generate a prompt.');
                return;
            }
            // Only add if not already present
            if (!prompts.some(p => p.name === lastUser && p.text === lastBot)) {
                prompts.push({ name: lastUser, text: lastBot });
                savePrompts();
                renderPrompts();
            } else {
                alert('Prompt already exists.');
            }
        };
        promptList.appendChild(addLi);
    }
    function switchSession(sessionId) {
        activeSessionId = sessionId;
        localStorage.setItem('kkh-active-session', sessionId);
        renderSessions();
        loadHistory();
    }
    function saveSessions() {
        localStorage.setItem('kkh-sessions', JSON.stringify(sessions));
    }
    function savePrompts() {
        localStorage.setItem('kkh-prompts', JSON.stringify(prompts));
    }
    addSessionBtn.onclick = () => {
        const name = prompt('Session name?');
        if (!name) return;
        const id = name.toLowerCase().replace(/\s+/g, '-') + '-' + Date.now();
        sessions.push({ name, id });
        saveSessions();
        renderSessions();
    };
    addPromptBtn.onclick = () => {
        // Instead of manual prompt, use last user message as prompt name and last bot reply as prompt text
        const key = 'kkh-chat-history-' + activeSessionId;
        const history = JSON.parse(localStorage.getItem(key) || '[]');
        if (history.length < 2) {
            alert('Ask a question first to generate a prompt.');
            return;
        }
        // Find last user message and last bot reply
        let lastUser = null, lastBot = null;
        for (let i = history.length - 1; i >= 0; i--) {
            if (!lastBot && history[i].sender === 'bot') lastBot = history[i].text;
            if (!lastUser && history[i].sender === 'user') lastUser = history[i].text;
            if (lastUser && lastBot) break;
        }
        if (!lastUser || !lastBot) {
            alert('Need both a user question and a bot answer to generate a prompt.');
            return;
        }
        prompts.push({ name: lastUser, text: lastBot });
        savePrompts();
        renderPrompts();
    };

    // --- Chat History Management ---
    function loadHistory() {
        chatWindow.innerHTML = '';
        const history = JSON.parse(localStorage.getItem('kkh-chat-history-' + activeSessionId) || '[]');
        if (history.length === 0) {
            // Show welcome message if no history
            appendMessage('bot', 'Hello! I am your KKH Nursing Chatbot. How can I assist you today?');
        } else {
            history.forEach(msg => appendMessage(msg.sender, msg.text, false));
        }
    }

    // Save message to localStorage
    function saveMessage(sender, text) {
        const key = 'kkh-chat-history-' + activeSessionId;
        const history = JSON.parse(localStorage.getItem(key) || '[]');
        history.push({ sender, text });
        localStorage.setItem(key, JSON.stringify(history));
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

    let knowledgeText = '';

    // Load the knowledge base from the PDF (as text)
    fetch('data/nursing_guide.txt')
      .then(response => response.text())
      .then(text => {
        knowledgeText = text;
      });

    // Replace searchKnowledge with backend API call
    async function searchKnowledge(query) {
      try {
        // Use local backend for development
        const response = await fetch('http://127.0.0.1:8000/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query })
        });
        if (!response.ok) throw new Error('Backend error');
        const data = await response.json();
        return data.answer || 'No relevant information found.';
      } catch (err) {
        return 'Error contacting backend: ' + err.message;
      }
    }

    // --- Prepend prompt to user message on submit ---
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const userText = userInput.value.trim();
        if (!userText) return;
        // Auto-create new session if in General
        if (activeSessionId === 'general') {
            const id = 'session-' + Date.now();
            const name = userText.slice(0, 20) + (userText.length > 20 ? '...' : '');
            sessions.push({ name, id });
            saveSessions();
            renderSessions();
            switchSession(id);
        }
        let promptText = prompts[activePromptIdx]?.text || '';
        let fullText = promptText ? promptText + '\n' + userText : userText;
        appendMessage('user', userText);
        userInput.value = '';
        showTyping();
        const botReply = await searchKnowledge(fullText);
        removeTyping();
        appendMessage('bot', botReply);
        // --- Auto-generate prompt after each Q&A ---
        // Only add if not already present
        if (!prompts.some(p => p.name === userText && p.text === botReply)) {
            prompts.push({ name: userText, text: botReply });
            savePrompts();
            renderPrompts();
        }
    });

    // Clear chat functionality
    clearChatBtn.addEventListener('click', () => {
        localStorage.removeItem('kkh-chat-history');
        chatWindow.innerHTML = '';
        appendMessage('bot', 'Hello! I am your KKH Nursing Chatbot. How can I assist you today?');
    });

    // Voice recording setup (Web Speech API)
    let recognizing = false;
    let recognition;

    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;

        recognition.onstart = () => {
            recognizing = true;
            micBtn.classList.add('recording');
            micBtn.setAttribute('aria-pressed', 'true');
        };
        recognition.onend = () => {
            recognizing = false;
            micBtn.classList.remove('recording');
            micBtn.setAttribute('aria-pressed', 'false');
        };
        recognition.onerror = (event) => {
            recognizing = false;
            micBtn.classList.remove('recording');
            micBtn.setAttribute('aria-pressed', 'false');
            alert('Voice recognition error: ' + event.error);
        };
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            userInput.value = transcript;
            userInput.focus();
        };

        micBtn.addEventListener('click', () => {
            if (recognizing) {
                recognition.stop();
            } else {
                recognition.start();
            }
        });
    } else {
        micBtn.disabled = true;
        micBtn.title = 'Voice recognition not supported in this browser.';
    }

    // Accessibility: focus chat window on load
    chatWindow.focus();
    renderSessions();
    renderPrompts();
    loadHistory();
});
