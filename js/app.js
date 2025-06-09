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
            const li = document.createElement('div');
            li.className = 'chat-session-row' + (session.id === activeSessionId ? ' active' : '');
            li.style.display = 'flex';
            li.style.alignItems = 'center';
            li.style.justifyContent = 'space-between';
            li.style.padding = '6px 14px';
            li.style.cursor = 'pointer';
            li.style.borderRadius = '8px';
            li.style.marginBottom = '2px';
            li.style.position = 'relative';
            if (session.id === activeSessionId) {
                li.style.background = '#f3f3f5';
                li.style.fontWeight = 'bold';
            } else {
                li.onmouseover = () => li.style.background = '#f7f7fa';
                li.onmouseout = () => li.style.background = '';
            }
            // Session name
            const nameSpan = document.createElement('span');
            nameSpan.className = 'session-name';
            nameSpan.textContent = session.name.length > 32 ? session.name.slice(0, 30) + '...' : session.name;
            nameSpan.title = session.name;
            nameSpan.style.overflow = 'hidden';
            nameSpan.style.textOverflow = 'ellipsis';
            nameSpan.style.whiteSpace = 'nowrap';
            nameSpan.style.flex = '1 1 auto';
            nameSpan.style.fontSize = '15px';
            nameSpan.style.lineHeight = '22px';
            li.appendChild(nameSpan);
            li.onclick = (e) => {
                // Only switch if not clicking menu
                if (!e.target.classList.contains('menu-btn')) switchSession(session.id);
            };
            // 3-dot menu for all except 'General'
            if (session.id !== 'general') {
                const menuBtn = document.createElement('button');
                menuBtn.innerHTML = '<span style="font-size:18px;">&#8942;</span>';
                menuBtn.title = 'More actions';
                menuBtn.className = 'session-action-btn menu-btn';
                menuBtn.style.background = 'none';
                menuBtn.style.border = 'none';
                menuBtn.style.cursor = 'pointer';
                menuBtn.style.marginLeft = '8px';
                menuBtn.style.padding = '2px 6px';
                menuBtn.onclick = (e) => {
                    e.stopPropagation();
                    document.querySelectorAll('.session-menu').forEach(m => m.remove());
                    const menu = document.createElement('div');
                    menu.className = 'session-menu';
                    menu.style.position = 'absolute';
                    menu.style.background = '#fff';
                    menu.style.border = '1px solid #e0e0e0';
                    menu.style.boxShadow = '0 4px 16px rgba(0,0,0,0.12)';
                    menu.style.zIndex = 1000;
                    menu.style.right = '8px';
                    menu.style.top = '36px';
                    menu.style.minWidth = '170px';
                    menu.style.borderRadius = '12px';
                    menu.style.padding = '8px 0';
                    menu.style.display = 'flex';
                    menu.style.flexDirection = 'column';
                    // Rename
                    const rename = document.createElement('div');
                    rename.innerHTML = '<span style="margin-right:10px;">✏️</span>Rename';
                    rename.className = 'session-menu-item';
                    rename.style.padding = '10px 20px';
                    rename.style.cursor = 'pointer';
                    rename.onmouseover = () => rename.style.background = '#f7f7fa';
                    rename.onmouseout = () => rename.style.background = '';
                    rename.onclick = (ev) => {
                        ev.stopPropagation();
                        const newName = prompt('Rename session:', session.name);
                        if (newName && newName.trim()) {
                            session.name = newName.trim();
                            saveSessions();
                            renderSessions();
                        }
                        menu.remove();
                    };
                    menu.appendChild(rename);
                    // Divider
                    const divider = document.createElement('div');
                    divider.style.borderTop = '1px solid #eee';
                    divider.style.margin = '4px 0';
                    menu.appendChild(divider);
                    // Delete
                    const del = document.createElement('div');
                    del.innerHTML = '<span style="margin-right:10px;color:#e55353;">🗑️</span><span style="color:#e55353;">Delete</span>';
                    del.className = 'session-menu-item';
                    del.style.padding = '10px 20px';
                    del.style.cursor = 'pointer';
                    del.onmouseover = () => del.style.background = '#f7f7fa';
                    del.onmouseout = () => del.style.background = '';
                    del.onclick = (ev) => {
                        ev.stopPropagation();
                        if (confirm('Delete this session?')) {
                            sessions.splice(idx, 1);
                            localStorage.removeItem('kkh-chat-history-' + session.id);
                            if (activeSessionId === session.id) activeSessionId = 'general';
                            saveSessions();
                            renderSessions();
                            switchSession(activeSessionId);
                        }
                        menu.remove();
                    };
                    menu.appendChild(del);
                    document.addEventListener('click', function closeMenu(e) {
                        if (!menu.contains(e.target) && e.target !== menuBtn) {
                            menu.remove();
                            document.removeEventListener('click', closeMenu);
                        }
                    });
                    li.appendChild(menu);
                };
                li.appendChild(menuBtn);
            }
            chatSessionsList.appendChild(li);
        });
        // Only add one new session button at the bottom (remove duplicate)
        const addBtn = document.createElement('button');
        addBtn.className = 'add-session-btn';
        addBtn.textContent = '+ New Chat';
        addBtn.style.width = '100%';
        addBtn.style.margin = '10px 0 0 0';
        addBtn.style.padding = '10px 0';
        addBtn.style.background = '#f3f3f5';
        addBtn.style.border = 'none';
        addBtn.style.borderRadius = '8px';
        addBtn.style.fontWeight = 'bold';
        addBtn.style.fontSize = '15px';
        addBtn.style.cursor = 'pointer';
        addBtn.onmouseover = () => addBtn.style.background = '#ececf1';
        addBtn.onmouseout = () => addBtn.style.background = '#f3f3f5';
        addBtn.onclick = () => {
            const name = prompt('Session name?');
            if (!name) return;
            const id = name.toLowerCase().replace(/\s+/g, '-') + '-' + Date.now();
            sessions.push({ name, id });
            saveSessions();
            renderSessions();
            switchSession(id);
        };
        chatSessionsList.appendChild(addBtn);
    }
    // --- Prompt Gallery as Chat History for Selected Prompt ---
    function getSessionHistoryPrompts() {
        // Get all user-bot Q&A pairs for the current session, deduplicated by user question
        const key = 'kkh-chat-history-' + activeSessionId;
        const history = JSON.parse(localStorage.getItem(key) || '[]');
        const seen = new Set();
        const pairs = [];
        for (let i = 0; i < history.length - 1; i++) {
            if (history[i].sender === 'user' && history[i + 1].sender === 'bot') {
                const userQ = history[i].text;
                if (!seen.has(userQ)) {
                    seen.add(userQ);
                    pairs.push({ name: userQ, text: history[i + 1].text });
                }
            }
        }
        return pairs;
    }

    function renderPrompts() {
        promptList.innerHTML = '';
        const sessionPrompts = getSessionHistoryPrompts();
        sessionPrompts.forEach((prompt, idx) => {
            const li = document.createElement('div');
            li.className = 'prompt-item-row';
            li.style.display = 'flex';
            li.style.alignItems = 'center';
            li.style.padding = '6px 14px';
            li.style.cursor = 'pointer';
            li.style.borderRadius = '8px';
            li.style.marginBottom = '2px';
            li.onmouseover = () => li.style.background = '#f7f7fa';
            li.onmouseout = () => li.style.background = '';
            // Truncate and wrap prompt names for sidebar fit
            let displayName = prompt.name.length > 32 ? prompt.name.slice(0, 30) + '...' : prompt.name;
            const nameSpan = document.createElement('span');
            nameSpan.className = 'prompt-name';
            nameSpan.textContent = displayName;
            nameSpan.title = prompt.name;
            nameSpan.style.overflow = 'hidden';
            nameSpan.style.textOverflow = 'ellipsis';
            nameSpan.style.whiteSpace = 'nowrap';
            nameSpan.style.flex = '1 1 auto';
            nameSpan.style.fontSize = '15px';
            nameSpan.style.lineHeight = '22px';
            li.appendChild(nameSpan);
            li.tabIndex = 0;
            // On click, show a modal or alert with the full chat history for this prompt
            li.onclick = () => {
                const key = 'kkh-chat-history-' + activeSessionId;
                const history = JSON.parse(localStorage.getItem(key) || '[]');
                let chatHistory = '';
                for (let i = 0; i < history.length - 1; i++) {
                    if (history[i].sender === 'user' && history[i].text === prompt.name && history[i + 1].sender === 'bot') {
                        chatHistory += 'You: ' + history[i].text + '\n';
                        chatHistory += 'Bot: ' + history[i + 1].text + '\n\n';
                    }
                }
                if (chatHistory) {
                    alert(chatHistory.trim());
                } else {
                    alert('No chat history found for this prompt.');
                }
            };
            promptList.appendChild(li);
        });
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
            // Only show welcome message if not already present
            if (!chatWindow.querySelector('.message.bot')) {
                appendMessage('bot', 'Hello! I am your KKH Nursing Chatbot. How can I assist you today?');
            }
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
        // No manual prompt gallery update needed; renderPrompts will reflect history
        renderPrompts();
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
