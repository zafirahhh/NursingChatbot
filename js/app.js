// KKH Nursing Chatbot - Final Grouped Sessions App.js (MCQ with radio, explanation, 5 options)

const BACKEND_URL_FINAL = "http://127.0.0.1:8000/ask";
const QUIZ_URL_FINAL = "http://127.0.0.1:8000/quiz";
const QUIZ_EVAL_URL_FINAL = "http://127.0.0.1:8000/quiz/evaluate";

// --- Sidebar Grouped Sessions UI Logic for new sidebar structure ---
document.addEventListener('DOMContentLoaded', () => {
  const chatWindow = document.getElementById('chat-window');
  const chatForm = document.getElementById('chat-form');
  const userInput = document.getElementById('user-input');
  const clearChatBtn = document.getElementById('clear-chat');
  const micBtn = document.getElementById('mic-btn');
  const avatars = { user: '👩', bot: '🤖' };

  // Load sessions or fallback
  let groupedSessions = JSON.parse(localStorage.getItem('kkh-grouped-sessions') || JSON.stringify([
    { category: 'General', expanded: true, chats: [{ name: "Welcome", id: "general-welcome" }] },
    { category: 'Quiz', expanded: true, chats: [] }
  ]));
  let activeSessionId = localStorage.getItem('kkh-active-session') || 'general-welcome';
  let currentQuiz = [];
  let quizAnswers = {};

  clearChatBtn.addEventListener('click', () => {
    localStorage.removeItem('kkh-grouped-sessions');
    localStorage.removeItem('kkh-active-session');
    // Remove all chat histories
    Object.keys(localStorage).forEach(key => {
      if (key.startsWith('kkh-chat-history-')) localStorage.removeItem(key);
    });
    location.reload();
  });

  function renderSessions() {
    const generalList = document.getElementById('session-general-wrapper');
    const quizList = document.getElementById('session-quiz-wrapper');
    generalList.innerHTML = '';
    quizList.innerHTML = '';

    groupedSessions.forEach(group => {
      const target = group.category === 'General' ? generalList : quizList;
      group.chats.forEach((chat, index) => {
        const div = document.createElement('div');
        div.className = 'chat-session';
        div.innerHTML = `
          <span>${chat.name}</span>
          <div class="chat-menu">⋮
            <div class="chat-dropdown">
              <div class="rename-option" data-group="${group.category}" data-index="${index}">Rename</div>
              <div class="delete-option" data-group="${group.category}" data-index="${index}">Delete</div>
            </div>
          </div>`;
        if (chat.id === activeSessionId) div.classList.add('active');
        div.onclick = (e) => {
          if (!e.target.classList.contains('chat-menu') && !e.target.classList.contains('chat-dropdown') && !e.target.classList.contains('rename-option') && !e.target.classList.contains('delete-option')) {
            switchSession(group, chat, index);
          }
        };
        target.appendChild(div);
      });
    });
    attachMenuHandlers();
  }

  // New session buttons
  document.querySelectorAll('.new-session-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const category = btn.getAttribute('data-category');
      const group = groupedSessions.find(g => g.category === category);
      const id = `${category.toLowerCase()}-${Date.now()}`;
      const newSession = {
        name: category === 'Quiz' ? `Quiz Attempt ${group.chats.length + 1}` : `Chat ${group.chats.length + 1}`,
        id
      };
      group.chats.push(newSession);
      localStorage.setItem('kkh-grouped-sessions', JSON.stringify(groupedSessions));
      activeSessionId = id;
      localStorage.setItem('kkh-active-session', activeSessionId);
      renderSessions();
      loadHistory();
    });
  });

  function attachMenuHandlers() {
    document.querySelectorAll('.rename-option').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const group = btn.getAttribute('data-group');
        const index = btn.getAttribute('data-index');
        const newName = prompt('Enter new name:');
        if (newName) {
          const g = groupedSessions.find(g => g.category === group);
          g.chats[index].name = newName;
          localStorage.setItem('kkh-grouped-sessions', JSON.stringify(groupedSessions));
          renderSessions();
        }
      });
    });
    document.querySelectorAll('.delete-option').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const group = btn.getAttribute('data-group');
        const index = btn.getAttribute('data-index');
        if (confirm('Delete this session?')) {
          const g = groupedSessions.find(g => g.category === group);
          const sessionId = g.chats[index].id;
          g.chats.splice(index, 1);
          localStorage.setItem('kkh-grouped-sessions', JSON.stringify(groupedSessions));
          localStorage.removeItem('kkh-chat-history-' + sessionId);
          if (activeSessionId === sessionId) {
            activeSessionId = 'general-welcome';
            localStorage.setItem('kkh-active-session', activeSessionId);
          }
          renderSessions();
          loadHistory();
        }
      });
    });
  }

  function switchSession(group, chat, index) {
    activeSessionId = chat.id;
    localStorage.setItem('kkh-active-session', activeSessionId);
    renderSessions();
    loadHistory();
    if (group.category === 'Quiz') {
      // Load quiz questions for this session
      fetch(`${QUIZ_URL_FINAL}?n=5`)
        .then(res => res.json())
        .then(data => {
          if (data.quiz) {
            appendGroupedMessage('bot', '📝 Here are your quiz questions:');
            currentQuiz = data.quiz;
            quizAnswers = {};
            data.quiz.forEach((q, idx) => {
              const quizContainer = document.createElement('div');
              quizContainer.className = 'quiz-block';
              const questionText = document.createElement('p');
              questionText.innerHTML = `<strong>Q${idx + 1}:</strong> ${q.question}`;
              quizContainer.appendChild(questionText);
              q.options.slice(0, 5).forEach((opt, i) => {
                const wrapper = document.createElement('label');
                wrapper.style.display = 'flex';
                wrapper.style.alignItems = 'center';
                wrapper.style.margin = '4px 0';
                const radio = document.createElement('input');
                radio.type = 'radio';
                radio.name = `quiz-${idx}`;
                radio.value = opt;
                radio.style.marginRight = '10px';
                radio.onclick = () => quizAnswers[idx] = opt;
                const text = document.createElement('span');
                text.textContent = `${String.fromCharCode(65 + i)}. ${opt}`;
                wrapper.appendChild(radio);
                wrapper.appendChild(text);
                quizContainer.appendChild(wrapper);
              });
              chatWindow.appendChild(quizContainer);
            });
            const submitBtn = document.createElement('button');
            submitBtn.textContent = 'Submit Quiz';
            submitBtn.className = 'sidebar-btn';
            submitBtn.style.marginTop = '1rem';
            submitBtn.onclick = async () => {
              const userResponses = currentQuiz.map((q, i) => ({
                question: q.question,
                answer: quizAnswers[i] || ''
              }));
              const result = await fetch(QUIZ_EVAL_URL_FINAL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ responses: userResponses })
              });
              const feedback = await result.json();
              let score = 0;
              feedback.forEach((item, i) => {
                const block = document.querySelectorAll('.quiz-block')[i];
                const inputs = block.querySelectorAll('input');
                const explanation = document.createElement('div');
                explanation.style.marginTop = '8px';
                explanation.style.fontSize = '14px';
                inputs.forEach(input => {
                  if (input.value === item.correctAnswer) {
                    input.parentElement.style.background = '#c8facc';
                  }
                  if (input.checked && input.value !== item.correctAnswer) {
                    input.parentElement.style.background = '#ffc8c8';
                  }
                });
                if (item.correct) score++;
                if (!item.correct) {
                  explanation.innerHTML = `❌ <strong>Explanation:</strong> ${item.explanation || 'Refer to nursing guide for details.'}`;
                  block.appendChild(explanation);
                }
              });
              appendGroupedMessage('bot', `✅ You scored ${score} out of ${currentQuiz.length}`);
            };
            chatWindow.appendChild(submitBtn);
            chatWindow.scrollTop = chatWindow.scrollHeight;
          }
        });
    }
  }

  function loadHistory() {
    chatWindow.innerHTML = '';
    const history = JSON.parse(localStorage.getItem('kkh-chat-history-' + activeSessionId) || '[]');
    if (history.length === 0) {
      appendGroupedMessage('bot', 'Hello! I am your KKH Nursing Chatbot. How can I assist you today?', false);
    } else {
      history.forEach(msg => appendGroupedMessage(msg.sender, msg.text, false));
    }
  }

  function appendGroupedMessage(sender, text, save = true) {
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
    if (save) saveGroupedMessage(sender, text);
  }

  function saveGroupedMessage(sender, text) {
    if (!activeSessionId) {
      console.warn('No active session ID set.');
      return;
    }
    const key = 'kkh-chat-history-' + activeSessionId;
    const history = JSON.parse(localStorage.getItem(key) || '[]');
    history.push({ sender, text });
    localStorage.setItem(key, JSON.stringify(history));
  }

  // Typing indicator
  function showTyping() {
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typing-indicator';
    typingDiv.className = 'message bot';
    typingDiv.innerHTML = `
      <span class="avatar">🤖</span>
      <div class="message-content">...</div>
    `;
    chatWindow.appendChild(typingDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }
  function removeTyping() {
    const typingDiv = document.getElementById('typing-indicator');
    if (typingDiv) typingDiv.remove();
  }

  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const userText = userInput.value.trim();
    if (!userText) return;
    console.log('[User Submit]', userText, 'Session:', activeSessionId);
    appendGroupedMessage('user', userText);
    userInput.value = '';
    const isQuiz = activeSessionId.startsWith('quiz');
    const url = isQuiz ? QUIZ_URL_FINAL : BACKEND_URL_FINAL;
    const payload = isQuiz ? { prompt: userText } : { question: userText, session: activeSessionId };
    console.log('[Submit to]', url);
    console.log('[Payload]', payload);
    showTyping();
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      removeTyping();
      if (data.answer) {
        appendGroupedMessage('bot', data.answer);
      } else if (data.summary) {
        appendGroupedMessage('bot', data.summary);
      } else if (data.full) {
        appendGroupedMessage('bot', data.full);
      } else if (data.quiz) {
        appendGroupedMessage('bot', '📝 Quiz Loaded');
      } else {
        appendGroupedMessage('bot', '⚠️ Unexpected response from backend.');
      }
    } catch (err) {
      removeTyping();
      appendGroupedMessage('bot', '❌ Failed to reach server: ' + err.message);
    }
  });

  renderSessions();
  loadHistory();
});
