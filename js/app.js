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

  // --- Restore Default Session Structure if Missing ---
  let groupedSessions = JSON.parse(localStorage.getItem('kkh-grouped-sessions'));
  if (!groupedSessions) {
    groupedSessions = [
      {
        category: "General",
        expanded: true,
        chats: [{ name: "Chat 1" }]
      },
      {
        category: "Quiz",
        expanded: true,
        chats: [
          { name: "Quiz Attempt 1" },
          { name: "Quiz Attempt 2" }
        ]
      }
    ];
    localStorage.setItem('kkh-grouped-sessions', JSON.stringify(groupedSessions));
  }

  // Load sessions or fallback
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
    const generalList = document.getElementById('general-sessions');
    const quizList = document.getElementById('quiz-sessions');
    if (!generalList || !quizList) return;

    generalList.innerHTML = '';
    quizList.innerHTML = '';

    groupedSessions.forEach(group => {
      const target = group.category === 'General' ? generalList : quizList;

      group.chats.forEach((chat, index) => {
        const chatDiv = document.createElement('div');
        chatDiv.className = 'chat-session';

        const nameSpan = document.createElement('span');
        nameSpan.textContent = chat.name;
        nameSpan.style.flex = '1';
        nameSpan.addEventListener('click', () => {
          // You can later use switchSession(group, chat, index)
          alert(`Switched to ${chat.name}`);
        });

        const menu = document.createElement('div');
        menu.className = 'chat-menu';
        menu.textContent = '⋮';

        const dropdown = document.createElement('div');
        dropdown.className = 'chat-dropdown';

        const renameOption = document.createElement('div');
        renameOption.className = 'rename-option';
        renameOption.textContent = 'Rename';
        renameOption.addEventListener('click', () => {
          const newName = prompt('Enter new session name:');
          if (newName) {
            chat.name = newName;
            localStorage.setItem('kkh-grouped-sessions', JSON.stringify(groupedSessions));
            renderSessions();
          }
        });

        const deleteOption = document.createElement('div');
        deleteOption.className = 'delete-option';
        deleteOption.textContent = 'Delete';
        deleteOption.addEventListener('click', () => {
          if (confirm('Delete this session?')) {
            group.chats.splice(index, 1);
            localStorage.setItem('kkh-grouped-sessions', JSON.stringify(groupedSessions));
            renderSessions();
          }
        });

        dropdown.appendChild(renameOption);
        dropdown.appendChild(deleteOption);
        menu.appendChild(dropdown);
        chatDiv.appendChild(nameSpan);
        chatDiv.appendChild(menu);
        target.appendChild(chatDiv);
      });
    });
  }

  // New session buttons
  document.querySelectorAll('.new-session-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const category = btn.getAttribute('data-category');
    const group = groupedSessions.find(g => g.category === category);
    if (!group) return;

    const newChat = {
      name: category === 'Quiz'
        ? `Quiz Attempt ${group.chats.length + 1}`
        : `Chat ${group.chats.length + 1}`
    };

    group.chats.push(newChat);
    localStorage.setItem('kkh-grouped-sessions', JSON.stringify(groupedSessions));
    renderSessions();
  });
});

  document.querySelector('.new-prompt-btn')?.addEventListener('click', () => {
    alert('New Prompt Clicked!');
  });

  // --- Rename / Delete ---
  function attachMenuHandlers() {
    document.querySelectorAll('.rename-option').forEach(btn => {
      btn.addEventListener('click', () => {
        const groupName = btn.getAttribute('data-group');
        const index = btn.getAttribute('data-index');
        const newName = prompt('Enter new session name:');
        if (newName) {
          const group = groupedSessions.find(g => g.category === groupName);
          group.chats[index].name = newName;
          localStorage.setItem('kkh-grouped-sessions', JSON.stringify(groupedSessions));
          renderSessions();
        }
      });
    });

    document.querySelectorAll('.delete-option').forEach(btn => {
      btn.addEventListener('click', () => {
        const groupName = btn.getAttribute('data-group');
        const index = btn.getAttribute('data-index');
        if (confirm('Are you sure you want to delete this session?')) {
          const group = groupedSessions.find(g => g.category === groupName);
          group.chats.splice(index, 1);
          localStorage.setItem('kkh-grouped-sessions', JSON.stringify(groupedSessions));
          renderSessions();
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
