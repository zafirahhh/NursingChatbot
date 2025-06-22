// KKH Nursing Chatbot - Final Grouped Sessions App.js (MCQ with radio, explanation, 5 options)

const BACKEND_URL_FINAL = "http://127.0.0.1:8000/ask";
const QUIZ_URL_FINAL = "http://127.0.0.1:8000/quiz";
const QUIZ_EVAL_URL_FINAL = "http://127.0.0.1:8000/quiz/evaluate";

document.addEventListener('DOMContentLoaded', () => {
  const chatWindow = document.getElementById('chat-window');
  const chatForm = document.getElementById('chat-form');
  const userInput = document.getElementById('user-input');
  const clearChatBtn = document.getElementById('clear-chat');
  const micBtn = document.getElementById('mic-btn');
  const chatSessionsList = document.getElementById('chat-sessions');

  const avatars = { user: '👩', bot: '🤖' };

  let groupedSessions = JSON.parse(localStorage.getItem('kkh-grouped-sessions') || JSON.stringify([
    {
      category: "General",
      expanded: true,
      chats: [ { name: "Welcome", id: "general-welcome" } ]
    },
    {
      category: "Quiz",
      expanded: true,
      chats: []
    }
  ]));

  let activeSessionId = localStorage.getItem('kkh-active-session') || 'general-welcome';
  let currentQuiz = [];
  let quizAnswers = {};

  function renderGroupedSessions() {
    chatSessionsList.innerHTML = '';

    groupedSessions.forEach(group => {
      const groupHeader = document.createElement('div');
      groupHeader.className = 'chat-session-group';
      groupHeader.style.fontWeight = 'bold';
      groupHeader.style.cursor = 'pointer';
      groupHeader.style.padding = '6px 10px';
      groupHeader.innerHTML = `${group.expanded ? '▼' : '▶'} ${group.category}`;
      groupHeader.onclick = () => {
        group.expanded = !group.expanded;
        saveGroupedSessions();
        renderGroupedSessions();
      };
      chatSessionsList.appendChild(groupHeader);

      if (group.expanded) {
        group.chats.forEach(session => {
          const li = document.createElement('div');
          li.className = 'chat-session-row' + (session.id === activeSessionId ? ' active' : '');
          li.style.display = 'flex';
          li.style.alignItems = 'center';
          li.style.padding = '4px 14px';
          li.style.cursor = 'pointer';
          li.style.borderRadius = '8px';
          li.style.marginLeft = '12px';
          li.style.marginBottom = '2px';
          li.style.fontSize = '14px';
          li.title = session.name;
          li.textContent = session.name.length > 32 ? session.name.slice(0, 30) + '...' : session.name;
          li.onclick = () => switchGroupedSession(session.id);
          chatSessionsList.appendChild(li);
        });

        if (group.category === 'Quiz') {
          const addQuizBtn = document.createElement('button');
          addQuizBtn.textContent = '+ New Quiz';
          addQuizBtn.className = 'add-session-btn';
          addQuizBtn.style.marginLeft = '12px';
          addQuizBtn.onclick = () => {
            const quizGroup = groupedSessions.find(g => g.category === 'Quiz');
            const id = 'quiz-' + Date.now();
            const name = `Quiz Attempt ${quizGroup.chats.length + 1}`;
            quizGroup.chats.push({ name, id });
            saveGroupedSessions();
            renderGroupedSessions();
            switchGroupedSession(id);
          };
          chatSessionsList.appendChild(addQuizBtn);
        }
      }
    });
  }

  function saveGroupedSessions() {
    localStorage.setItem('kkh-grouped-sessions', JSON.stringify(groupedSessions));
  }

  function loadGroupedHistory() {
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
    const key = 'kkh-chat-history-' + activeSessionId;
    const history = JSON.parse(localStorage.getItem(key) || '[]');
    history.push({ sender, text });
    localStorage.setItem(key, JSON.stringify(history));
  }

  async function switchGroupedSession(sessionId) {
    activeSessionId = sessionId;
    localStorage.setItem('kkh-active-session', sessionId);
    renderGroupedSessions();
    loadGroupedHistory();

    if (sessionId.startsWith('quiz')) {
      const response = await fetch(`${QUIZ_URL_FINAL}?n=5`);
      const data = await response.json();
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
    }
  }

  renderGroupedSessions();
  loadGroupedHistory();
});
