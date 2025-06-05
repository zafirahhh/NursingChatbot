// KKH Nursing Chatbot - Basic Chat Interactivity

document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.getElementById('chat-window');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const clearChatBtn = document.getElementById('clear-chat');
    const micBtn = document.getElementById('mic-btn');

    // Avatars
    const avatars = {
        user: '👩',
        bot: '🤖'
    };

    // Load chat history from localStorage
    function loadHistory() {
        const history = JSON.parse(localStorage.getItem('kkh-chat-history') || '[]');
        if (history.length === 0) {
            // Show welcome message if no history
            appendMessage('bot', 'Hello! I am your KKH Nursing Chatbot. How can I assist you today?');
        } else {
            history.forEach(msg => appendMessage(msg.sender, msg.text, false));
        }
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

    let knowledgeText = '';

    // Load the knowledge base from the PDF (as text)
    fetch('data/nursing_guide.txt')
      .then(response => response.text())
      .then(text => {
        knowledgeText = text;
      });

    function searchKnowledge(query) {
      if (!knowledgeText) return 'Knowledge base not loaded yet.';
      const lines = knowledgeText.split('\n');
      const keywords = query.toLowerCase().split(/\s+/);
      // Expanded parameter synonyms
      const paramMap = {
        'heart rate': ['heart rate', 'pulse', 'pr'],
        'blood pressure': ['blood pressure', 'bp'],
        'respiratory rate': ['respiratory rate', 'rr'],
        'temperature': ['temperature', 'temp'],
        'weight': ['weight'],
        'height': ['height'],
        'spo2': ['spo2', 'oxygen saturation'],
        'glucose': ['glucose', 'sugar'],
        'bilirubin': ['bilirubin', 'jaundice'],
        'urine': ['urine', 'output', 'input', 'fluid'],
      };
      // Try to extract age and parameter from the query for table-aware search
      const ageMatch = query.match(/(\d+)[ -]?(year|month|day|week)s?/i);
      let age = ageMatch ? ageMatch[0] : null;
      let param = null;
      for (const key in paramMap) {
        if (paramMap[key].some(p => new RegExp(`\\b${p}\\b`, 'i').test(query))) {
          param = key;
          break;
        }
      }
      // Table-aware extraction with improved age group matching
      if (param && age) {
        for (let i = 0; i < lines.length; i++) {
          if (paramMap[param].some(p => new RegExp(`\\b${p}\\b`, 'i').test(lines[i]))) {
            // Look for a table structure below
            for (let j = i + 1; j < Math.min(i + 15, lines.length); j++) {
              // Try to match age group (e.g., 2 years matches 1-3 years)
              const row = lines[j].toLowerCase();
              const ageNum = parseInt(age);
              if (row.match(/(\d+)[-–](\d+)/)) {
                const [, min, max] = row.match(/(\d+)[-–](\d+)/);
                if (ageNum >= parseInt(min) && ageNum <= parseInt(max)) {
                  const header = lines[i];
                  return `Relevant table for ${param} (age: ${age}):\n${header}\n${lines[j]}`;
                }
              } else if (row.includes(age.replace(/s?$/,''))) {
                const header = lines[i];
                return `Relevant table for ${param} (age: ${age}):\n${header}\n${lines[j]}`;
              }
            }
          }
        }
      }
      // Fuzzy matching helper with word boundaries
      function fuzzyIncludes(line, word) {
        return new RegExp(`\\b${word.replace(/[^a-z0-9]/g, '')}\\b`, 'i').test(line);
      }
      // Try to find a line that contains all keywords (fuzzy)
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].toLowerCase();
        if (keywords.every(word => fuzzyIncludes(line, word))) {
          // Return this line and the next 2 lines for context
          return lines.slice(i, i + 3).join('\n');
        }
      }
      // Fallback: return the first line that contains any keyword (fuzzy)
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].toLowerCase();
        if (keywords.some(word => fuzzyIncludes(line, word))) {
          return lines.slice(i, i + 3).join('\n');
        }
      }
      return 'No relevant information found in the nursing guide.';
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
            // Use the knowledge base for the bot reply
            const botReply = searchKnowledge(userText);
            appendMessage('bot', botReply);
        }, 900);
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
    loadHistory();
});
