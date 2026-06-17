
const state = {
    currentTab: 'chat',
    chatHistory: [],
    topicsPage: 1,
    topicsPerPage: 20,
    totalTopics: 0,
    isLoading: false,
};

// ============== DOM Elements ==============
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const elements = {
    chatInput: $('#chat-input'),
    sendBtn: $('#send-btn'),
    chatMessages: $('#chat-messages'),
    welcomeScreen: $('#welcome-screen'),
    suggestionGrid: $('#suggestion-grid'),
    personasContainer: $('#personas-container'),
    topicsList: $('#topics-list'),
    topicSearch: $('#topic-search'),
    prevPage: $('#prev-page'),
    nextPage: $('#next-page'),
    pageInfo: $('#page-info'),
    systemStatsGrid: $('#system-stats-grid'),
    mobileMenuBtn: $('#mobile-menu-btn'),
    sidebar: $('#sidebar'),
};

// ============== API Client ==============
const api = {
    async chat(question) {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });
        return res.json();
    },

    async getPersonaSummary() {
        const res = await fetch('/api/persona/summary');
        return res.json();
    },

    async getPersona(user) {
        const res = await fetch(`/api/persona?user=${encodeURIComponent(user)}`);
        return res.json();
    },

    async getTopics(page = 1, perPage = 20) {
        const res = await fetch(`/api/topics?page=${page}&per_page=${perPage}`);
        return res.json();
    },

    async getStats() {
        const res = await fetch('/api/stats');
        return res.json();
    },

    async getSuggestions() {
        const res = await fetch('/api/suggestions');
        return res.json();
    },
};

// ============== Tab Navigation ==============
function switchTab(tabName) {
    state.currentTab = tabName;

    // Update nav items
    $$('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.tab === tabName);
    });

    // Update tab content
    $$('.tab-content').forEach(tab => {
        tab.classList.toggle('active', tab.id === `tab-${tabName}`);
    });

    // Load tab data
    if (tabName === 'personas') loadPersonas();
    if (tabName === 'topics') loadTopics();
    if (tabName === 'system') loadSystemStats();
}

// ============== Chat Functions ==============
function addMessage(role, content, sources = null) {
    // Hide welcome screen
    if (elements.welcomeScreen) {
        elements.welcomeScreen.style.display = 'none';
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? 'U' : 'AI';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    // Convert markdown-like formatting to HTML
    contentDiv.innerHTML = formatMessage(content);

    // Add source badges for bot messages
    if (role === 'bot' && sources) {
        const sourceDiv = document.createElement('div');
        sourceDiv.className = 'source-info';

        if (sources.topics_retrieved > 0) {
            sourceDiv.innerHTML += `<span class="source-badge">📋 ${sources.topics_retrieved} topics</span>`;
        }
        if (sources.messages_retrieved > 0) {
            sourceDiv.innerHTML += `<span class="source-badge">💬 ${sources.messages_retrieved} msg chunks</span>`;
        }
        if (sources.checkpoints_retrieved > 0) {
            sourceDiv.innerHTML += `<span class="source-badge">📌 ${sources.checkpoints_retrieved} checkpoints</span>`;
        }
        contentDiv.appendChild(sourceDiv);
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    elements.chatMessages.appendChild(messageDiv);

    // Scroll to bottom
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function addTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'message bot';
    indicator.id = 'typing-indicator';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = 'AI';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;

    indicator.appendChild(avatar);
    indicator.appendChild(contentDiv);
    elements.chatMessages.appendChild(indicator);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function removeTypingIndicator() {
    const indicator = $('#typing-indicator');
    if (indicator) indicator.remove();
}

function formatMessage(text) {
    if (!text) return '';

    // Escape HTML first
    let html = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Headers
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Italic
    html = html.replace(/_(.+?)_/g, '<em>$1</em>');

    // Horizontal rules
    html = html.replace(/^---$/gm, '<hr>');

    // Lists
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

    // Line breaks
    html = html.replace(/\n\n/g, '<br><br>');
    html = html.replace(/\n/g, '<br>');

    // Score bars (custom format from persona)
    html = html.replace(/\[([█░]+)\]/g, '<span style="font-family: monospace; color: var(--accent-violet); letter-spacing: -1px;">$1</span>');

    return html;
}

async function sendMessage() {
    const question = elements.chatInput.value.trim();
    if (!question || state.isLoading) return;

    state.isLoading = true;
    elements.sendBtn.disabled = true;
    elements.chatInput.value = '';
    elements.chatInput.style.height = 'auto';

    // Add user message
    addMessage('user', question);

    // Show typing indicator
    addTypingIndicator();

    try {
        const result = await api.chat(question);
        removeTypingIndicator();

        if (result.success) {
            addMessage('bot', result.answer, result.sources);
        } else {
            addMessage('bot', `⚠️ Error: ${result.error || 'Unknown error occurred'}`);
        }
    } catch (error) {
        removeTypingIndicator();
        addMessage('bot', `⚠️ Connection error: ${error.message}. Make sure the server is running.`);
    }

    state.isLoading = false;
    elements.sendBtn.disabled = !elements.chatInput.value.trim();
}

// ============== Personas ==============
async function loadPersonas() {
    try {
        const result = await api.getPersonaSummary();
        if (!result.success) return;

        elements.personasContainer.innerHTML = '';

        Object.entries(result.summaries).forEach(([userId, persona]) => {
            const card = createPersonaCard(userId, persona);
            elements.personasContainer.appendChild(card);
        });
    } catch (error) {
        elements.personasContainer.innerHTML = `
            <div class="loading-state">
                <p>⚠️ Failed to load personas: ${error.message}</p>
            </div>
        `;
    }
}

function createPersonaCard(userId, persona) {
    const card = document.createElement('div');
    card.className = 'persona-card';

    const userClass = userId.includes('1') ? 'user1' : 'user2';
    const userInitial = userId.includes('1') ? 'U1' : 'U2';

    let traitsHTML = '';
    if (persona.top_traits && persona.top_traits.length > 0) {
        traitsHTML = persona.top_traits.map(t => `
            <div class="trait-bar-container">
                <div class="trait-name">
                    <span>${t.trait}</span>
                    <span>${Math.round(t.score * 100)}%</span>
                </div>
                <div class="trait-bar">
                    <div class="trait-bar-fill" style="width: ${t.score * 100}%"></div>
                </div>
            </div>
        `).join('');
    }

    let habitsHTML = '';
    if (persona.top_habits && persona.top_habits.length > 0) {
        habitsHTML = persona.top_habits.map(h =>
            `<span class="fact-tag">🔄 ${h.category}: ${h.detail.substring(0, 60)}</span>`
        ).join('');
    }

    let factsHTML = '';
    if (persona.top_facts && persona.top_facts.length > 0) {
        factsHTML = persona.top_facts.map(f =>
            `<span class="fact-tag">📌 ${f.category}: ${f.detail.substring(0, 60)}</span>`
        ).join('');
    }

    const style = persona.communication_style || {};
    const styleHTML = `
        <div class="style-grid">
            <div class="style-item">
                <div class="style-item-label">Msg Style</div>
                <div class="style-item-value">${style.message_length_style || 'N/A'}</div>
            </div>
            <div class="style-item">
                <div class="style-item-label">Avg Words</div>
                <div class="style-item-value">${style.avg_message_length_words || 'N/A'}</div>
            </div>
            <div class="style-item">
                <div class="style-item-label">Formality</div>
                <div class="style-item-value">${style.formality?.overall_formality || 'N/A'}</div>
            </div>
            <div class="style-item">
                <div class="style-item-label">Enthusiasm</div>
                <div class="style-item-value">${style.punctuation?.enthusiasm_level || 'N/A'}</div>
            </div>
        </div>
    `;

    card.innerHTML = `
        <div class="persona-header">
            <div class="persona-avatar ${userClass}">${userInitial}</div>
            <div>
                <div class="persona-name">${userId}</div>
                <div class="persona-msg-count">${(persona.message_count || 0).toLocaleString()} messages analyzed</div>
            </div>
        </div>

        ${traitsHTML ? `
            <div class="persona-section">
                <div class="persona-section-title">🎭 Personality Traits</div>
                ${traitsHTML}
            </div>
        ` : ''}

        ${habitsHTML ? `
            <div class="persona-section">
                <div class="persona-section-title">🔄 Habits</div>
                <div>${habitsHTML}</div>
            </div>
        ` : ''}

        ${factsHTML ? `
            <div class="persona-section">
                <div class="persona-section-title">📌 Personal Facts</div>
                <div>${factsHTML}</div>
            </div>
        ` : ''}

        <div class="persona-section">
            <div class="persona-section-title">💬 Communication Style</div>
            ${styleHTML}
        </div>
    `;

    return card;
}

// ============== Topics ==============
async function loadTopics(page = 1) {
    try {
        state.topicsPage = page;
        const result = await api.getTopics(page, state.topicsPerPage);

        if (!result.success) return;

        state.totalTopics = result.total;
        elements.topicsList.innerHTML = '';

        result.topics.forEach(topic => {
            const card = createTopicCard(topic);
            elements.topicsList.appendChild(card);
        });

        // Update pagination
        elements.pageInfo.textContent = `Page ${result.page} of ${result.total_pages}`;
        elements.prevPage.disabled = page <= 1;
        elements.nextPage.disabled = page >= result.total_pages;
    } catch (error) {
        elements.topicsList.innerHTML = `
            <div class="loading-state">
                <p>⚠️ Failed to load topics: ${error.message}</p>
            </div>
        `;
    }
}

function createTopicCard(topic) {
    const card = document.createElement('div');
    card.className = 'topic-card';

    let entitiesHTML = '';
    if (topic.key_entities && topic.key_entities.length > 0) {
        entitiesHTML = `
            <div class="topic-entities">
                ${topic.key_entities.map(e => `<span class="entity-tag">${e}</span>`).join('')}
            </div>
        `;
    }

    card.innerHTML = `
        <div class="topic-card-header">
            <span class="topic-id">Topic #${topic.topic_id}</span>
            <span class="topic-range">msgs ${topic.start_msg_index}–${topic.end_msg_index} (${topic.message_count} msgs)</span>
        </div>
        <div class="topic-label">${topic.topic_label}</div>
        <div class="topic-summary">${escapeHtml(topic.summary || '')}</div>
        ${entitiesHTML}
    `;

    // Click to expand summary
    card.addEventListener('click', () => {
        const summary = card.querySelector('.topic-summary');
        summary.classList.toggle('expanded');
    });

    return card;
}

// ============== System Stats ==============
async function loadSystemStats() {
    try {
        const result = await api.getStats();
        if (!result.success) return;

        const stats = result.stats;
        elements.systemStatsGrid.innerHTML = '';

        const statItems = [
            { label: 'Total Messages', value: stats.total_messages?.toLocaleString() || '0' },
            { label: 'Topic Checkpoints', value: stats.total_topic_checkpoints?.toLocaleString() || '0' },
            { label: '100-Msg Checkpoints', value: stats.total_100msg_checkpoints?.toLocaleString() || '0' },
            { label: 'Message Chunks', value: stats.total_message_chunks?.toLocaleString() || '0' },
            { label: 'Topic Index Size', value: stats.topic_index_size?.toLocaleString() || '0' },
            { label: 'Message Index Size', value: stats.message_index_size?.toLocaleString() || '0' },
        ];

        statItems.forEach(item => {
            const card = document.createElement('div');
            card.className = 'system-stat-card';
            card.innerHTML = `
                <div class="system-stat-value">${item.value}</div>
                <div class="system-stat-label">${item.label}</div>
            `;
            elements.systemStatsGrid.appendChild(card);
        });

        // Update sidebar stats
        $('#stat-messages').textContent = formatNumber(stats.total_messages || 0);
        $('#stat-topics').textContent = formatNumber(stats.total_topic_checkpoints || 0);
        $('#stat-checkpoints').textContent = formatNumber(stats.total_100msg_checkpoints || 0);
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

// ============== Suggestions ==============
async function loadSuggestions() {
    try {
        const result = await api.getSuggestions();
        if (!result.success) return;

        elements.suggestionGrid.innerHTML = '';

        result.suggestions.forEach(category => {
            category.queries.slice(0, 3).forEach(query => {
                const card = document.createElement('div');
                card.className = 'suggestion-card';
                card.innerHTML = `
                    <div class="suggestion-category">${category.category}</div>
                    <div>${query}</div>
                `;
                card.addEventListener('click', () => {
                    elements.chatInput.value = query;
                    elements.sendBtn.disabled = false;
                    sendMessage();
                });
                elements.suggestionGrid.appendChild(card);
            });
        });
    } catch (error) {
        // Fallback suggestions
        const fallback = [
            { cat: 'Persona', q: "What kind of person is User 1?" },
            { cat: 'Persona', q: "What are User 2's habits?" },
            { cat: 'Style', q: "How does User 1 talk?" },
            { cat: 'Topics', q: "What are the main topics discussed?" },
            { cat: 'Facts', q: "What hobbies are mentioned?" },
            { cat: 'Facts', q: "What pets do users have?" },
        ];

        elements.suggestionGrid.innerHTML = '';
        fallback.forEach(item => {
            const card = document.createElement('div');
            card.className = 'suggestion-card';
            card.innerHTML = `
                <div class="suggestion-category">${item.cat}</div>
                <div>${item.q}</div>
            `;
            card.addEventListener('click', () => {
                elements.chatInput.value = item.q;
                elements.sendBtn.disabled = false;
                sendMessage();
            });
            elements.suggestionGrid.appendChild(card);
        });
    }
}

// ============== Utilities ==============
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatNumber(num) {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
}

// ============== Event Listeners ==============
function initializeEvents() {
    // Navigation
    $$('.nav-item').forEach(item => {
        item.addEventListener('click', () => switchTab(item.dataset.tab));
    });

    // Chat input
    elements.chatInput.addEventListener('input', () => {
        elements.sendBtn.disabled = !elements.chatInput.value.trim();
        // Auto-resize textarea
        elements.chatInput.style.height = 'auto';
        elements.chatInput.style.height = Math.min(elements.chatInput.scrollHeight, 120) + 'px';
    });

    elements.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    elements.sendBtn.addEventListener('click', sendMessage);

    // Pagination
    elements.prevPage.addEventListener('click', () => {
        if (state.topicsPage > 1) loadTopics(state.topicsPage - 1);
    });

    elements.nextPage.addEventListener('click', () => {
        loadTopics(state.topicsPage + 1);
    });

    // Topic search
    let searchTimeout;
    elements.topicSearch.addEventListener('input', () => {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            const query = elements.topicSearch.value.toLowerCase();
            $$('.topic-card').forEach(card => {
                const text = card.textContent.toLowerCase();
                card.style.display = text.includes(query) ? '' : 'none';
            });
        }, 300);
    });

    // Mobile menu
    elements.mobileMenuBtn.addEventListener('click', () => {
        elements.sidebar.classList.toggle('open');
    });

    // Close sidebar on outside click (mobile)
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768 &&
            !elements.sidebar.contains(e.target) &&
            !elements.mobileMenuBtn.contains(e.target)) {
            elements.sidebar.classList.remove('open');
        }
    });
}

// ============== Initialize ==============
document.addEventListener('DOMContentLoaded', () => {
    initializeEvents();
    loadSuggestions();
    loadSystemStats();
});
