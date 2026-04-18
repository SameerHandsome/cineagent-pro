/* frontend/js/chat.js — CineAgent Pro chat interface */

const API   = "http://localhost:8000";
const token = localStorage.getItem("cineagent_token");

if (!token) window.location.href = "index.html";

let currentSessionId = null;
let isStreaming      = false;

// ── FIX: Replace isFirstMessage boolean with a Set of session IDs that already
// have a real title. A boolean was unreliable because loadSession() reset it
// to true whenever the DB title was "Untitled Project" (e.g. after a page
// reload or sidebar click before the PATCH had completed). A Set keyed by
// session ID persists across loadSession() calls and can never be reset by
// sidebar re-renders or loadSessions() polling.
const titledSessions = new Set();

// ── Markdown renderer ─────────────────────────────────────────────────────────
function renderMarkdown(text) {
  return text
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm, "<h2>$1</h2>")
    .replace(/^# (.+)$/gm, "<h1>$1</h1>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, "<code>$1</code>")
    .replace(/^---$/gm, "<hr>")
    .replace(/^\| (.+) \|$/gm, (_, cells) => {
      const tds = cells.split(" | ").map((c) => `<td>${c}</td>`).join("");
      return `<tr>${tds}</tr>`;
    })
    .replace(/(<tr>.*<\/tr>\n?)+/g, (rows) => `<table>${rows}</table>`)
    .replace(/^\d+\. (.+)$/gm, "<li>$1</li>")
    .replace(/(<li>.*<\/li>\n?)+/g, (items) => `<ol>${items}</ol>`)
    .replace(/^- (.+)$/gm, "<li>$1</li>")
    .replace(/(<li>.*<\/li>\n?)+/g, (items) => `<ul>${items}</ul>`)
    .replace(/\n\n/g, "</p><p>")
    .replace(/^(?!<[h|l|t|u|o|p])/gm, "")
    .replace(/([^>])\n([^<])/g, "$1<br>$2");
}

// ── Auth header ───────────────────────────────────────────────────────────────
function authHeaders(extra = {}) {
  return { Authorization: `Bearer ${token}`, "Content-Type": "application/json", ...extra };
}

// ── Generate a meaningful session title from the user's first message ─────────
function generateTitle(text) {
  if (!text || !text.trim()) return "Untitled Project";
  const words = text.trim().split(/\s+/).slice(0, 6).join(" ");
  const cleaned = words.replace(/[^\w\s'&-]/g, "").trim();
  return cleaned.length > 40 ? cleaned.slice(0, 40) + "…" : cleaned || "Untitled Project";
}

// ── Update session title on backend ──────────────────────────────────────────
async function updateSessionTitle(sessionId, title) {
  try {
    await fetch(`${API}/api/sessions/${sessionId}`, {
      method:  "PATCH",
      headers: authHeaders(),
      body:    JSON.stringify({ title }),
    });
  } catch {
    console.warn("Could not update session title.");
  }
}

// ── Session list ──────────────────────────────────────────────────────────────
async function loadSessions() {
  try {
    const res = await fetch(`${API}/api/sessions`, { headers: authHeaders() });
    if (res.status === 401) { logout(); return; }
    const sessions = await res.json();
    renderSessionList(sessions);
  } catch {
    console.warn("Could not load sessions.");
  }
}

function renderSessionList(sessions) {
  const list = document.getElementById("session-list");
  list.innerHTML = "";

  if (!sessions.length) {
    list.innerHTML = '<li style="padding:12px;font-size:13px;color:var(--text-muted)">No projects yet</li>';
    return;
  }

  sessions.forEach((s) => {
    const isActive = s.id === currentSessionId;
    const li = document.createElement("li");
    li.className = "session-item" + (isActive ? " active" : "");
    li.dataset.sessionId = s.id;

    const displayTitle = s.title || "Untitled Project";

    // FIX: Only sync the header from DB title if the session is in titledSessions
    // (meaning we already set a real title). If DB still shows "Untitled Project"
    // but we've locally set a title, keep the local one — don't overwrite it.
    if (isActive) {
      if (titledSessions.has(s.id) && displayTitle !== "Untitled Project") {
        document.getElementById("session-title").textContent = displayTitle;
      }
    }

    const nameSpan = document.createElement("span");
    nameSpan.className = "session-name";
    nameSpan.textContent = displayTitle;
    nameSpan.title = displayTitle;

    const delBtn = document.createElement("button");
    delBtn.className = "session-delete-btn";
    delBtn.title = "Delete project";
    delBtn.innerHTML = "&#x2715;";
    delBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete "${displayTitle}"?`)) return;
      try {
        const res = await fetch(`${API}/api/sessions/${s.id}`, {
          method: "DELETE",
          headers: authHeaders(),
        });
        if (res.ok) {
          // Clean up the Set so deleted session doesn't linger
          titledSessions.delete(s.id);
          if (currentSessionId === s.id) {
            currentSessionId = null;
            document.getElementById("session-title").textContent = "New Project";
            document.getElementById("messages-container").innerHTML = `
              <div class="welcome-screen" id="welcome-screen">
                <div class="welcome-icon">🎬</div>
                <h2>Start a new project</h2>
                <p>Describe your concept to get started.</p>
              </div>`;
          }
          await loadSessions();
        } else {
          const err = await res.json().catch(() => ({}));
          alert(`Could not delete session: ${err.detail || res.status}`);
        }
      } catch {
        console.warn("Could not delete session:", s.id);
      }
    });

    li.appendChild(nameSpan);
    li.appendChild(delBtn);
    li.addEventListener("click", () => loadSession(s.id, displayTitle));
    list.appendChild(li);
  });
}

async function loadSession(sessionId, title) {
  currentSessionId = sessionId;

  // FIX: Never reset title state from DB here. If DB title is "Untitled Project"
  // but titledSessions already has this ID, the user already named it locally —
  // we don't want the next message to re-PATCH with the follow-up text.
  // titledSessions is the source of truth, not the DB title string.
  const hasRealTitle = title && title !== "Untitled Project";
  if (hasRealTitle) {
    titledSessions.add(sessionId); // DB confirmed a real title exists
  }

  document.getElementById("session-title").textContent = title || "New Project";
  document.getElementById("welcome-screen")?.remove();

  const container = document.getElementById("messages-container");
  container.innerHTML = "";

  try {
    const res = await fetch(`${API}/api/sessions/${sessionId}/messages`, { headers: authHeaders() });
    const messages = await res.json();
    messages.forEach((m) => appendMessage(m.role, m.content, false));
    container.scrollTop = container.scrollHeight;
  } catch {
    console.warn("Could not load messages for session:", sessionId);
  }

  await loadSessions();
}

// ── New session ───────────────────────────────────────────────────────────────
document.getElementById("new-session-btn").addEventListener("click", () => {
  currentSessionId = null;
  document.getElementById("session-title").textContent = "New Project";
  document.getElementById("messages-container").innerHTML = `
    <div class="welcome-screen" id="welcome-screen">
      <div class="welcome-icon">🎬</div>
      <h2>Start a new project</h2>
      <p>Paste a logline, treatment, or concept. CineAgent will handle script analysis, budget estimation, casting suggestions, and market intelligence.</p>
      <div class="example-prompts">
        <button class="example-pill" data-prompt="A rogue AI on a deep space mining vessel begins making autonomous decisions to protect the crew, even when those decisions conflict with the mission.">
          🚀 Sci-fi AI concept
        </button>
        <button class="example-pill" data-prompt="What would a 30-day indie shoot cost with SAG actors for a contained horror thriller?">
          💰 Budget question
        </button>
        <button class="example-pill" data-prompt="Who are the best casting options for a $4M psychological thriller with a strong female lead?">
          🎭 Casting query
        </button>
        <button class="example-pill" data-prompt="How did comparable micro-budget horror films perform at box office in the last 3 years?">
          📊 Market research
        </button>
      </div>
    </div>`;
  document.querySelectorAll(".session-item").forEach(el => el.classList.remove("active"));
});

// ── Append message ─────────────────────────────────────────────────────────────
function appendMessage(role, content, streaming = false) {
  const welcome = document.getElementById("welcome-screen");
  if (welcome) welcome.remove();

  const container = document.getElementById("messages-container");

  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";

  if (role === "assistant") {
    bubble.innerHTML = streaming ? "" : renderMarkdown(content);
    if (streaming) bubble.classList.add("typing-cursor");
  } else {
    bubble.textContent = content;
  }

  const msgActions = document.createElement("div");
  msgActions.className = "msg-actions";

  if (!streaming) {
    const dotsBtn = document.createElement("button");
    dotsBtn.className = "msg-dots-btn";
    dotsBtn.innerHTML = "&#8943;";
    dotsBtn.title = "Message actions";

    const dropdown = document.createElement("div");
    dropdown.className = "msg-dropdown hidden";

    const copyBtn = document.createElement("button");
    copyBtn.textContent = "Copy";
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(content).catch(() => {});
      dropdown.classList.add("hidden");
    });

    dropdown.appendChild(copyBtn);
    dotsBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      document.querySelectorAll(".msg-dropdown").forEach(d => d.classList.add("hidden"));
      dropdown.classList.toggle("hidden");
    });

    msgActions.appendChild(dotsBtn);
    msgActions.appendChild(dropdown);
  }

  if (role === "user") {
    row.appendChild(msgActions);
    row.appendChild(bubble);
  } else {
    row.appendChild(bubble);
    row.appendChild(msgActions);
  }

  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
  return bubble;
}

// ── Agent status pills ─────────────────────────────────────────────────────────
const AGENT_PILL_MAP = {
  "AGENT:SCRIPT_ANALYST":   "pill-script",
  "AGENT:BUDGET_PLANNER":   "pill-budget",
  "AGENT:CASTING_DIRECTOR": "pill-casting",
  "AGENT:MARKET_INTEL":     "pill-market",
};

function setAgentActive(agentKey) {
  const pillId = AGENT_PILL_MAP[agentKey];
  if (!pillId) return;
  const pill = document.getElementById(pillId);
  if (pill) {
    pill.classList.add("active");
    setTimeout(() => {
      pill.classList.remove("active");
      pill.classList.add("complete");
    }, 1500);
  }
}

function resetAgentPills() {
  Object.values(AGENT_PILL_MAP).forEach((id) => {
    const pill = document.getElementById(id);
    if (pill) pill.classList.remove("active", "complete");
  });
}

// ── Send message ──────────────────────────────────────────────────────────────
async function sendMessage(text) {
  if (isStreaming || !text.trim()) return;
  isStreaming = true;

  // ── STEP 1: Pre-create session with real title on the very first send ──────
  // Only fires when currentSessionId is null (genuinely new session).
  // titledSessions.has(id) gates all subsequent title logic — once a session
  // is in the Set, no follow-up message can ever trigger a PATCH.
  let titleWasSet = false;
  if (!currentSessionId) {
    const titleForNew = generateTitle(text);
    try {
      const res = await fetch(`${API}/api/sessions`, {
        method:  "POST",
        headers: authHeaders(),
        body:    JSON.stringify({ title: titleForNew }),
      });
      const s = await res.json();
      currentSessionId = s.id;
      titledSessions.add(currentSessionId); // mark as titled immediately
      titleWasSet = true;
      document.getElementById("session-title").textContent = titleForNew;
    } catch {
      console.warn("Could not pre-create session; will patch title after [SESSION:] arrives.");
    }
  }

  // ── STEP 2: If session exists but title not yet set, PATCH now ────────────
  // This handles the case where pre-create failed (Step 1 catch) and we're
  // now sending the first message with a backend-generated session_id.
  // titledSessions.has() ensures this fires AT MOST ONCE per session ever.
  if (!titleWasSet && currentSessionId && !titledSessions.has(currentSessionId)) {
    const title = generateTitle(text);
    document.getElementById("session-title").textContent = title;
    await updateSessionTitle(currentSessionId, title);
    titledSessions.add(currentSessionId); // mark as titled — never PATCH again
    titleWasSet = true;
  }

  const input   = document.getElementById("message-input");
  const sendBtn = document.getElementById("send-btn");
  input.value   = "";
  input.style.height = "auto";
  sendBtn.disabled   = true;
  updateCharCount(0);

  const statusBar = document.getElementById("agent-status-bar");
  statusBar.classList.remove("hidden");
  resetAgentPills();

  appendMessage("user", text);
  const assistantBubble = appendMessage("assistant", "", true);
  let reportBuffer = "";

  try {
    const res = await fetch(`${API}/api/chat`, {
      method:  "POST",
      headers: authHeaders(),
      body:    JSON.stringify({ message: text, session_id: currentSessionId }),
    });

    if (res.status === 429) {
      assistantBubble.classList.remove("typing-cursor");
      assistantBubble.textContent = "⚠️ Rate limit reached. Please wait a moment.";
      return;
    }
    if (!res.ok) {
      assistantBubble.classList.remove("typing-cursor");
      assistantBubble.textContent = "⚠️ Error connecting to CineAgent. Is the backend running?";
      return;
    }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split("\n");

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6);

        if (data === "[DONE]") break;

        if (data.startsWith("[ERROR]")) {
          assistantBubble.classList.remove("typing-cursor");
          assistantBubble.innerHTML = `<span style="color:var(--danger)">⚠️ ${data}</span>`;
          return;
        }

        // Handle [SESSION:<uuid>] — intercept and NEVER render in the bubble.
        // Only acts if title wasn't already set (Step 1 pre-create failed path).
        if (data.startsWith("[SESSION:")) {
          const backendSid = data.slice(9, -1);
          if (!currentSessionId) {
            currentSessionId = backendSid;
          }
          // Only PATCH if this session hasn't been titled yet
          if (!titleWasSet && !titledSessions.has(currentSessionId)) {
            const title = generateTitle(text);
            document.getElementById("session-title").textContent = title;
            await updateSessionTitle(currentSessionId, title);
            titledSessions.add(currentSessionId);
            titleWasSet = true;
          }
          continue; // never falls through to reportBuffer
        }

        if (data.startsWith("[AGENT:")) {
          setAgentActive(data.slice(1, -1));
          continue;
        }

        reportBuffer += data;
        assistantBubble.innerHTML = renderMarkdown(reportBuffer);
        assistantBubble.scrollIntoView({ behavior: "smooth", block: "end" });
      }
    }
  } catch (err) {
    assistantBubble.classList.remove("typing-cursor");
    assistantBubble.textContent = "⚠️ Stream interrupted. Please try again.";
    console.error("Stream error:", err);
  } finally {
    assistantBubble.classList.remove("typing-cursor");
    statusBar.classList.add("hidden");
    isStreaming      = false;
    sendBtn.disabled = false;
    input.focus();
    await loadSessions();
  }
}

// ── Input handling ─────────────────────────────────────────────────────────────
const input   = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");

function updateCharCount(len) {
  document.getElementById("char-count").textContent = `${len} / 4000`;
}

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
  const len = input.value.length;
  sendBtn.disabled = len === 0 || isStreaming;
  updateCharCount(len);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) sendMessage(input.value.trim());
  }
});

sendBtn.addEventListener("click", () => sendMessage(input.value.trim()));

// ── Close message dropdowns on outside click ──────────────────────────────────
document.addEventListener("click", (e) => {
  if (!e.target.closest(".msg-actions")) {
    document.querySelectorAll(".msg-dropdown").forEach(d => d.classList.add("hidden"));
  }
  if (e.target.classList.contains("example-pill")) {
    const prompt = e.target.dataset.prompt;
    input.value  = prompt;
    updateCharCount(prompt.length);
    sendBtn.disabled = false;
    input.focus();
  }
});

// ── Session search ─────────────────────────────────────────────────────────────
document.getElementById("session-search").addEventListener("input", (e) => {
  const q = e.target.value.toLowerCase();
  document.querySelectorAll(".session-item").forEach((item) => {
    item.style.display = item.textContent.toLowerCase().includes(q) ? "" : "none";
  });
});

// ── Sidebar toggle (mobile) ────────────────────────────────────────────────────
document.getElementById("sidebar-toggle").addEventListener("click", () => {
  document.getElementById("sidebar").classList.toggle("open");
});

// ── Logout ─────────────────────────────────────────────────────────────────────
function logout() {
  localStorage.removeItem("cineagent_token");
  window.location.href = "index.html";
}
document.getElementById("logout-btn").addEventListener("click", logout);

// ── Init ───────────────────────────────────────────────────────────────────────
loadSessions();
input.focus();