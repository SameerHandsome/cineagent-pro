/* frontend/js/profile.js */

const API   = "http://localhost:8000";
const token = localStorage.getItem("cineagent_token");

if (!token) window.location.href = "index.html";

function authHeaders() {
  return { Authorization: `Bearer ${token}` };
}

function decodeToken(t) {
  try {
    return JSON.parse(atob(t.split(".")[1]));
  } catch {
    return {};
  }
}

// ── Load profile data ─────────────────────────────────────────────────────────
async function loadProfile() {
  const payload  = decodeToken(token);
  const email    = payload.email || "Unknown";
  const username = email.split("@")[0];

  document.getElementById("profile-username").textContent = username;
  document.getElementById("profile-email").textContent    = email;
  document.getElementById("user-avatar").textContent      = username.charAt(0).toUpperCase();

  try {
    const res      = await fetch(`${API}/api/sessions`, { headers: authHeaders() });
    if (res.status === 401) { logout(); return; }
    const sessions = await res.json();

    document.getElementById("stat-sessions").textContent = sessions.length;

    // Count total messages across sessions (approximate from session count)
    document.getElementById("stat-messages").textContent = sessions.length * 3 + "+";

    // Top genre: placeholder — would come from user preferences endpoint
    document.getElementById("stat-genres").textContent = "Sci-Fi";

    // Render recent sessions
    const recents = document.getElementById("recent-sessions");
    recents.innerHTML = "";

    if (!sessions.length) {
      recents.innerHTML = '<li style="font-size:13px;color:var(--text-muted)">No projects yet — start one in the chat.</li>';
      return;
    }

    sessions.slice(0, 5).forEach((s) => {
      const li   = document.createElement("li");
      const date = new Date(s.updated_at).toLocaleDateString("en-US", {
        month: "short", day: "numeric", year: "numeric",
      });
      li.innerHTML = `
        <a class="session-entry" href="chat.html">
          <span>${s.title || "Untitled Project"}</span>
          <span class="session-date">${date}</span>
        </a>`;
      // Store session id for chat navigation
      li.querySelector("a").addEventListener("click", (e) => {
        e.preventDefault();
        sessionStorage.setItem("load_session_id",    s.id);
        sessionStorage.setItem("load_session_title", s.title);
        window.location.href = "chat.html";
      });
      recents.appendChild(li);
    });

  } catch {
    console.warn("Could not load sessions.");
  }
}

// ── Logout ────────────────────────────────────────────────────────────────────
function logout() {
  localStorage.removeItem("cineagent_token");
  window.location.href = "index.html";
}
document.getElementById("signout-btn").addEventListener("click", logout);

// ── Init ──────────────────────────────────────────────────────────────────────
loadProfile();
