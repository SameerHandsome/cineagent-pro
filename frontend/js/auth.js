/* frontend/js/auth.js — login, register, GitHub OAuth */

const API = "http://localhost:8000";

// ── Tab switching ────────────────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    const target = tab.dataset.tab;
    document.getElementById("login-form").classList.toggle("hidden", target !== "login");
    document.getElementById("register-form").classList.toggle("hidden", target !== "register");
    clearErrors();
  });
});

function clearErrors() {
  document.getElementById("login-error").textContent = "";
  document.getElementById("register-error").textContent = "";
}

function setLoading(formId, loading) {
  const btnId = formId === "login-form" ? "login-btn" : "register-btn";
  const btn = document.getElementById(btnId);
  btn.disabled = loading;
  btn.querySelector(".btn-text").classList.toggle("hidden", loading);
  btn.querySelector(".btn-spinner").classList.toggle("hidden", !loading);
}

// ── Token helpers ────────────────────────────────────────────────────────────
function saveToken(token) {
  localStorage.setItem("cineagent_token", token);
}

function redirect() {
  window.location.href = "chat.html";
}

// ── Redirect if already logged in ───────────────────────────────────────────
if (localStorage.getItem("cineagent_token")) {
  redirect();
}

// ── Check for GitHub OAuth callback token in URL ─────────────────────────────
// The backend redirects to / with ?token=... after GitHub OAuth.
const urlParams = new URLSearchParams(window.location.search);
const oauthToken = urlParams.get("token");
if (oauthToken) {
  saveToken(oauthToken);
  redirect();
}

// ── Login ────────────────────────────────────────────────────────────────────
document.getElementById("login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  clearErrors();
  setLoading("login-form", true);

  const email    = document.getElementById("login-email").value.trim();
  const password = document.getElementById("login-password").value;

  try {
    const res = await fetch(`${API}/auth/login`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ email, password }),
    });

    const data = await res.json();

    if (!res.ok) {
      document.getElementById("login-error").textContent =
        data.detail || "Login failed. Check your credentials.";
      return;
    }

    saveToken(data.access_token);
    redirect();
  } catch {
    document.getElementById("login-error").textContent =
      "Network error — is the backend running?";
  } finally {
    setLoading("login-form", false);
  }
});

// ── Register ─────────────────────────────────────────────────────────────────
document.getElementById("register-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  clearErrors();
  setLoading("register-form", true);

  const username = document.getElementById("reg-username").value.trim();
  const email    = document.getElementById("reg-email").value.trim();
  const password = document.getElementById("reg-password").value;

  if (password.length < 8) {
    document.getElementById("register-error").textContent = "Password must be at least 8 characters.";
    setLoading("register-form", false);
    return;
  }

  try {
    const res = await fetch(`${API}/auth/register`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ username, email, password }),
    });

    const data = await res.json();

    if (!res.ok) {
      document.getElementById("register-error").textContent =
        data.detail || "Registration failed.";
      return;
    }

    saveToken(data.access_token);
    redirect();
  } catch {
    document.getElementById("register-error").textContent =
      "Network error — is the backend running?";
  } finally {
    setLoading("register-form", false);
  }
});
