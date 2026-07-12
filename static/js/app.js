const predictBtn = document.getElementById("predictBtn");
const newsInput = document.getElementById("newsInput");
const errorText = document.getElementById("errorText");
const aiWarningText = document.getElementById("aiWarningText");
const resultBox = document.getElementById("resultBox");
const predictionLabel = document.getElementById("predictionLabel");
const confidenceValue = document.getElementById("confidenceValue");
const resultEmoji = document.getElementById("resultEmoji");
const confidenceBar = document.getElementById("confidenceBar");
const loadingWrap = document.getElementById("loadingWrap");
const aiResultValue = document.getElementById("aiResultValue");
const sourcesValue = document.getElementById("sourcesValue");
const finalDecisionValue = document.getElementById("finalDecisionValue");
const statusValue = document.getElementById("statusValue");
const explanationText = document.getElementById("explanationText");
let typingInterval = null;

const modalOverlay = document.getElementById("authModalOverlay");
const loginFormWrap = document.getElementById("loginFormWrap");
const signupFormWrap = document.getElementById("signupFormWrap");
const closeAuthModalBtn = document.getElementById("closeAuthModal");
const openModalButtons = document.querySelectorAll("[data-open-modal]");
const logoutButtons = document.querySelectorAll("[data-auth-action='logout']");

const loginEmail = document.getElementById("loginEmail");
const loginPassword = document.getElementById("loginPassword");
const loginError = document.getElementById("loginError");
const loginSubmitBtn = document.getElementById("loginSubmitBtn");

const signupUsername = document.getElementById("signupUsername");
const signupEmail = document.getElementById("signupEmail");
const signupPassword = document.getElementById("signupPassword");
const signupError = document.getElementById("signupError");
const signupSubmitBtn = document.getElementById("signupSubmitBtn");

const TYPING_DEFAULT = ["Analyzing", "Analyzing.", "Analyzing..", "Analyzing..."];

function startTyping(frames) {
    const el = document.getElementById("typingLabel");
    if (!el) return;
    const useFrames = frames && frames.length ? frames : TYPING_DEFAULT;
    let i = 0;
    el.textContent = useFrames[0];
    typingInterval = setInterval(() => {
        i = (i + 1) % useFrames.length;
        el.textContent = useFrames[i];
    }, 450);
}

function stopTyping(fallbackText) {
    if (typingInterval) {
        clearInterval(typingInterval);
        typingInterval = null;
    }
    const el = document.getElementById("typingLabel");
    if (el) el.textContent = fallbackText || "Done.";
}

function showModal(mode) {
    if (!modalOverlay) return;
    modalOverlay.classList.remove("hidden");
    loginFormWrap?.classList.add("hidden");
    signupFormWrap?.classList.add("hidden");
    if (mode === "login") {
        loginFormWrap?.classList.remove("hidden");
    } else {
        signupFormWrap?.classList.remove("hidden");
    }
}

function hideModal() {
    if (!modalOverlay) return;
    modalOverlay.classList.add("hidden");
    if (loginError) loginError.textContent = "";
    if (signupError) signupError.textContent = "";
}

openModalButtons.forEach((button) => {
    button.addEventListener("click", () => {
        const mode = button.getAttribute("data-open-modal");
        showModal(mode);
    });
});

if (closeAuthModalBtn) {
    closeAuthModalBtn.addEventListener("click", hideModal);
}

if (modalOverlay) {
    modalOverlay.addEventListener("click", (event) => {
        if (event.target === modalOverlay) {
            hideModal();
        }
    });
}

async function submitSignup() {
    const username = signupUsername?.value.trim() || "";
    const email = signupEmail?.value.trim() || "";
    const password = signupPassword?.value.trim() || "";

    signupError.textContent = "";

    if (!username || !email || !password) {
        signupError.textContent = "Please fill all fields.";
        return;
    }

    const response = await fetch("/auth/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, email, password })
    });

    const data = await response.json();

    if (!response.ok) {
        signupError.textContent = data.error || "Signup failed.";
        return;
    }

    window.location.reload();
}

async function submitLogin() {
    const email = loginEmail?.value.trim() || "";
    const password = loginPassword?.value.trim() || "";

    loginError.textContent = "";

    if (!email || !password) {
        loginError.textContent = "Please enter email and password.";
        return;
    }

    const response = await fetch("/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password })
    });

    const data = await response.json();

    if (!response.ok) {
        loginError.textContent = data.error || "Login failed.";
        return;
    }

    window.location.reload();
}

async function submitLogout() {
    await fetch("/auth/logout", { method: "POST" });
    window.location.reload();
}

if (signupSubmitBtn) {
    signupSubmitBtn.addEventListener("click", submitSignup);
}

if (loginSubmitBtn) {
    loginSubmitBtn.addEventListener("click", submitLogin);
}

logoutButtons.forEach((button) => {
    button.addEventListener("click", submitLogout);
});

function formatSources(data) {
    const summary = data.search_summary;
    const n = summary?.article_count ?? 0;
    const src = summary?.sources?.[0];
    const extra = src?.title ? ` — ${src.title.slice(0, 80)}${src.title.length > 80 ? "…" : ""}` : "";
    if (n <= 0) return "None retrieved (add NEWS_API_KEY for live headlines)";
    return `${n} article(s)${extra}`;
}

function applyStatusClasses(el, statusRaw) {
    if (!el) return;
    const s = String(statusRaw || "").toLowerCase();
    el.classList.remove("status-verified", "status-review", "status-needs");
    if (s.includes("verification")) {
        el.classList.add("status-needs");
    } else if (s.includes("verified")) {
        el.classList.add("status-verified");
    } else {
        el.classList.add("status-review");
    }
}

const VERIFY_FRAMES = ["Verifying facts", "Verifying facts.", "Verifying facts..", "Verifying facts..."];

if (predictBtn) {
    predictBtn.addEventListener("click", async () => {
        const newsText = newsInput.value.trim();
        errorText.textContent = "";
        if (aiWarningText) {
            aiWarningText.textContent = "";
            aiWarningText.classList.add("hidden");
        }
        resultBox.classList.add("hidden");
        resultBox.classList.remove("result-fake", "result-real", "result-unclear");
        if (loadingWrap) loadingWrap.classList.add("hidden");
        stopTyping("Verifying facts...");

        if (!newsText) {
            errorText.textContent = "Please enter some news text first.";
            return;
        }

        try {
            if (loadingWrap) loadingWrap.classList.remove("hidden");
            predictBtn.disabled = true;
            predictBtn.textContent = "Verifying...";
            startTyping(VERIFY_FRAMES);

            const apiBase = window.location.origin || "";
            const response = await fetch(`${apiBase}/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ news_text: newsText, country: "Global" }),
                cache: "no-store",
            });

            const rawText = await response.text();
            let data = {};
            try {
                data = rawText ? JSON.parse(rawText) : {};
            } catch {
                errorText.textContent =
                    `Server returned non-JSON (HTTP ${response.status}). If you see HTML, the app may have crashed — check the terminal running python backend/app.py.`;
                return;
            }

            if (!response.ok) {
                errorText.textContent = data.error || `Request failed (HTTP ${response.status}).`;
                return;
            }

            predictionLabel.textContent = data.prediction;
            confidenceValue.textContent = Math.round(Number(data.confidence) || 0);
            sourcesValue.textContent = formatSources(data);
            statusValue.textContent = data.status || "Needs Review";
            applyStatusClasses(statusValue, data.status);

            const ai = data.ai_result || {};
            const fd = data.final_decision || {};
            if (aiResultValue) {
                aiResultValue.textContent = ai.label
                    ? `${ai.label} (${Math.round(Number(ai.confidence) || 0)}%)`
                    : String(data.prediction || "-");
            }
            if (finalDecisionValue) {
                finalDecisionValue.textContent = fd.label
                    ? `${fd.label} (${Math.round(Number(fd.confidence) || 0)}%)`
                    : String(data.prediction || "-");
            }
            if (explanationText) {
                explanationText.textContent = data.explanation || "-";
            }
            if (aiWarningText) {
                const prov = String(data.provider || "").toLowerCase();
                const exp = String(data.explanation || "").toLowerCase();
                const isMlFallback =
                    prov === "ml" ||
                    exp.includes("style-based") ||
                    exp.includes("ai fact-check was unavailable");
                if (isMlFallback) {
                    aiWarningText.textContent =
                        "Warning: Real AI fact-check did not run (Gemini quota or API error). " +
                        "Result below is only a rough ML text-style guess — do not trust high confidence.";
                    aiWarningText.classList.remove("hidden");
                } else if (prov === "groq" || prov === "gemini" || prov === "openai") {
                   // aiWarningText.textContent = `Verified using ${prov} AI fact-check.`;
                    aiWarningText.classList.remove("hidden");
                }
            }
            const confidence = Number(data.confidence) || 0;
            if (confidenceBar) {
                confidenceBar.style.width = `${Math.min(100, Math.max(0, confidence))}%`;
            }

            const pred = String(data.prediction || "").toLowerCase();
            if (pred === "fake") {
                resultEmoji.textContent = "❌";
                resultBox.classList.add("result-fake");
            } else if (pred === "unclear") {
                resultEmoji.textContent = "⚠️";
                resultBox.classList.add("result-unclear");
            } else {
                resultEmoji.textContent = "✅";
                resultBox.classList.add("result-real");
            }

            resultBox.classList.remove("hidden");
        } catch (error) {
            const hint =
                error && error.message ? String(error.message) : String(error || "");
            errorText.textContent =
                `Network error — could not complete request. ${hint ? `(${hint}) ` : ""}` +
                "Keep one terminal open with: python backend/app.py (wait until you see \"Serving on\" and do not close it). " +
                "Then use http://127.0.0.1:5000/predict-page";
        } finally {
            stopTyping("Verifying facts...");
            if (loadingWrap) loadingWrap.classList.add("hidden");
            predictBtn.disabled = false;
            predictBtn.textContent = "Analyze News";
        }
    });
}
