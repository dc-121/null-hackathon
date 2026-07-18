const form = document.getElementById("prompt-form");
const promptInput = document.getElementById("prompt");
const phraseInput = document.getElementById("phrase-tokens");
const phraseValue = document.getElementById("phrase-value");
const modeInput = document.getElementById("delivery-mode");
const runButton = document.getElementById("run");
const status = document.getElementById("status");
const results = document.getElementById("results");
const response = document.getElementById("response");
const generationTime = document.getElementById("generation-time");
const replayTime = document.getElementById("replay-time");
const phraseStrip = document.getElementById("phrase-strip");
const selectedText = document.getElementById("selected-text");
const selectedTags = document.getElementById("selected-tags");
const signalRows = document.getElementById("signal-rows");
const taggedText = document.getElementById("tagged-text");
const audio = document.getElementById("audio");

let currentRun = null;
let selectedPhrase = 0;

phraseInput.addEventListener("input", () => {
  phraseValue.value = phraseInput.value;
});

function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function signed(value) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(4)}`;
}

function dominantLabel(phrase) {
  if (!phrase.components.length) return "neutral";
  const winner = phrase.components[0];
  return `${winner.name} · ${Math.round(winner.intensity * 100)}% intensity`;
}

function renderPhraseStrip() {
  phraseStrip.replaceChildren();
  currentRun.phrases.forEach((phrase, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = index === selectedPhrase ? "phrase active" : "phrase";
    button.setAttribute("aria-pressed", index === selectedPhrase ? "true" : "false");
    const number = document.createElement("span");
    const copy = document.createElement("strong");
    const mixture = document.createElement("small");
    number.textContent = String(index + 1);
    copy.textContent = phrase.text;
    mixture.textContent = dominantLabel(phrase);
    button.append(number, copy, mixture);
    button.addEventListener("click", () => {
      selectedPhrase = index;
      renderPhraseStrip();
      renderInspector();
    });
    phraseStrip.appendChild(button);
  });
}

function bipolarBar(value, maximum, className) {
  const normalized = maximum ? Math.min(1, Math.abs(value) / maximum) : 0;
  const width = normalized * 50;
  const left = value < 0 ? 50 - width : 50;
  return `<span class="bipolar"><i class="${className}" style="left:${left}%;width:${width}%"></i></span>`;
}

function renderInspector() {
  const phrase = currentRun.phrases[selectedPhrase];
  selectedText.textContent = phrase.text;
  selectedTags.textContent = phrase.direction || "[neutral]";
  const maxRaw = Math.max(...phrase.signals.map((signal) => Math.abs(signal.raw)), 0.001);
  const maxZ = Math.max(...phrase.signals.map((signal) => Math.abs(signal.z)), 0.001);
  const maxDeltaZ = Math.max(...phrase.signals.map((signal) => Math.abs(signal.delta_z)), 0.001);
  const selected = new Set(phrase.components.map((component) => component.name));
  const ordered = [...phrase.signals].sort((a, b) => b.weight - a.weight || b.evidence - a.evidence);
  signalRows.replaceChildren();
  ordered.forEach((signal) => {
    const row = document.createElement("div");
    const controlsVoice = selected.has(signal.name);
    row.className = controlsVoice ? "signal-row selected" : "signal-row";
    row.setAttribute("role", "row");
    row.innerHTML = `
      <strong>${signal.name}${controlsVoice ? "<small>voice</small>" : ""}</strong>
      <span class="metric-cell"><span class="metric">${bipolarBar(signal.raw, maxRaw, "raw")}</span><code>${signed(signal.raw)}</code></span>
      <span class="metric-cell"><span class="metric">${bipolarBar(signal.z, maxZ, "zscore")}</span><code>${signed(signal.z)}</code></span>
      <span class="metric-cell"><span class="metric">${bipolarBar(signal.delta_z, maxDeltaZ, "delta")}</span><code>${signed(signal.delta_z)}</code></span>
      <span class="metric-cell"><span class="weight"><i style="width:${signal.weight * 100}%"></i></span><code>${percent(signal.weight)} · evidence ${signal.evidence.toFixed(2)}</code></span>
    `;
    signalRows.appendChild(row);
  });
}

function renderRun(data) {
  currentRun = data;
  selectedPhrase = 0;
  response.textContent = data.response;
  generationTime.textContent = `generation ${data.timings.generation_seconds.toFixed(2)}s`;
  replayTime.textContent = `activation replay ${data.timings.replay_seconds.toFixed(2)}s`;
  taggedText.textContent = data.tagged_text;
  renderPhraseStrip();
  renderInspector();
  results.hidden = false;
  audio.src = `/api/audio/${data.speech_id}`;
  audio.load();
  audio.play().catch(() => {});
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = promptInput.value.trim();
  if (!prompt) return;
  runButton.disabled = true;
  status.textContent = "Generating locally, replaying layer 28, then selecting one strong voice direction…";
  try {
    const request = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        max_new_tokens: 64,
        delivery_mode: modeInput.value,
        phrase_tokens: Number(phraseInput.value),
      }),
    });
    if (!request.ok) {
      const error = await request.json().catch(() => ({}));
      throw new Error(error.detail || `Request failed (${request.status})`);
    }
    const data = await request.json();
    renderRun(data);
    status.textContent = "Trace complete. Audio is streaming from the exact tags shown below.";
  } catch (error) {
    status.textContent = `Error: ${error.message}`;
  } finally {
    runButton.disabled = false;
  }
});
