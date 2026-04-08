const form = document.getElementById("ask-form");
const questionInput = document.getElementById("question");
const topKInput = document.getElementById("top_k");
const topKValue = document.getElementById("top_k_value");
const submitButton = document.getElementById("submit-button");
const answerBox = document.getElementById("answer-box");
const sourcesBox = document.getElementById("sources-box");
const statusText = document.getElementById("status-text");

topKInput.addEventListener("input", () => {
  topKValue.value = topKInput.value;
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey) {
    return;
  }

  event.preventDefault();
  if (submitButton.disabled) {
    return;
  }

  form.requestSubmit();
});

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderSources(sources) {
  if (!sources.length) {
    sourcesBox.innerHTML = '<div class="empty-state">No retrieved chunks were returned for this question.</div>';
    return;
  }

  sourcesBox.innerHTML = sources
    .map((source) => {
      const metadata = source.metadata || {};
      const file = escapeHtml(metadata.source_file || "unknown");
      const page = escapeHtml(String(metadata.page_number || "unknown"));
      const chunk = escapeHtml(String(metadata.chunk_index ?? "unknown"));
      const text = escapeHtml(source.text || "");

      return `
        <article class="source-card">
          <div class="source-topline">
            <div class="source-file">${file}</div>
            <div class="source-distance">distance ${source.distance.toFixed(4)}</div>
          </div>
          <div class="source-meta">page ${page} · chunk ${chunk}</div>
          <div class="source-text">${text}</div>
        </article>
      `;
    })
    .join("");
}

function setBusyState(isBusy) {
  submitButton.disabled = isBusy;
  submitButton.textContent = isBusy ? "Running Query..." : "Run Technical Query";
  statusText.classList.toggle("status-busy", isBusy);
}

async function readEventStream(response, handlers) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";

    for (const eventBlock of events) {
      const lines = eventBlock.split("\n");
      let eventName = "message";
      const dataLines = [];

      for (const line of lines) {
        if (line.startsWith("event:")) {
          eventName = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).trim());
        }
      }

      if (!dataLines.length) {
        continue;
      }

      const payload = JSON.parse(dataLines.join("\n"));
      const handler = handlers[eventName];
      if (handler) {
        handler(payload);
      }
    }
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();
  if (!question) {
    statusText.textContent = "Enter a question before running the query.";
    statusText.classList.add("status-error");
    questionInput.focus();
    return;
  }

  statusText.classList.remove("status-error");
  statusText.textContent = "Retrieving relevant chunks and starting stream...";
  answerBox.classList.remove("empty");
  answerBox.textContent = "";
  sourcesBox.innerHTML = '<div class="empty-state">Retrieval in progress.</div>';
  setBusyState(true);

  try {
    const response = await fetch("/api/ask/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        top_k: Number(topKInput.value),
      }),
    });

    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.detail || "Request failed.");
    }

    let retrievedCount = 0;

    await readEventStream(response, {
      retrieved(payload) {
        const sources = payload.retrieved || [];
        retrievedCount = sources.length;
        renderSources(sources);
        statusText.textContent = `Retrieved ${retrievedCount} chunk(s). Streaming answer...`;
      },
      token(payload) {
        answerBox.textContent += payload.content || "";
      },
      done(payload) {
        if (!answerBox.textContent.trim()) {
          answerBox.textContent = payload.answer || "";
        }
        statusText.textContent = `Completed query with ${retrievedCount} retrieved chunk(s).`;
      },
      error(payload) {
        throw new Error(payload.detail || "Streaming request failed.");
      },
    });
  } catch (error) {
    answerBox.textContent = error.message;
    sourcesBox.innerHTML = '<div class="empty-state">No evidence available because the request failed.</div>';
    statusText.textContent = "The query failed. Check whether Ollama and the vector store are ready.";
    statusText.classList.add("status-error");
  } finally {
    setBusyState(false);
  }
});
