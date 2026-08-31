const $ = (selector) => document.querySelector(selector);

const state = {
  conversationId: null,
  imagePath: null,
  imageUrl: null,
  running: false,
  timer: null,
  startedAt: 0,
};

const defectLabels = {
  insufficient_solder: "少锡",
  excessive_solder: "多锡",
  short: "短路 / 桥连",
  shifted_component: "元件偏移",
};

const intentLabels = {
  identify_defect: "缺陷识别",
  diagnose_cause: "原因诊断",
  optimize_process: "工艺优化",
  explain_result: "上下文追问",
  explain_evidence: "证据解释",
  provide_data: "补充现场数据",
  new_case: "新诊断 Case",
};

function node(tag, className = "", text = "") {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== "") item.textContent = text;
  return item;
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail || payload);
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return payload;
}

function setStatus(text, error = false) {
  $("#form-status").textContent = text;
  $("#form-status").classList.toggle("error", error);
}

function scrollBottom() {
  const list = $("#message-list");
  requestAnimationFrame(() => { list.scrollTop = list.scrollHeight; });
}

function setRunning(running, title = "正在理解你的问题…") {
  state.running = running;
  $("#send-message").disabled = running;
  $("#message-input").disabled = running;
  $("#progress-panel").hidden = !running;
  if (state.timer) clearInterval(state.timer);
  if (running) {
    state.startedAt = Date.now();
    $("#progress-title").textContent = title;
    $("#elapsed-time").textContent = "0s";
    state.timer = setInterval(() => {
      $("#elapsed-time").textContent = `${Math.floor((Date.now() - state.startedAt) / 1000)}s`;
    }, 500);
    scrollBottom();
  }
}

function renderProgress(rows) {
  const container = $("#progress-steps");
  container.replaceChildren();
  rows.forEach((row) => {
    const item = node("div", `progress-step ${row.status}`);
    item.append(node("i"), node("span", "", row.stage));
    container.append(item);
  });
  const running = rows.find((row) => row.status === "running");
  if (running) $("#progress-title").textContent = running.stage;
}

async function waitForJob(jobId) {
  while (true) {
    const job = await api(`/api/v1/conversation-jobs/${encodeURIComponent(jobId)}`);
    renderProgress(job.progress || []);
    if (job.status === "completed") return job.result;
    if (job.status === "failed") throw new Error(job.error || "Conversation job failed");
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
}

function messageShell(role) {
  const article = node("article", `chat-message ${role}`);
  article.append(node("div", "message-avatar", role === "user" ? "你" : "AI"));
  const bubble = node("div", "message-bubble");
  article.append(bubble);
  return { article, bubble };
}

function renderUser(content, imageUrl = null) {
  $("#welcome-message").hidden = true;
  const { article, bubble } = messageShell("user");
  if (imageUrl) {
    const image = node("img", "message-image");
    image.src = imageUrl;
    image.alt = "用户提交的缺陷图片";
    bubble.append(image);
  }
  bubble.append(node("p", "message-text", content || "请分析这张缺陷图片。"));
  $("#message-list").append(article);
  scrollBottom();
}

function addFact(grid, label, value, emphasized = false) {
  if (value == null) return;
  const row = node("div", `fact ${emphasized ? "emphasized" : ""}`);
  row.append(node("span", "", label), node("b", "", String(value)));
  grid.append(row);
}

function renderOptimization(container, candidates) {
  candidates.forEach((candidate) => {
    const optimization = candidate.optimization_result || {};
    if (!["accepted", "rejected", "failed"].includes(optimization.recommendation_status)) return;
    const validation = (((candidate.validation_result || {}).response || {}).data || {});
    const initial = validation.overall || {};
    const recommendation = (((optimization.optimization_response || {}).data || {}).recommended_parameters || {});
    const revalidation = (((optimization.revalidation_response || {}).data || {}).overall || {});
    const card = node("section", "inline-card optimization-card");
    const title = node("div", "inline-title");
    title.append(node("b", "", "推荐参数与复验"), node("span", "", optimization.recommendation_status === "accepted" ? "复验通过" : "未通过"));
    card.append(title);
    const facts = node("div", "fact-grid");
    addFact(facts, "优化前 PWI", initial.max_pwi);
    addFact(facts, "优化前 VTE", validation.vte_mean == null ? null : `${validation.vte_mean}%`);
    addFact(facts, "推荐链速", recommendation.belt_speed_cm_min == null ? null : `${recommendation.belt_speed_cm_min} cm/min`);
    addFact(facts, "复验 PWI", revalidation.max_pwi, true);
    const spiRevalidation = ((optimization.revalidation_response || {}).data || {});
    addFact(facts, "复验 VTE", spiRevalidation.vte_mean == null ? null : `${spiRevalidation.vte_mean}%`, true);
    addFact(facts, "复验结论", revalidation.qualified == null ? null : (revalidation.qualified ? "合格" : "不合格"), true);
    card.append(facts);
    if (recommendation.squeegee_pressure_kgf != null) {
      const printing = node("div", "fact-grid printing-grid");
      addFact(printing, "刮刀压力", recommendation.squeegee_pressure_kgf);
      addFact(printing, "刮刀速度", recommendation.squeegee_speed_m_s);
      addFact(printing, "分离速度", recommendation.separation_speed_m_s);
      addFact(printing, "分离距离", recommendation.separation_distance_mm);
      card.append(printing);
    }
    if (Array.isArray(recommendation.zone_means_c)) {
      const zones = node("div", "zone-values");
      recommendation.zone_means_c.forEach((value, index) => {
        const item = node("span");
        item.append(node("small", "", `Z${index + 1}`), node("b", "", `${value}°`));
        zones.append(item);
      });
      card.append(zones);
    }
    container.append(card);
  });
}

function details(title, count = null, open = false) {
  const box = node("details", "answer-details");
  box.open = open;
  const summary = node("summary", "", title);
  if (count != null) summary.append(node("span", "", `${count} 条`));
  box.append(summary);
  return box;
}

function renderResultDetails(bubble, result, executionTrace = []) {
  const candidates = result?.candidates || [];
  renderOptimization(bubble, candidates);
  if (candidates.length) {
    const box = details("候选致因", candidates.length, true);
    candidates.forEach((candidate) => {
      const cause = candidate.candidate_cause || {};
      const item = node("div", `candidate-row ${candidate.assessment_status || ""}`);
      item.append(node("b", "", cause.display_name_zh || cause.canonical_name || candidate.relationship_id));
      item.append(node("span", "", candidate.assessment_status || "not_evaluated"));
      const reason = (candidate.validation_result || {}).reason;
      if (reason) item.append(node("p", "", reason));
      box.append(item);
    });
    bubble.append(box);
  }
  const evidence = result?.rag_evidence || [];
  if (evidence.length) {
    const box = details("知识证据", evidence.length);
    evidence.forEach((row, index) => {
      const citation = row.citation || {};
      const item = node("div", "evidence-row");
      item.append(node("b", "", `证据${index + 1} · ${citation.source_id || row.chunk_id || "知识库"}`));
      item.append(node("p", "", row.text || ""));
      box.append(item);
    });
    bubble.append(box);
  }
  const toolTrace = result?.tool_trace || [];
  const trace = [...executionTrace, ...toolTrace.map((row) => ({stage: row.tool_name || row.phase, status: row.success === false ? "failed" : "completed"}))];
  if (trace.length) {
    const box = details("执行与上下文轨迹", trace.length);
    trace.forEach((row) => {
      const item = node("div", `trace-row ${row.status || ""}`);
      item.append(node("i"), node("span", "", row.stage || "Agent"), node("em", "", row.status === "failed" ? "失败" : "完成"));
      box.append(item);
    });
    bubble.append(box);
  }
}

function renderAssistant(content, metadata = {}, turn = null) {
  $("#welcome-message").hidden = true;
  const { article, bubble } = messageShell("assistant");
  const intent = turn?.intent || metadata.intent;
  const head = node("div", "answer-head");
  head.append(node("span", "", intentLabels[intent] || "Agent回复"));
  if (turn?.reused_context || metadata.reused_context) head.append(node("em", "", "已复用上下文"));
  bubble.append(head, node("div", "report-text", content || "系统没有生成文本回复。"));
  const result = turn?.result || metadata.result;
  if (result) renderResultDetails(bubble, result, turn?.execution_trace || []);
  $("#message-list").append(article);
  scrollBottom();
}

async function ensureConversation(forceNew = false) {
  if (!forceNew) state.conversationId = localStorage.getItem("pcba_conversation_id");
  if (state.conversationId) {
    try {
      const conversation = await api(`/api/v1/conversations/${encodeURIComponent(state.conversationId)}`);
      $("#conversation-chip").textContent = state.conversationId.slice(0, 8);
      conversation.messages.forEach((message) => {
        if (message.role === "user") renderUser(message.content, null);
        else renderAssistant(message.content, message.metadata || {});
      });
      return;
    } catch (_) {
      localStorage.removeItem("pcba_conversation_id");
    }
  }
  const conversation = await api("/api/v1/conversations", {method: "POST"});
  state.conversationId = conversation.conversation_id;
  localStorage.setItem("pcba_conversation_id", state.conversationId);
  $("#conversation-chip").textContent = state.conversationId.slice(0, 8);
}

async function uploadImage(file) {
  const form = new FormData();
  form.append("image", file);
  $("#image-state").textContent = "上传中";
  const payload = await api("/api/v1/agent/images", {method: "POST", body: form});
  state.imagePath = payload.data.image_path;
  $("#image-state").textContent = "已就绪";
}

function selectImage(file) {
  if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
  state.imageUrl = URL.createObjectURL(file);
  const preview = $("#image-preview");
  preview.src = state.imageUrl;
  preview.hidden = false;
  $("#image-placeholder").hidden = true;
  uploadImage(file).catch((error) => setStatus(error.message, true));
}

async function sendMessage(event) {
  event.preventDefault();
  if (state.running) return;
  const content = $("#message-input").value.trim();
  if (!content && !state.imagePath) return setStatus("请发送问题或图片。", true);
  const imagePath = state.imagePath;
  const imageUrl = state.imageUrl;
  renderUser(content, imageUrl);
  $("#message-input").value = "";
  setRunning(true);
  setStatus("Agent正在处理，本轮不会重复执行不必要的模块。");
  try {
    const submitted = await api(`/api/v1/conversations/${encodeURIComponent(state.conversationId)}/message-jobs`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({content, image_path: imagePath}),
    });
    const turn = await waitForJob(submitted.job_id);
    renderAssistant(turn.assistant_text, {}, turn);
    setStatus(turn.status === "needs_input" ? "请直接在对话中补充所需信息。" : "回复完成。可以继续追问。 ");
    state.imagePath = null;
    state.imageUrl = null;
    $("#image-input").value = "";
    $("#image-preview").hidden = true;
    $("#image-preview").removeAttribute("src");
    $("#image-placeholder").hidden = false;
    $("#image-state").textContent = "";
  } catch (error) {
    renderAssistant(`本轮执行失败：${error.message}`);
    setStatus(error.message, true);
  } finally {
    setRunning(false);
  }
}

$("#chat-form").addEventListener("submit", sendMessage);
$("#message-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#chat-form").requestSubmit();
  }
});
$("#image-input").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) selectImage(file);
});
$("#new-conversation").addEventListener("click", async () => {
  if (state.running) return;
  localStorage.removeItem("pcba_conversation_id");
  $("#message-list").querySelectorAll(".chat-message:not(#welcome-message)").forEach((item) => item.remove());
  $("#welcome-message").hidden = false;
  state.conversationId = null;
  await ensureConversation(true);
  setStatus("已建立新的诊断会话。 ");
});
document.querySelectorAll(".prompt-examples button").forEach((button) => {
  button.addEventListener("click", () => { $("#message-input").value = button.textContent; $("#message-input").focus(); });
});

ensureConversation().then(() => setStatus("可以开始提问")).catch((error) => setStatus(error.message, true));
