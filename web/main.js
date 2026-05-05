// 前端演示逻辑：摄像头采集 + 调后端推理接口。

const pipelineNodes = Array.from(document.querySelectorAll(".pipeline-node"));
const statusIndicator = document.getElementById("status-indicator");
const btnToggle = document.getElementById("btn-toggle");
const identitySelect = document.getElementById("identity-select");
const panelInput = document.getElementById("panel-input");
const panelOutput = document.getElementById("panel-output");

let activeStep = 0;
let running = false;
let timerId = null;
let inferTimerId = null;
let mediaStream = null;
let inferBusy = false;

const videoEl = document.createElement("video");
videoEl.autoplay = true;
videoEl.muted = true;
videoEl.playsInline = true;
videoEl.className = "h-full w-full object-cover";

const outputImg = document.createElement("img");
outputImg.alt = "模型输出";
outputImg.className = "h-full w-full object-cover";

const hiddenCanvas = document.createElement("canvas");
const hiddenCtx = hiddenCanvas.getContext("2d");

if (panelInput) {
  panelInput.innerHTML = "";
  panelInput.appendChild(videoEl);
}
if (panelOutput) {
  panelOutput.innerHTML = "";
  panelOutput.appendChild(outputImg);
}

function setRunningState(isRunning) {
  running = isRunning;

  if (running) {
    statusIndicator.innerHTML = `
      <span class="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
      推理进行中
    `;
    btnToggle.textContent = "停止";
    btnToggle.classList.remove("bg-emerald-500", "hover:bg-emerald-400");
    btnToggle.classList.add("bg-rose-500", "hover:bg-rose-400");
    startPipelineAnimation();
  } else {
    statusIndicator.innerHTML = `
      <span class="h-1.5 w-1.5 rounded-full bg-slate-500"></span>
      等待启动
    `;
    btnToggle.textContent = "启动推理";
    btnToggle.classList.remove("bg-rose-500", "hover:bg-rose-400");
    btnToggle.classList.add("bg-emerald-500", "hover:bg-emerald-400");
    stopPipelineAnimation();
  }
}

function updatePipelineHighlight() {
  pipelineNodes.forEach((node) => {
    const step = parseInt(node.dataset.step || "0", 10);
    if (step === activeStep) {
      node.classList.add(
        "ring-2",
        "ring-offset-2",
        "ring-offset-slate-950",
        "ring-emerald-400",
        "shadow-[0_0_22px_rgba(16,185,129,0.9)]"
      );
    } else {
      node.classList.remove(
        "ring-2",
        "ring-offset-2",
        "ring-offset-slate-950",
        "ring-emerald-400",
        "shadow-[0_0_22px_rgba(16,185,129,0.9)]"
      );
    }
  });
}

function stepPipeline() {
  activeStep = (activeStep + 1) % pipelineNodes.length;
  updatePipelineHighlight();
}

function startPipelineAnimation() {
  if (timerId != null) return;
  updatePipelineHighlight();
  timerId = window.setInterval(stepPipeline, 800);
}

function updateStatusText(text, dotClass = "bg-slate-500") {
  statusIndicator.innerHTML = `
    <span class="h-1.5 w-1.5 rounded-full ${dotClass}"></span>
    ${text}
  `;
}

async function checkBackendHealth() {
  try {
    const resp = await fetch("/api/health");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.json();
  } catch {
    return null;
  }
}

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("当前浏览器不支持摄像头接口");
  }
  mediaStream = await navigator.mediaDevices.getUserMedia({
    // Lower capture resolution to reduce encode/upload latency.
    video: { width: 320, height: 240 },
    audio: false,
  });
  videoEl.srcObject = mediaStream;
  await videoEl.play();
}

function stopCamera() {
  if (!mediaStream) return;
  mediaStream.getTracks().forEach((track) => track.stop());
  mediaStream = null;
}

function frameToDataURI() {
  if (!hiddenCtx || videoEl.videoWidth === 0 || videoEl.videoHeight === 0) return null;
  hiddenCanvas.width = videoEl.videoWidth;
  hiddenCanvas.height = videoEl.videoHeight;
  hiddenCtx.drawImage(videoEl, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
  // Lower JPEG quality for faster transfer and decode on both sides.
  return hiddenCanvas.toDataURL("image/jpeg", 0.6);
}

async function inferOnce() {
  if (!running || inferBusy) return;
  const dataUri = frameToDataURI();
  if (!dataUri) return;
  inferBusy = true;
  try {
    const resp = await fetch("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_base64: dataUri,
        target_id: parseInt(identitySelect?.value || "0", 10),
      }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const result = await resp.json();
    if (result?.image_base64) outputImg.src = result.image_base64;
    const modeText = result?.mode === "model" ? "模型推理" : "演示模式";
    updateStatusText(`推理中（${modeText}）`, "bg-emerald-400 animate-pulse");
  } catch {
    updateStatusText("推理异常，已暂停", "bg-rose-500");
    resetUiToIdleWithoutStatusOverride();
  } finally {
    inferBusy = false;
  }
}

function startInferenceLoop() {
  if (inferTimerId != null) return;
  // Faster polling improves perceived real-time responsiveness.
  inferTimerId = window.setInterval(inferOnce, 120);
}

function stopInferenceLoop() {
  if (inferTimerId != null) {
    window.clearInterval(inferTimerId);
    inferTimerId = null;
  }
}

function resetUiToIdleWithoutStatusOverride() {
  running = false;
  stopInferenceLoop();
  stopCamera();
  stopPipelineAnimation();
  btnToggle.textContent = "启动推理";
  btnToggle.classList.remove("bg-rose-500", "hover:bg-rose-400");
  btnToggle.classList.add("bg-emerald-500", "hover:bg-emerald-400");
}

function stopPipelineAnimation() {
  if (timerId != null) {
    window.clearInterval(timerId);
    timerId = null;
  }
  pipelineNodes.forEach((node) => {
    node.classList.remove(
      "ring-2",
      "ring-offset-2",
      "ring-offset-slate-950",
      "ring-emerald-400",
      "shadow-[0_0_22px_rgba(16,185,129,0.9)]"
    );
  });
}

if (btnToggle) {
  btnToggle.addEventListener("click", async () => {
    if (running) {
      setRunningState(false);
      stopInferenceLoop();
      stopCamera();
      return;
    }

    updateStatusText("正在检查后端连接...", "bg-sky-500 animate-pulse");
    const health = await checkBackendHealth();
    if (!health?.ok) {
      updateStatusText("后端未启动，请先运行 web_app.py", "bg-rose-500");
      return;
    }

    try {
      updateStatusText("后端已连接，正在请求摄像头权限...", "bg-sky-500 animate-pulse");
      await startCamera();
      setRunningState(true);
      startInferenceLoop();
      inferOnce();
    } catch (err) {
      const isInsecure = !window.isSecureContext;
      const hint = isInsecure
        ? "当前为非安全上下文，请改用 HTTPS 或 localhost"
        : "请检查浏览器摄像头权限是否允许";
      const detail = err instanceof Error ? err.message : "未知错误";
      updateStatusText(`无法访问摄像头：${hint}（${detail}）`, "bg-rose-500");
      resetUiToIdleWithoutStatusOverride();
    }
  });
}

// 初始状态：未启动
setRunningState(false);

