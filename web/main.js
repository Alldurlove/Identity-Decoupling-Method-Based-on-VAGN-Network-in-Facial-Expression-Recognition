// 简单的前端交互逻辑：仅用于 UI 动画模拟，不包含真实推理或摄像头访问。

const pipelineNodes = Array.from(document.querySelectorAll(".pipeline-node"));
const statusIndicator = document.getElementById("status-indicator");
const btnToggle = document.getElementById("btn-toggle");
const qualityRange = document.getElementById("quality-range");
const qualityLabel = document.getElementById("quality-label");

let activeStep = 0;
let running = false;
let timerId = null;

function setRunningState(isRunning) {
  running = isRunning;

  if (running) {
    statusIndicator.innerHTML = `
      <span class="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
      模拟推理进行中
    `;
    btnToggle.textContent = "停止模拟";
    btnToggle.classList.remove("bg-emerald-500", "hover:bg-emerald-400");
    btnToggle.classList.add("bg-rose-500", "hover:bg-rose-400");
    startPipelineAnimation();
  } else {
    statusIndicator.innerHTML = `
      <span class="h-1.5 w-1.5 rounded-full bg-slate-500"></span>
      等待启动
    `;
    btnToggle.textContent = "启动模拟推理";
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

function updateQualityLabel() {
  const value = parseInt(qualityRange.value || "1", 10);
  let label = "平衡";
  if (value === 0) label = "低延迟优先";
  if (value === 2) label = "画质优先";
  qualityLabel.textContent = label;
}

if (btnToggle) {
  btnToggle.addEventListener("click", () => {
    setRunningState(!running);
  });
}

if (qualityRange) {
  qualityRange.addEventListener("input", updateQualityLabel);
  updateQualityLabel();
}

// 初始状态：未启动
setRunningState(false);

