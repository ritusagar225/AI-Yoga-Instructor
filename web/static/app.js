// Establish WebSocket connection
const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
const wsUrl = `${wsProtocol}${window.location.host}/ws`;
let socket = null;

// HTML Elements
const webcam = document.getElementById('webcam');
const canvas = document.getElementById('skeleton-canvas');
const ctx = canvas.getContext('2d');
const poseDropdown = document.getElementById('pose-dropdown');
const connectionDot = document.getElementById('connection-dot');
const connectionText = document.getElementById('connection-text');
const hudState = document.getElementById('hud-state');
const hudTimer = document.getElementById('hud-timer');
const xaiList = document.getElementById('xai-attribution-list');
const feedbackList = document.getElementById('feedback-tips-list');

const btnStart = document.getElementById('btn-start-session');
const btnEnd = document.getElementById('btn-end-session');
const btnVoiceToggle = document.getElementById('btn-voice-toggle');
const btnCloseModal = document.getElementById('btn-close-modal');
const summaryModal = document.getElementById('summary-modal');
const modalContent = document.getElementById('summary-modal-content');

// Dynamic states
let isMuted = false;
let isSessionActive = false;
let lastSpeechTime = 0;
const speechCooldown = 5000; // Speak at most once every 5 seconds
let activeTargetPose = "";

// Skeletal links configuration
const CONNECTIONS = [
    ["left_shoulder", "right_shoulder"],
    ["left_shoulder", "left_hip"],
    ["right_shoulder", "right_hip"],
    ["left_hip", "right_hip"],
    ["left_shoulder", "left_elbow"],
    ["left_elbow", "left_wrist"],
    ["right_shoulder", "right_elbow"],
    ["right_elbow", "right_wrist"],
    ["left_hip", "left_knee"],
    ["left_knee", "left_ankle"],
    ["right_hip", "right_knee"],
    ["right_knee", "right_ankle"]
];

// Color Theme mapping
const COLOR_GREEN = "#2ecc71";
const COLOR_YELLOW = "#f39c12";
const COLOR_RED = "#e74c3c";
const COLOR_WHITE = "#f5f5f7";

// 1. Initialize Chart.js
let scoreChart = null;
const scoreHistory = [];
const timeHistory = [];

function initChart() {
    const chartCtx = document.getElementById('realtime-score-chart').getContext('2d');
    scoreChart = new Chart(chartCtx, {
        type: 'line',
        data: {
            labels: timeHistory,
            datasets: [{
                label: 'Overall Accuracy (%)',
                data: scoreHistory,
                borderColor: '#8e44ad',
                backgroundColor: 'rgba(142, 68, 173, 0.15)',
                borderWidth: 2,
                fill: true,
                tension: 0.3,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { min: 0, max: 100, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#95a5a6' } }
            }
        }
    });
}

function updateChart(score) {
    if (!scoreChart) return;
    
    const now = new Date().toLocaleTimeString();
    timeHistory.push(now);
    scoreHistory.push(score);
    
    // Keep sliding window of 40 points
    if (timeHistory.length > 40) {
        timeHistory.shift();
        scoreHistory.shift();
    }
    
    scoreChart.update();
}

// 2. Setup WebRTC Camera Stream
async function initCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: "user" },
            audio: false
        });
        webcam.srcObject = stream;
        
        // Wait for metadata to load to set canvas boundaries
        webcam.onloadedmetadata = () => {
            canvas.width = webcam.videoWidth;
            canvas.height = webcam.videoHeight;
            startWebsocketLoop();
        };
    } catch (e) {
        console.error("Camera access denied:", e);
        feedbackList.innerHTML = `
            <div class="feedback-item severity-high">
                <div class="feedback-text-col">
                    <span class="feedback-joint">Camera System</span>
                    <span class="feedback-tip">Could not initialize webcam stream. Please check permissions.</span>
                </div>
            </div>`;
    }
}

// 3. Connect WebSocket & Frame Processing Loop
function startWebsocketLoop() {
    socket = new WebSocket(wsUrl);
    
    socket.onopen = () => {
        connectionDot.classList.add('connected');
        connectionText.innerText = "WebSocket: Connected";
        console.log("WebSocket connection established.");
    };
    
    socket.onclose = () => {
        connectionDot.classList.remove('connected');
        connectionText.innerText = "WebSocket: Disconnected";
        console.log("WebSocket disconnected. Retrying in 3s...");
        setTimeout(startWebsocketLoop, 3000);
    };

    socket.onmessage = (event) => {
        const data = jsonParseSafe(event.data);
        if (!data) return;
        
        if (data.type === 'pipeline_update') {
            processPipelineUpdate(data);
        } else if (data.type === 'session_summary') {
            showSessionSummary(data.summary);
        }
    };
    
    // Process frames at 12 FPS to conserve bandwidth and prevent frame queue lag
    const hiddenCanvas = document.createElement('canvas');
    const hiddenCtx = hiddenCanvas.getContext('2d');
    
    const sendFrame = () => {
        if (socket && socket.readyState === WebSocket.OPEN && !webcam.paused) {
            hiddenCanvas.width = webcam.videoWidth;
            hiddenCanvas.height = webcam.videoHeight;
            hiddenCtx.drawImage(webcam, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
            
            hiddenCanvas.toBlob((blob) => {
                if (blob) {
                    socket.send(blob);
                }
            }, 'image/jpeg', 0.65); // Compress quality
        }
        setTimeout(sendFrame, 80); // Throttle
    };
    
    sendFrame();
}

// 4. Process WebSocket response data
function processPipelineUpdate(data) {
    // A. Render canvas overlays
    renderSkeletonCanvas(data.skeleton, data.errors, data.angles);
    
    // B. Update state machine HUD overlays
    hudState.innerText = data.state.toUpperCase();
    hudState.className = "state-badge";
    if (data.state.includes("Holding")) {
        hudState.classList.add("holding");
    } else if (data.state.includes("Entering")) {
        hudState.classList.add("entering");
    } else if (data.state.includes("Exiting")) {
        hudState.classList.add("exiting");
    }
    
    hudTimer.innerText = `Hold: ${data.hold_duration.toFixed(1)}s`;

    // C. Update live gauges
    updateGauge("gauge-overall", "val-overall", data.overall_score);
    updateGauge("gauge-stability", "val-stability", data.breakdown.stability);
    updateGauge("gauge-symmetry", "val-symmetry", data.breakdown.symmetry);
    
    // D. Update chart
    updateChart(data.overall_score);
    
    // E. Render Explainable AI (XAI) deductions
    renderXAIDeductions(data.deductions);
    
    // F. Render prioritized feedback lists
    renderFeedbackTips(data.errors, data.person_detected);
    
    // G. Trigger browser Text-to-Speech (TTS)
    if (!isMuted && data.tts_feedback && data.errors.length > 0) {
        speakText(data.tts_feedback);
    }
}

// 5. Render Canvas skeletons
function renderSkeletonCanvas(skeleton, errors, angles) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (!skeleton || Object.keys(skeleton).length === 0) return;
    
    const errJoints = {};
    if (errors) {
        errors.forEach(err => {
            const cleanKey = err.joint.toLowerCase().replace(" ", "_");
            errJoints[cleanKey] = err.severity;
        });
    }

    // A. Draw bones (connections)
    CONNECTIONS.forEach(([j1, j2]) => {
        if (skeleton[j1] && skeleton[j2]) {
            const pt1 = skeleton[j1];
            const pt2 = skeleton[j2];
            
            // Map coordinates onto canvas sizes
            const x1 = pt1[0] * canvas.width;
            const y1 = pt1[1] * canvas.height;
            const x2 = pt2[0] * canvas.width;
            const y2 = pt2[1] * canvas.height;
            
            // Determine bone color: color connection red/yellow if joint has active error
            let color = COLOR_GREEN;
            let width = 3;
            
            [j1, j2].forEach(joint => {
                for (let key in errJoints) {
                    if (key.includes(joint)) {
                        color = errJoints[key] === "High" ? COLOR_RED : COLOR_YELLOW;
                        width = 5;
                    }
                }
            });
            
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.strokeStyle = color;
            ctx.lineWidth = width;
            ctx.lineCap = "round";
            ctx.stroke();
        }
    });

    // B. Draw joints
    Object.keys(skeleton).forEach(jointName => {
        // Skip facial/foot details to avoid visual clutter
        if (["left_heel", "right_heel", "left_foot_index", "right_foot_index", "left_eye", "right_eye", "left_ear", "right_ear"].includes(jointName)) return;
        
        const pt = skeleton[jointName];
        const x = pt[0] * canvas.width;
        const y = pt[1] * canvas.height;
        
        let color = COLOR_GREEN;
        let radius = 6;
        
        for (let key in errJoints) {
            if (key.includes(jointName)) {
                color = errJoints[key] === "High" ? COLOR_RED : COLOR_YELLOW;
                radius = 9;
            }
        }
        
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#ffffff";
        ctx.stroke();
    });

    // C. Draw joint angle text readouts
    const angleLabels = {
        "left_elbow": "left_elbow", "right_elbow": "right_elbow",
        "left_knee": "left_knee", "right_knee": "right_knee",
        "left_hip": "left_hip", "right_hip": "right_hip"
    };

    ctx.font = "bold 13px 'Space Grotesk'";
    ctx.fillStyle = COLOR_WHITE;
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 3;
    
    Object.entries(angleLabels).forEach(([jointKey, lmKey]) => {
        if (angles[jointKey] !== undefined && skeleton[lmKey]) {
            const angleVal = Math.round(angles[jointKey]);
            const pt = skeleton[lmKey];
            const x = pt[0] * canvas.width;
            const y = pt[1] * canvas.height;
            
            const text = `${angleVal}°`;
            const offset_x = lmKey.includes("right") ? 15 : -35;
            
            ctx.strokeText(text, x + offset_x, y - 10);
            ctx.fillText(text, x + offset_x, y - 10);
        }
    });
}

// 6. Helper for updating Conic Gradient circular Gauges
function updateGauge(gaugeId, valId, value) {
    const gauge = document.getElementById(gaugeId);
    const valueEl = document.getElementById(valId);
    if (!gauge || !valueEl) return;
    
    const roundVal = Math.round(value);
    valueEl.innerText = `${roundVal}%`;
    
    // Choose dynamic color
    let color = COLOR_GREEN;
    if (roundVal < 60) color = COLOR_RED;
    else if (roundVal < 80) color = COLOR_YELLOW;
    
    gauge.style.background = `conic-gradient(${color} ${roundVal}%, rgba(255, 255, 255, 0.05) ${roundVal}%)`;
}

// 7. Render Explainable AI attribution bars
function renderXAIDeductions(deductions) {
    xaiList.innerHTML = "";
    const entries = Object.entries(deductions || {});
    
    if (entries.length === 0) {
        xaiList.innerHTML = `<div class="xai-empty">No active deviations. Baseline Score: 100%</div>`;
        return;
    }
    
    entries.forEach(([jointName, value]) => {
        const percent = Math.min(100, Math.abs(value) * 4); // Scale for visual representation
        
        const row = document.createElement("div");
        row.className = "xai-row";
        row.innerHTML = `
            <div class="xai-label-wrapper">
                <span>${jointName}</span>
                <span class="xai-val">${value} pts</span>
            </div>
            <div class="xai-bar-bg">
                <div class="xai-bar-fill" style="width: ${percent}%"></div>
            </div>`;
        xaiList.appendChild(row);
    });
}

// 8. Render active error corrections list
function renderFeedbackTips(errors, personDetected) {
    feedbackList.innerHTML = "";
    
    if (!personDetected) {
        feedbackList.innerHTML = `
            <div class="feedback-empty-state">
                <i class="fa-solid fa-user-slash"></i>
                <p>No person detected. Stand in full view of camera.</p>
            </div>`;
        return;
    }
    
    if (!errors || errors.length === 0) {
        feedbackList.innerHTML = `
            <div class="feedback-empty-state">
                <i class="fa-solid fa-check-circle" style="color: ${COLOR_GREEN}"></i>
                <p>Alignment is correct. Keep holding the posture!</p>
            </div>`;
        return;
    }
    
    errors.forEach(err => {
        const severityClass = `severity-${err.severity.toLowerCase()}`;
        const item = document.createElement("div");
        item.className = `feedback-item ${severityClass}`;
        item.innerHTML = `
            <div class="feedback-text-col">
                <span class="feedback-joint">${err.joint}</span>
                <span class="feedback-tip">${err.tip}</span>
            </div>
            <span class="feedback-severity-tag">${err.severity}</span>`;
        feedbackList.appendChild(item);
    });
}

// 9. Speak Text-to-Speech feedback (throttled)
function speakText(text) {
    const now = Date.now();
    if (now - lastSpeechTime < speechCooldown) return;
    
    if (window.speechSynthesis) {
        // Cancel active speaks to prevent queue delays
        window.speechSynthesis.cancel();
        
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        
        window.speechSynthesis.speak(utterance);
        lastSpeechTime = now;
    }
}

// 10. Load available dropdown poses from API
async function loadPosesDropdown() {
    try {
        const res = await fetch("/api/poses");
        const data = await res.json();
        
        const poses = data.poses || {};
        Object.entries(poses).forEach(([key, val]) => {
            const opt = document.createElement("option");
            opt.value = key;
            opt.innerText = val.display_name;
            poseDropdown.appendChild(opt);
        });
    } catch (e) {
        console.error("Failed to load poses dropdown config:", e);
    }
}

// 11. End Session & Show Summary
function showSessionSummary(summary) {
    if (!summary) return;
    
    // Build hold times list HTML
    let holdsHtml = "";
    Object.entries(summary.hold_times || {}).forEach(([pose, dur]) => {
        holdsHtml += `
            <div class="summary-hold-row">
                <span>${pose.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase())}</span>
                <span>${dur.toFixed(1)}s</span>
            </div>`;
    });
    
    if (!holdsHtml) {
        holdsHtml = `<div class="xai-empty">No pose holds completed.</div>`;
    }

    modalContent.innerHTML = `
        <div class="summary-stats-grid">
            <div class="summary-stat-box">
                <div class="summary-stat-val">${summary.average_overall_score}%</div>
                <div class="summary-stat-label">Average Session Score</div>
            </div>
            <div class="summary-stat-box">
                <div class="summary-stat-val">${(summary.total_practice_time_seconds / 60).toFixed(1)}m</div>
                <div class="summary-stat-label">Practice Duration</div>
            </div>
            <div class="summary-stat-box">
                <div class="summary-stat-val">${summary.average_symmetry_score}%</div>
                <div class="summary-stat-label">Bilateral Balance</div>
            </div>
            <div class="summary-stat-box">
                <div class="summary-stat-val">${summary.average_stability_score}%</div>
                <div class="summary-stat-label">Holding Stability</div>
            </div>
        </div>
        
        <div class="summary-holds-list">
            <h3>Successful Pose Holds</h3>
            ${holdsHtml}
        </div>`;
        
    summaryModal.classList.add("active");
}

// Event Listeners
poseDropdown.addEventListener('change', (e) => {
    activeTargetPose = e.target.value;
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(jsonStringify({
            action: "set_pose",
            pose_name: activeTargetPose
        }));
    }
});

btnStart.addEventListener('click', () => {
    isSessionActive = true;
    btnStart.disabled = true;
    btnEnd.disabled = false;
    
    // Send start signal
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(jsonStringify({
            action: "set_pose",
            pose_name: activeTargetPose
        }));
    }
    
    // Clear graphs
    scoreHistory.length = 0;
    timeHistory.length = 0;
    if (scoreChart) scoreChart.update();
});

btnEnd.addEventListener('click', () => {
    isSessionActive = false;
    btnStart.disabled = false;
    btnEnd.disabled = true;
    
    // Request session save from server
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(jsonStringify({
            action: "save_session"
        }));
    }
});

btnVoiceToggle.addEventListener('click', () => {
    isMuted = !isMuted;
    btnVoiceToggle.innerHTML = isMuted ? '<i class="fa-solid fa-volume-xmark"></i>' : '<i class="fa-solid fa-volume-high"></i>';
});

btnCloseModal.addEventListener('click', () => {
    summaryModal.classList.remove("active");
});

// Utilities
function jsonParseSafe(str) {
    try { return JSON.parse(str); } catch (e) { return null; }
}

function jsonStringify(obj) {
    try { return JSON.stringify(obj); } catch (e) { return ""; }
}

// Initialise
window.addEventListener('load', () => {
    initChart();
    loadPosesDropdown();
    initCamera();
});
