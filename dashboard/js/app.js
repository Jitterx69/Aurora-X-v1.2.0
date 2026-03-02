/**
 * AURORA-X Dashboard — Real-time WebSocket Client
 *
 * Connects to the AURORA-X API, receives live telemetry,
 * and renders ChartJS charts + DOM updates.
 */

// ============ Configuration ============
const WS_URL = `ws://${location.hostname || 'localhost'}:8000/ws`;
const API_BASE = `http://${location.hostname || 'localhost'}:8000/api`;
const MAX_CHART_POINTS = 120;
const RECONNECT_DELAY_MS = 3000;

// ============ State ============
let ws = null;
let sensorChart = null;
let faultChart = null;
let twinAnimFrame = null;
let logEntryCount = 0;
const sensorHistory = {
    labels: [],
    temperature: [],
    vibration: [],
    pressure: [],
    flow: [],
};

// ============ Chart Initialization ============
function initSensorChart() {
    const ctx = document.getElementById('sensor-chart').getContext('2d');
    sensorChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Temperature (°C)',
                    data: [],
                    borderColor: '#ffb800',
                    backgroundColor: 'rgba(255, 184, 0, 0.05)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true,
                    pointRadius: 0,
                },
                {
                    label: 'Vibration (mm/s)',
                    data: [],
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.05)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true,
                    pointRadius: 0,
                },
                {
                    label: 'Pressure (bar)',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.05)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true,
                    pointRadius: 0,
                },
                {
                    label: 'Flow (L/min)',
                    data: [],
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.05)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true,
                    pointRadius: 0,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 300 },
            scales: {
                x: {
                    display: true,
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { color: '#4a5568', maxTicksLimit: 10, font: { size: 10 } },
                },
                y: {
                    display: true,
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { color: '#4a5568', font: { size: 10 } },
                },
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#8892a8',
                        usePointStyle: true,
                        font: { size: 11 },
                        padding: 16,
                    },
                },
            },
            interaction: { intersect: false, mode: 'index' },
        },
    });
}

function initFaultChart() {
    const ctx = document.getElementById('fault-chart').getContext('2d');
    faultChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Normal', 'Bearing Wear', 'Misalignment', 'Cavitation', 'Overheating'],
            datasets: [{
                data: [90, 3, 2, 3, 2],
                backgroundColor: [
                    'rgba(0, 255, 136, 0.8)',
                    'rgba(0, 212, 255, 0.8)',
                    'rgba(123, 47, 247, 0.8)',
                    'rgba(255, 184, 0, 0.8)',
                    'rgba(255, 51, 102, 0.8)',
                ],
                borderWidth: 0,
                hoverOffset: 8,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: '#8892a8',
                        usePointStyle: true,
                        font: { size: 11 },
                        padding: 12,
                    },
                },
            },
        },
    });
}

// ============ Digital Twin Canvas ============
function drawTwinVisualization(data) {
    const canvas = document.getElementById('twin-canvas');
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    // Background grid
    ctx.strokeStyle = 'rgba(255,255,255,0.03)';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 20) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
    }
    for (let y = 0; y < h; y += 20) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }

    const time = Date.now() / 1000;
    const speed = data?.speed || 1500;
    const deg = data?.degradation || 0;
    const temp = data?.temperature || 80;

    // Draw rotating shaft
    const cx = w / 2;
    const cy = h / 2;
    const shaftRadius = 40;

    // Outer housing
    ctx.strokeStyle = `rgba(0, 212, 255, ${0.3 + deg * 0.5})`;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, shaftRadius + 20, 0, Math.PI * 2);
    ctx.stroke();

    // Shaft rotation
    const angle = (time * speed / 60 * Math.PI * 2) % (Math.PI * 2);
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angle);

    // Shaft body
    const grad = ctx.createRadialGradient(0, 0, 5, 0, 0, shaftRadius);
    grad.addColorStop(0, `rgba(0, 212, 255, 0.4)`);
    grad.addColorStop(1, `rgba(123, 47, 247, 0.1)`);
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(0, 0, shaftRadius, 0, Math.PI * 2);
    ctx.fill();

    // Shaft markers (like a compass cross)
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.6)';
    ctx.lineWidth = 2;
    for (let i = 0; i < 4; i++) {
        const a = (Math.PI / 2) * i;
        ctx.beginPath();
        ctx.moveTo(Math.cos(a) * 10, Math.sin(a) * 10);
        ctx.lineTo(Math.cos(a) * shaftRadius, Math.sin(a) * shaftRadius);
        ctx.stroke();
    }

    ctx.restore();

    // Temperature halo
    const tempNorm = Math.min(temp / 450, 1);
    ctx.strokeStyle = `rgba(255, ${Math.round(255 * (1 - tempNorm))}, 0, ${0.2 + tempNorm * 0.4})`;
    ctx.lineWidth = 3;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(cx, cy, shaftRadius + 30, 0, Math.PI * 2 * tempNorm);
    ctx.stroke();
    ctx.setLineDash([]);

    // Degradation indicator
    if (deg > 0.01) {
        ctx.fillStyle = `rgba(255, 51, 102, ${deg * 0.8})`;
        ctx.font = '11px JetBrains Mono';
        ctx.fillText(`Deg: ${(deg * 100).toFixed(1)}%`, 10, h - 10);
    }

    // Labels
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText(`${speed.toFixed(0)} RPM`, w - 80, 20);
    ctx.fillText(`${temp.toFixed(1)}°C`, w - 80, 35);

    twinAnimFrame = requestAnimationFrame(() => drawTwinVisualization(data));
}

// ============ WebSocket Connection ============
function connectWebSocket() {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        document.getElementById('connection-status').textContent = 'Live';
        document.querySelector('.status-dot').classList.add('live');
        addLogEntry('INFO', 'WebSocket connected to AURORA-X platform');
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleUpdate(data);
        } catch (e) {
            console.error('Parse error:', e);
        }
    };

    ws.onclose = () => {
        document.getElementById('connection-status').textContent = 'Disconnected';
        document.querySelector('.status-dot').classList.remove('live');
        addLogEntry('WARN', 'WebSocket disconnected. Reconnecting...');
        setTimeout(connectWebSocket, RECONNECT_DELAY_MS);
    };

    ws.onerror = (err) => {
        console.error('WebSocket error:', err);
        addLogEntry('ERROR', 'WebSocket connection error');
    };
}

// ============ Data Handling ============
function handleUpdate(data) {
    // Update metrics bar
    if (data.metrics) {
        document.getElementById('total-events').textContent = (data.metrics.events_processed || 0).toLocaleString();
        document.getElementById('events-per-sec').textContent = (data.metrics.events_per_second || 0).toFixed(1);
        const uptime = data.metrics.uptime_seconds || 0;
        document.getElementById('uptime').textContent = formatUptime(uptime);
    }

    // Process asset data
    if (data.assets) {
        const assetId = Object.keys(data.assets)[0];
        if (assetId) {
            updateAssetDashboard(data.assets[assetId]);
        }
    }

    // Process twin data
    if (data.twins) {
        const twinId = Object.keys(data.twins)[0];
        if (twinId) {
            updateTwinPanel(data.twins[twinId]);
        }
    }

    // Safety data
    if (data.safety) {
        updateSafetyPanel(data.safety);
    }
}

function updateAssetDashboard(assetData) {
    const state = assetData.state || assetData;
    const fault = assetData.fault_report || {};

    // Health cards
    const temp = state.temperature ?? state.temp ?? '--';
    const vib = state.vibration ?? state.vib ?? '--';
    const pressure = state.pressure ?? '--';
    const deg = state.degradation ?? 0;

    if (typeof temp === 'number') {
        document.getElementById('val-temp').textContent = temp.toFixed(1);
        document.getElementById('bar-temp').style.width = `${Math.min(temp / 450 * 100, 100)}%`;
    }

    if (typeof vib === 'number') {
        document.getElementById('val-vib').textContent = vib.toFixed(2);
        document.getElementById('bar-vib').style.width = `${Math.min(vib / 15 * 100, 100)}%`;
    }

    if (typeof pressure === 'number') {
        document.getElementById('val-pressure').textContent = pressure.toFixed(1);
        document.getElementById('bar-pressure').style.width = `${Math.min(pressure / 120 * 100, 100)}%`;
    }

    // Degradation
    const degPct = (deg * 100);
    document.getElementById('val-degradation').textContent = degPct.toFixed(1);
    document.getElementById('bar-degradation').style.width = `${Math.min(degPct, 100)}%`;

    // Health score
    const healthScore = Math.max(0, Math.round((1 - deg) * 100));
    document.getElementById('val-health').textContent = healthScore;
    const ringOffset = 220 - (220 * healthScore / 100);
    document.getElementById('health-ring-fill').setAttribute('stroke-dashoffset', ringOffset);

    // Sensor chart update
    const timeLabel = new Date().toLocaleTimeString('en-US', { hour12: false });
    sensorHistory.labels.push(timeLabel);
    sensorHistory.temperature.push(typeof temp === 'number' ? temp : 0);
    sensorHistory.vibration.push(typeof vib === 'number' ? vib * 10 : 0); // Scale for visibility
    sensorHistory.pressure.push(typeof pressure === 'number' ? pressure : 0);
    sensorHistory.flow.push(state.flow ?? 100);

    // Trim history
    if (sensorHistory.labels.length > MAX_CHART_POINTS) {
        sensorHistory.labels.shift();
        sensorHistory.temperature.shift();
        sensorHistory.vibration.shift();
        sensorHistory.pressure.shift();
        sensorHistory.flow.shift();
    }

    // Update chart
    if (sensorChart) {
        sensorChart.data.labels = sensorHistory.labels;
        sensorChart.data.datasets[0].data = sensorHistory.temperature;
        sensorChart.data.datasets[1].data = sensorHistory.vibration;
        sensorChart.data.datasets[2].data = sensorHistory.pressure;
        sensorChart.data.datasets[3].data = sensorHistory.flow;
        sensorChart.update('none');
    }

    // Fault distribution
    if (fault.fault_distribution && faultChart) {
        const dist = fault.fault_distribution;
        faultChart.data.datasets[0].data = [
            (dist.normal || 0) * 100,
            (dist.bearing_wear || 0) * 100,
            (dist.misalignment || 0) * 100,
            (dist.cavitation || 0) * 100,
            (dist.overheating || 0) * 100,
        ];
        faultChart.update();
    }

    // RL actions
    if (assetData.rl_action) {
        const act = assetData.rl_action;
        document.getElementById('rl-maintenance').style.width = `${(act.maintenance_intensity || 0) * 100}%`;
        document.getElementById('rl-maintenance-val').textContent = `${((act.maintenance_intensity || 0) * 100).toFixed(0)}%`;

        const speedPct = ((act.speed_adjustment || 0) + 0.2) / 0.4 * 100;
        document.getElementById('rl-speed').style.width = `${speedPct}%`;
        document.getElementById('rl-speed-val').textContent = `${((act.speed_adjustment || 0) * 100).toFixed(0)}%`;

        document.getElementById('rl-cooling').style.width = `${(act.cooling_adjustment || 0) * 100}%`;
        document.getElementById('rl-cooling-val').textContent = `${((act.cooling_adjustment || 0) * 100).toFixed(0)}%`;

        const loadPct = ((act.load_balance || 0) + 0.1) / 0.2 * 100;
        document.getElementById('rl-load').style.width = `${loadPct}%`;
        document.getElementById('rl-load-val').textContent = `${((act.load_balance || 0) * 100).toFixed(0)}%`;
    }
}

function updateTwinPanel(twinData) {
    if (!twinData) return;

    const speed = twinData.shaft_speed ?? twinData.speed ?? 1500;
    const bearingTemp = twinData.bearing_temp ?? twinData.temperature ?? 80;
    const sealIntegrity = (1 - (twinData.seal_degradation ?? 0)) * 100;
    const survival = (twinData.survival_probability ?? 0.99) * 100;

    document.getElementById('twin-speed').textContent = `${speed.toFixed(0)} rpm`;
    document.getElementById('twin-bearing-temp').textContent = `${bearingTemp.toFixed(1)} °C`;
    document.getElementById('twin-seal').textContent = `${sealIntegrity.toFixed(1)} %`;
    document.getElementById('twin-survival').textContent = `${survival.toFixed(1)} %`;

    // Update canvas visualization
    if (twinAnimFrame) cancelAnimationFrame(twinAnimFrame);
    drawTwinVisualization({
        speed: speed,
        temperature: bearingTemp,
        degradation: twinData.bearing_degradation ?? 0,
    });
}

function updateSafetyPanel(safetyData) {
    if (!safetyData) return;

    document.getElementById('safety-violations').textContent = safetyData.violations ?? 0;
    document.getElementById('safety-interventions').textContent = safetyData.interventions ?? 0;
    document.getElementById('safety-fallbacks').textContent = safetyData.fallbacks ?? 0;

    const totalIssues = (safetyData.violations ?? 0) + (safetyData.fallbacks ?? 0);
    const badge = document.getElementById('safety-status-badge');
    if (totalIssues > 10) {
        badge.textContent = 'DANGER';
        badge.className = 'panel-badge safety-badge danger';
    } else if (totalIssues > 0) {
        badge.textContent = 'WARNING';
        badge.className = 'panel-badge safety-badge warning';
    } else {
        badge.textContent = 'SAFE';
        badge.className = 'panel-badge safety-badge';
    }
}

// ============ Event Log ============
function addLogEntry(level, message) {
    const container = document.getElementById('event-log');
    const entry = document.createElement('div');
    const levelClass = level === 'WARN' ? 'log-warn' : level === 'ERROR' ? 'log-error' :
        level === 'FAULT' ? 'log-fault' : 'log-info';
    entry.className = `log-entry ${levelClass}`;

    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    entry.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-level">${level}</span>
        <span class="log-msg">${message}</span>
    `;

    container.insertBefore(entry, container.firstChild);
    logEntryCount++;

    // Limit entries
    if (logEntryCount > 100) {
        container.removeChild(container.lastChild);
    }
}

// ============ Helpers ============
function formatUptime(seconds) {
    if (seconds < 60) return `${seconds.toFixed(0)}s`;
    if (seconds < 3600) return `${(seconds / 60).toFixed(0)}m`;
    return `${(seconds / 3600).toFixed(1)}h`;
}

// ============ Fallback: REST Polling ============
async function pollAPI() {
    try {
        const res = await fetch(`${API_BASE}/status`);
        const data = await res.json();
        handleUpdate({ metrics: data.metrics });
    } catch (e) {
        // API not available yet
    }
}

// ============ Initialization ============
document.addEventListener('DOMContentLoaded', () => {
    initSensorChart();
    initFaultChart();
    connectWebSocket();

    // Start twin visualization with defaults
    drawTwinVisualization({ speed: 1500, temperature: 80, degradation: 0 });

    // Fallback polling
    setInterval(pollAPI, 5000);

    addLogEntry('INFO', 'AURORA-X Dashboard initialized');
    addLogEntry('INFO', 'Waiting for platform data stream...');
});
