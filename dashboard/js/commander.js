class AuroraCommander {
    constructor() {
        this.socket = null;
        this.uptimeStart = Date.now();
        this.currentView = 'tactical';
        this.config = null;
        this.isLogPaused = false;
        this.eventLogBuffer = [];
        this.init();
    }

    init() {
        this.setupNavigation();
        this.setupTerminal();
        this.setupResizer();
        this.setupParameterSliders();
        this.setupLogControls();
        this.connectWebSocket();
        this.startUptimeTicker();
        this.refreshAudit();
        console.log("AURORA-X Commander Initialized.");
    }

    /* ============ Navigation ============ */
    setupNavigation() {
        const navItems = document.querySelectorAll('.nav-item');
        navItems.forEach(item => {
            item.addEventListener('click', () => {
                const view = item.getAttribute('data-view');
                this.switchView(view);
                navItems.forEach(i => i.classList.remove('active'));
                item.classList.add('active');
            });
        });

        // Settings sidebar navigation
        document.querySelectorAll('.settings-nav-item').forEach(item => {
            item.addEventListener('click', () => {
                const section = item.getAttribute('data-section');
                this.switchSettingsSection(section);
                document.querySelectorAll('.settings-nav-item').forEach(i => i.classList.remove('active'));
                item.classList.add('active');
            });
        });

        // Settings buttons
        const btnSave = document.getElementById('btn-save-settings');
        if (btnSave) btnSave.addEventListener('click', () => this.saveSettings());

        const btnReset = document.getElementById('btn-reset-settings');
        if (btnReset) btnReset.addEventListener('click', () => this.refreshConfig());
    }

    switchView(viewId) {
        this.currentView = viewId;
        document.querySelectorAll('.view-container').forEach(v => v.classList.remove('active'));
        document.getElementById(`view-${viewId}`).classList.add('active');
        document.getElementById('header-title').innerText = viewId.toUpperCase().replace('-', ' ');

        if (viewId === 'security') this.refreshAudit();
        if (viewId === 'settings') {
            this.refreshSecurityStatus();
            this.refreshConfig();
        }
        if (viewId === 'reviews') this.refreshReviewQueue();
        if (viewId === 'log') this.refreshEventLog();
    }

    async refreshSecurityStatus() {
        try {
            const resp = await fetch('/api/security/status');
            this.securityStatus = await resp.json();
            console.log("Security Status Updated:", this.securityStatus);
        } catch (e) { console.error("Failed to fetch security status", e); }
    }

    switchSettingsSection(sectionId) {
        const title = document.getElementById('settings-section-title');
        if (title) title.innerText = sectionId.toUpperCase().replace('_', ' ') + " SETTINGS";

        if (sectionId === 'security') {
            this.renderSettingsForm(sectionId);
            return;
        }

        if (!this.config) {
            const container = document.getElementById('settings-form-container');
            if (container) container.innerHTML = '<div class="loading-spinner">WAITING FOR CONFIG...</div>';
            return;
        }
        this.renderSettingsForm(sectionId);
    }

    /* ============ WebSocket Telemetry ============ */
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        this.socket = new WebSocket(wsUrl);

        this.socket.onopen = () => {
            this.logTerminal("CONNECTION ESTABLISHED TO CORE SERVICES.");
            document.getElementById('status-dot').classList.add('live');
            document.getElementById('status-text').innerText = "LIVE STREAM";
            document.getElementById('offline-overlay').style.display = 'none';
        };

        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.updateTactical(data);
            this.updateParameters(data);

            // Handle mode changes from backend
            if (data.mode && data.mode !== this.currentMode) {
                this.currentMode = data.mode;
                this.applyModeUI(data.mode);
            }

            // Route to new views
            if (this.currentView === 'twin') this.updateTwin(data);
            if (this.currentView === 'faults') this.updateFaults(data);
            if (this.currentView === 'log' && !this.isLogPaused) this.updateEventLogWS(data);
        };

        this.socket.onclose = () => {
            this.logTerminal("CONNECTION LOST. RECONNECTING...", "err");
            document.getElementById('status-dot').classList.remove('live');
            document.getElementById('status-text').innerText = "OFFLINE";
            document.getElementById('offline-overlay').style.display = 'flex';
            setTimeout(() => this.connectWebSocket(), 3000);
        };
    }

    updateTactical(data) {
        if (!data.assets) return;

        const assetId = Object.keys(data.assets)[0];
        if (!assetId) return;

        const latest = data.assets[assetId];
        const sensors = latest.event?.sensors || {};
        const twin = data.twins?.[assetId] || {};

        // KPI Cards
        if (sensors.pressure != null) {
            document.getElementById('val-load').innerHTML = `${sensors.pressure.toFixed(2)}<span class="card-unit">%</span>`;
            document.getElementById('bar-load').style.width = `${Math.min(sensors.pressure, 100)}%`;
        }
        if (sensors.thermal != null) {
            document.getElementById('val-temp').innerHTML = `${sensors.thermal.toFixed(1)}<span class="card-unit">°C</span>`;
            document.getElementById('bar-temp').style.width = `${Math.min(sensors.thermal, 100)}%`;
        }
        if (twin.rul?.rul_mean != null) {
            const rulHrs = twin.rul.rul_mean;
            const rulDisplay = rulHrs > 1000 ? `${(rulHrs / 1000).toFixed(1)}K` : rulHrs.toFixed(0);
            document.getElementById('val-rul').innerHTML = `${rulDisplay}<span class="card-unit">HRS</span>`;
            document.getElementById('bar-rul').style.width = `${Math.min(rulHrs / 500, 100)}%`;
        }
        if (twin.health_score != null) {
            document.getElementById('val-health').innerHTML = `${twin.health_score.toFixed(1)}<span class="card-unit">%</span>`;
            document.getElementById('bar-health').style.width = `${Math.min(twin.health_score, 100)}%`;
        }

        // Fused State Vector
        const stateLabels = ['RPM', 'OMEGA', 'TEMPERATURE', 'PRESSURE', 'DEGRADATION', 'HAZARD RATE', 'LOAD', 'WEAR INDEX'];
        const stateDisplay = document.getElementById('state-vector-display');
        if (twin.physics_state && Array.isArray(twin.physics_state)) {
            stateDisplay.innerHTML = '';
            twin.physics_state.forEach((val, i) => {
                const label = stateLabels[i] || `STATE_${i}`;
                const line = document.createElement('div');
                line.className = 'state-line';
                line.innerHTML = `<span class="state-label">${label}</span><span class="state-val">${val.toFixed(4)}</span>`;
                stateDisplay.appendChild(line);
            });
        }

        // Fault Ensemble
        if (latest.fault_report) {
            const report = latest.fault_report;
            const severity = (report.severity_index || 0) * 100;
            const dist = report.fault_distribution || {};

            const badge = document.getElementById('severity-badge');
            const level = report.severity_level || (severity < 20 ? 'nominal' : severity < 50 ? 'minor' : 'critical');
            badge.textContent = (report.primary_fault || 'NOMINAL').toUpperCase();
            badge.className = `severity-badge severity-${level === 'none' || level === 'nominal' ? 'nominal' : level === 'minor' ? 'minor' : 'critical'}`;

            const container = document.getElementById('fault-distribution');
            container.innerHTML = '';
            Object.entries(dist).sort((a, b) => b[1] - a[1]).forEach(([fault, prob]) => {
                const pct = (prob * 100).toFixed(1);
                const barClass = fault === 'normal' ? 'fault-normal' : prob > 0.3 ? 'fault-high' : prob > 0.15 ? 'fault-mid' : 'fault-low';
                const row = document.createElement('div');
                row.className = 'fault-bar-row';
                row.innerHTML = `<span class="fault-bar-label">${fault}</span><div class="fault-bar-track"><div class="fault-bar-fill ${barClass}" style="width: ${pct}%"></div></div><span class="fault-bar-pct">${pct}%</span>`;
                container.appendChild(row);
            });
        }

        // RL Safety
        if (data.safety) {
            const s = data.safety;
            const totalChecks = (s.violations || 0) + (s.interventions || 0) + (s.fallbacks || 0);
            document.getElementById('safety-checks').textContent = totalChecks.toLocaleString();
            document.getElementById('safety-violations').textContent = (s.violations || 0).toLocaleString();
            document.getElementById('safety-barriers').textContent = (s.interventions || 0).toLocaleString();
            const rate = totalChecks > 0 ? ((1 - s.violations / totalChecks) * 100).toFixed(1) : '100.0';
            document.getElementById('safety-rate').textContent = `${rate}%`;
        }

        // Metrics Strip
        if (data.metrics) {
            document.getElementById('metric-eps').textContent = (data.metrics.events_per_second || 0).toFixed(1);
            document.getElementById('metric-total').textContent = (data.metrics.events_processed || 0).toLocaleString();
        }
        if (data.gateway) document.getElementById('metric-gateway').textContent = (data.gateway.events_processed || 0).toLocaleString();
        if (data.event_log_size != null) document.getElementById('metric-log').textContent = data.event_log_size.toLocaleString();
    }

    /* ============ Digital Twin View ============ */
    updateTwin(data) {
        if (!data.assets) return;
        const selector = document.getElementById('twin-asset-selector');
        const assetId = selector ? selector.value : Object.keys(data.assets)[0];
        const twin = data.twins?.[assetId];

        // Update Twin Mode Badge
        const modeBadge = document.getElementById('twin-mode-badge');
        if (modeBadge && data.twin_mode) {
            modeBadge.innerText = data.twin_mode.toUpperCase();
            if (data.twin_mode === 'simulation') {
                modeBadge.style.background = 'rgba(255, 167, 38, 0.1)';
                modeBadge.style.color = '#ffa726';
            } else {
                modeBadge.style.background = 'rgba(0, 255, 0, 0.1)';
                modeBadge.style.color = '#4caf50';
            }
        }

        if (!twin) return;

        // Health Gauge
        const health = twin.health_score || 0;
        document.getElementById('twin-gauge-val').innerText = `${health.toFixed(0)}%`;
        const offset = (1 - health / 100) * 283;
        document.getElementById('twin-gauge-fill').style.strokeDasharray = `${283 - offset}, 283`;

        // Stats
        document.getElementById('twin-deep-survival').innerText = `${((twin.survival_probability || 0) * 100).toFixed(1)}%`;
        document.getElementById('twin-deep-degradation').innerText = twin.degradation != null ? twin.degradation.toExponential(3) : '0.000e+0';
        document.getElementById('twin-deep-rul').innerText = twin.rul ? `${twin.rul.rul_mean.toFixed(0)} HRS` : '— HRS';

        // Physics Grid
        const stateLabels = ['RPM', 'OMEGA', 'TEMP', 'PRES', 'DEGR', 'HAZA', 'LOAD', 'WEAR'];
        const grid = document.getElementById('twin-physics-grid');
        grid.innerHTML = '';
        if (twin.physics_state) {
            twin.physics_state.forEach((val, i) => {
                const item = document.createElement('div');
                item.className = 'twin-stat';
                item.innerHTML = `<div class="twin-stat-label">${stateLabels[i] || `IDX-${i}`}</div><div class="twin-stat-value" style="font-size: 14px">${val.toFixed(5)}</div>`;
                grid.appendChild(item);
            });
        }

        // Degradation Trend Bars
        const trendCanvas = document.getElementById('twin-trend-canvas');
        if (trendCanvas && twin.degradation_trend) {
            trendCanvas.innerHTML = '';
            twin.degradation_trend.forEach((val, i) => {
                const bar = document.createElement('div');
                bar.style.flex = '1';
                // Scale height to failure threshold (usually 0.8+) - 100% height = 0.8 degradation
                const height = Math.min(100, (val / 0.8) * 100);
                bar.style.height = `${Math.max(2, height)}%`;
                bar.style.background = 'var(--aurora-blue)';

                // Opacity logic: Past (dims), Present (full), Future (glassy/predictive)
                if (i < 4) {
                    bar.style.opacity = (0.2 + (i * 0.1)).toFixed(1); // Past fades in
                } else if (i === 4) {
                    bar.style.opacity = '1'; // Present is solid
                    bar.style.boxShadow = '0 0 10px rgba(0, 229, 255, 0.5)';
                } else {
                    bar.style.opacity = '0.4'; // Future is predictive glass
                    bar.style.border = '1px dashed var(--aurora-blue)';
                    bar.style.background = 'transparent';
                }

                trendCanvas.appendChild(bar);
            });
        }
    }

    /* ============ Fault Analysis View ============ */
    updateFaults(data) {
        if (!data.assets) return;
        const assetId = Object.keys(data.assets)[0];
        const latest = data.assets[assetId];
        if (!latest.fault_report) return;

        const report = latest.fault_report;
        const severity = (report.severity_index || 0) * 100;

        // 1. Severity Distribution Bars (Probability across types)
        const container = document.getElementById('severity-dist-bars');
        if (container) {
            container.innerHTML = '';
            if (report.fault_distribution) {
                Object.entries(report.fault_distribution).forEach(([type, prob]) => {
                    if (type === 'normal') return;
                    const bar = document.createElement('div');
                    const h = Math.max(5, prob * 100);
                    const color = prob > 0.4 ? '#ff4d4d' : prob > 0.1 ? '#ffa726' : 'var(--aurora-blue)';
                    bar.style.cssText = `flex:1; height:${h}%; background:${color}; opacity:0.8; min-width:10px; border-radius: 2px 2px 0 0;`;
                    bar.title = `${type.replace('_', ' ').toUpperCase()}: ${(prob * 100).toFixed(1)}%`;
                    container.appendChild(bar);
                });
            }
        }

        // 2. Ensemble Breakdown (Dynamic Bars)
        if (report.residual_analysis && report.ml_prediction) {
            // Mapping backend confidence/weights to the 3 bars
            const resVal = (report.residual_analysis.max_severity || 0) * 100;
            const mlVal = (report.confidence?.ml_confidence || 0) * 100;
            const tempVal = (report.confidence?.temporal_confidence || 0) * 100;

            this.updateEnsembleBar('residual', resVal);
            this.updateEnsembleBar('ml', mlVal);
            this.updateEnsembleBar('temporal', tempVal);
        }

        // 3. Root Cause Attribution (XAI)
        const attrList = document.getElementById('fault-attribution-list');
        if (attrList && report.attribution) {
            attrList.innerHTML = '';
            Object.entries(report.attribution).forEach(([sensor, score]) => {
                const row = document.createElement('div');
                row.className = 'attribution-row';
                const pct = (score * 100).toFixed(0);
                row.innerHTML = `
                <div class="attribution-info">
                    <span class="card-label" style="margin-bottom:0">${sensor.toUpperCase()}</span>
                    <span style="color:var(--text-prime); font-weight:bold">${pct}%</span>
                </div>
                <div class="monochrome-bar" style="height: 3px;">
                    <div class="monochrome-bar-fill" style="width: ${pct}%; background: var(--aurora-blue)"></div>
                </div>
            `;
                attrList.appendChild(row);
            });
            if (Object.keys(report.attribution).length === 0) {
                attrList.innerHTML = '<div class="timeline-empty" style="font-size: 10px;">SIGNALS WITHIN NOMINAL TOLERANCE</div>';
            }
        }

        // 4. Recommendations
        const recList = document.getElementById('fault-recommendations');
        if (recList && report.recommendations) {
            recList.innerHTML = '';
            report.recommendations.forEach(rec => {
                const item = document.createElement('div');
                item.className = 'recommendation-item';
                item.innerText = rec;
                recList.appendChild(item);
            });
        }

        // 5. Fault Timeline (Fetch history)
        this.refreshFaultTimeline(assetId);

        // Active Faults Count Badge
        const activeCount = Object.values(report.fault_distribution || {}).filter(p => p > 0.1).length - (report.fault_distribution?.normal > 0.7 ? 1 : 0);
        const badge = document.getElementById('fault-count-badge');
        if (badge) {
            badge.innerText = `${Math.max(0, activeCount)} ACTIVE SIGNATURES`;
            badge.style.color = activeCount > 0 ? '#ff4d4d' : '#4caf50';
        }
    }

    updateEnsembleBar(type, value) {
        const idMap = { 'residual': 'bar-res-weight', 'ml': 'bar-ml-weight', 'temporal': 'bar-temp-weight' };
        const el = document.getElementById(idMap[type]);
        if (el) {
            el.style.width = `${value}%`;
        }
    }

    async refreshFaultTimeline(assetId) {
        const container = document.getElementById('fault-timeline');
        if (!container) return;

        try {
            const resp = await fetch(`/api/assets/${assetId}/faults?n=10`);
            const data = await resp.json();

            if (data.faults && data.faults.length > 0) {
                container.innerHTML = '';
                // Show latest first
                data.faults.reverse().forEach(report => {
                    if (report.primary_fault === 'normal' && report.severity_index < 0.2) return;

                    const card = document.createElement('div');
                    card.className = `fault-event severity-${report.severity_level || 'minor'}`;

                    // Fix: Handle missing or zero timestamp
                    const timestamp = report.timestamp || (Date.now() / 1000);
                    const date = new Date(timestamp * 1000).toLocaleTimeString();

                    card.innerHTML = `
                    <div class="fault-event-header">
                        <span class="fault-event-time">${date}</span>
                        <span class="fault-event-level" style="color: ${report.severity_index > 0.6 ? '#ff4d4d' : '#ffa726'}">
                            ${report.severity_level.toUpperCase()}
                        </span>
                    </div>
                    <div class="fault-event-title">${report.primary_fault.replace('_', ' ').toUpperCase()} IRREGULARITY DETECTED</div>
                    <div class="fault-event-meta">
                        Probability Core: ${(report.primary_probability * 100).toFixed(1)}% | 
                        Ensemble Validation: Verified
                    </div>
                `;
                    container.appendChild(card);
                });

                if (container.innerHTML === '') {
                    container.innerHTML = '<div class="timeline-empty">NO FAULTS DETECTED IN CURRENT WINDOW</div>';
                }
            }
        } catch (e) {
            console.error("Failed to refresh fault timeline", e);
        }
    }

    /* ============ Event Log View ============ */
    setupLogControls() {
        const btnPause = document.getElementById('btn-pause-log');
        if (btnPause) {
            btnPause.addEventListener('click', () => {
                this.isLogPaused = !this.isLogPaused;
                btnPause.innerText = this.isLogPaused ? "RESUME" : "PAUSE";
                btnPause.style.background = this.isLogPaused ? "rgba(255, 150, 0, 0.1)" : "";
            });
        }
        const btnClear = document.getElementById('btn-clear-log');
        if (btnClear) {
            btnClear.addEventListener('click', () => {
                document.getElementById('event-stream').innerHTML = '';
            });
        }
    }

    async refreshEventLog() {
        try {
            const resp = await fetch('/api/events/stream?limit=50');
            const data = await resp.json();
            const container = document.getElementById('event-stream');
            container.innerHTML = '';
            data.events.forEach(ev => this.appendEventRow(ev));

            // Stats
            const statsResp = await fetch('/api/pipeline/stats');
            const stats = await statsResp.json();
            this.updatePipelineUI(stats);
        } catch (e) { console.error("Event log fetch failed", e); }
    }

    updatePipelineUI(stats) {
        if (stats.gateway) {
            const rej = stats.gateway.events_rejected || 0;
            const total = stats.gateway.events_processed || 1;
            document.getElementById('pipe-rejection').innerText = `${(rej / total * 100).toFixed(2)}%`;
            document.getElementById('pipe-compression').innerText = stats.gateway.compression ? "LZ4" : "NONE";
        }
        if (stats.pipeline) document.getElementById('pipe-window').innerText = stats.pipeline.window_size;
        if (stats.event_log) {
            const occ = (stats.event_log.size / stats.event_log.max * 100).toFixed(2);
            document.getElementById('pipe-cache').innerText = `${occ}%`;
        }

        // Sensor health mock
        const grid = document.getElementById('sensor-health-grid');
        grid.innerHTML = '';
        ['VIB', 'THR', 'PRE', 'FLO', 'SPD', 'AMP', 'VOL', 'ACO', 'HUM'].forEach(s => {
            const hex = document.createElement('div');
            hex.className = 'sensor-hex active';
            hex.innerText = s;
            grid.appendChild(hex);
        });
    }

    updateEventLogWS(data) {
        if (!data.assets) return;
        const assetId = Object.keys(data.assets)[0];
        const latest = data.assets[assetId];
        this.appendEventRow({
            asset_id: assetId,
            timestamp: latest.event?.timestamp || Date.now() / 1000,
            sensors: latest.event?.sensors || {},
            fault_type: latest.fault_report?.primary_fault
        });
    }

    appendEventRow(ev) {
        const container = document.getElementById('event-stream');
        if (!container) return;
        const row = document.createElement('div');
        row.className = `event-row ${ev.fault_type && ev.fault_type !== 'normal' ? 'fault' : ''}`;
        const time = new Date(ev.timestamp * 1000).toLocaleTimeString();
        const sensorData = Object.entries(ev.sensors).map(([k, v]) => `${k}=${v.toFixed(1)}`).join(' ');
        row.innerHTML = `<span class="event-ts">[${time}]</span><span class="event-id">${ev.asset_id}</span><span class="event-data">${sensorData}</span>`;
        container.prepend(row);
        if (container.children.length > 100) container.lastChild.remove();
    }

    /* ============ Settings Management ============ */
    async refreshConfig() {
        const container = document.getElementById('settings-form-container');
        container.innerHTML = '<div class="loading-spinner">FETCHING PLATFORM CONFIGURATION...</div>';
        try {
            await this.refreshSecurityStatus(); // Sync security first
            const resp = await fetch('/api/config');
            const data = await resp.json();
            this.config = data.config;
            this.applyModeUI(this.config.platform.mode);
            this.switchSettingsSection('platform');
        } catch (e) {
            container.innerHTML = `<div class="err">FAILED TO LOAD CONFIG: ${e.message}</div>`;
        }
    }

    applyModeUI(mode) {
        this.currentMode = mode;
        const terminal = document.getElementById('view-terminal');
        const emergencyOverlay = document.getElementById('emergency-overlay');
        const saveBtn = document.getElementById('btn-save-settings');

        // Reset
        emergencyOverlay.style.display = 'none';
        if (terminal) terminal.classList.remove('restricted-view');
        if (saveBtn) {
            saveBtn.innerText = "APPLY CONFIGURATION";
            saveBtn.style.background = "";
        }

        if (mode === 'emergency') {
            emergencyOverlay.style.display = 'flex';
        } else if (mode === 'production') {
            if (terminal) terminal.classList.add('restricted-view');
            if (saveBtn) {
                saveBtn.innerText = "PUSH FOR REVIEW";
                saveBtn.style.background = "var(--border-medium)";
            }
        } else if (mode === 'staging') {
            if (saveBtn) {
                saveBtn.innerText = "PUSH FOR REVIEW";
            }
        } else if (mode === 'maintenance') {
            // Limited access handled by hiding specific sidebar items if needed
        }
    }

    renderSettingsForm(sectionId) {
        const container = document.getElementById('settings-form-container');
        container.innerHTML = '';

        if (sectionId === 'security') {
            this.renderSecuritySection(container);
            return;
        }

        const section = this.config[sectionId];
        if (!section) {
            container.innerHTML = `<div class="timeline-empty">SECTION '${sectionId.toUpperCase()}' NOT FOUND IN YAML</div>`;
            return;
        }

        Object.entries(section).forEach(([key, val]) => {
            const group = document.createElement('div');
            group.className = 'setting-group';

            const label = document.createElement('label');
            label.className = 'setting-label';
            label.innerText = key.replace(/_/g, ' ');
            group.appendChild(label);

            let control;
            const dropdowns = {
                'platform.mode': ['development', 'staging', 'production', 'maintenance', 'emergency'],
                'ingestion.broker.backend': ['memory', 'kafka'],
                'pipeline.anomaly_scorer.method': ['zscore', 'isolation_forest'],
                'estimation.kalman.type': ['ekf', 'ukf'],
                'digital_twin.solver': ['rk4', 'euler'],
                'digital_twin.mode': ['realtime', 'simulation'],
                'fault_detection.temporal.model': ['lstm', 'transformer'],
                'rl.algorithm': ['cppo', 'sac'],
                'storage.timeseries.backend': ['sqlite', 'timescaledb'],
                'storage.cache.backend': ['memory', 'redis'],
                'storage.model_registry.backend': ['local', 'mlflow']
            };

            const fullKey = `${sectionId}.${key}`;

            if (typeof val === 'boolean') {
                control = document.createElement('select');
                control.className = 'input-field setting-control';
                control.innerHTML = `<option value="true" ${val ? 'selected' : ''}>ENABLED</option><option value="false" ${!val ? 'selected' : ''}>DISABLED</option>`;
            } else if (dropdowns[fullKey]) {
                control = document.createElement('select');
                control.className = 'input-field setting-control';

                // Tier-based permission masking
                const allowed = this.securityStatus?.allowed_modes || [];
                control.innerHTML = dropdowns[fullKey].map(opt => {
                    const isRestricted = fullKey === 'platform.mode' && !allowed.includes(opt);
                    return `<option value="${opt}" ${val === opt ? 'selected' : ''} ${isRestricted ? 'disabled' : ''}>${opt.toUpperCase()} ${isRestricted ? '(RESTRICTED)' : ''}</option>`;
                }).join('');
            } else if (typeof val === 'number') {
                control = document.createElement('input');
                control.type = 'number';
                control.step = 'any';
                control.className = 'input-field setting-control';
                control.value = val;
            } else {
                control = document.createElement('input');
                control.type = 'text';
                control.className = 'input-field setting-control';
                control.value = typeof val === 'object' ? JSON.stringify(val) : val;
            }

            control.dataset.section = sectionId;
            control.dataset.key = key;
            group.appendChild(control);
            container.appendChild(group);
        });
    }

    renderSecuritySection(container) {
        const status = this.securityStatus || { tier: 'junior', key_active: false };
        const tierClass = `tier-${status.tier}`;

        container.innerHTML = `
            <div class="setting-group" style="padding-bottom: 20px; border-bottom: 1px solid var(--border-dim);">
                <label class="setting-label">ACTIVE SECURITY TIER</label>
                <div style="display: flex; align-items: center; gap: 15px;">
                    <span class="tier-badge ${tierClass}">${status.tier.toUpperCase()} LEVEL ACCESS</span>
                    <span style="font-size: 11px; opacity: 0.5;">${status.key_active ? 'VERIFIED HARDWARE KEY' : 'DEFAULT ACCESS'}</span>
                </div>
            </div>
            
            <div class="setting-group" style="margin-top: 30px;">
                <label class="setting-label">SOFTWARE ACTIVATION KEY</label>
                <p style="font-size: 11px; color: var(--text-dim); margin-bottom: 15px;">Enter your unique 12-char activation code to upgrade your engineering privilege level.</p>
                <div style="display: flex; gap: 10px;">
                    <input type="password" id="software-key-input" class="input-field" placeholder="AX-XXXX-XXXX" style="flex: 1;">
                    <button class="btn-glass" onclick="window.commander.activateKey()">ACTIVATE</button>
                </div>
                <div id="activation-status" style="margin-top: 10px; font-size: 11px;"></div>
            </div>
            
            <div class="panel" style="margin-top: 40px; background: rgba(255,167,38,0.05); border: 1px solid rgba(255,167,38,0.2);">
                <div class="panel-body" style="font-size: 11px; color: #ffa726;">
                    <strong>MASTER OVERRIDE NOTICE:</strong> Activation keys are cryptographically bound to hardware. Changes require Master-level biometric signature verification.
                </div>
            </div>
        `;
    }

    async activateKey() {
        const key = document.getElementById('software-key-input').value;
        const status = document.getElementById('activation-status');
        status.innerText = "VALIDATING KEY...";
        try {
            const resp = await fetch('/api/security/activate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ key: key })
            });
            const res = await resp.json();
            if (res.status === 'SUCCESS') {
                status.innerText = `SUCCESS: ACCESS UPGRADED TO ${res.tier.toUpperCase()}`;
                status.style.color = "#4caf50";
                setTimeout(() => this.refreshConfig(), 1500);
            } else {
                status.innerText = `ERROR: ${res.message}`;
                status.style.color = "#ff4d4d";
            }
        } catch (e) { status.innerText = "ERROR: NETWORK FAILURE"; }
    }

    async saveSettings(reviewed = false, biometric_verified = false) {
        const updates = {};
        let isModeChange = false;

        document.querySelectorAll('.setting-control').forEach(ctrl => {
            const sec = ctrl.dataset.section;
            const key = ctrl.dataset.key;
            let val = ctrl.value;

            if (ctrl.type === 'number') val = parseFloat(val);
            if (ctrl.tagName === 'SELECT' && (val === 'true' || val === 'false')) val = val === 'true';

            if (!updates[sec]) updates[sec] = {};

            // Detect mode changes for biometric prompting
            if (sec === 'platform' && key === 'mode' && val !== this.currentMode) {
                isModeChange = true;
            }

            updates[sec][key] = val;
        });

        // Trigger Biometric for non-junior tiers on mode switch
        if (isModeChange && this.securityStatus?.tier !== 'junior' && !biometric_verified) {
            this.showBiometricPrompt(() => this.saveSettings(reviewed, true));
            return;
        }

        const btn = document.getElementById('btn-save-settings');
        const isPush = (this.currentMode === 'production' || this.currentMode === 'staging') && !reviewed;

        btn.innerText = isPush ? "PUSHING FOR REVIEW..." : "APPLYING...";

        try {
            const url = `/api/config?biometric_verified=${biometric_verified}${reviewed ? '&reviewed=true' : ''}`;
            const resp = await fetch(url, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(updates)
            });
            const res = await resp.json();

            if (res.status === 'CONFIG_UPDATED') {
                btn.innerText = "SUCCESS";
                btn.style.borderColor = "#4caf50";
                setTimeout(() => {
                    this.applyModeUI(this.currentMode);
                    btn.style.borderColor = "";
                }, 2000);
                this.config = updates;
                this.logTerminal(`CONFIG UPDATED: ${res.count} parameters changed.`, "info");
            } else if (res.status === 'REVIEW_REQUIRED') {
                btn.innerText = "PUSHING FOR REVIEW...";
                await this.submitPushForReview(updates);
                btn.innerText = "PUSHED FOR REVIEW";
                btn.style.borderColor = "#ffa726";
                setTimeout(() => {
                    this.applyModeUI(this.currentMode);
                    btn.style.borderColor = "";
                }, 2000);
            }
        } catch (e) {
            btn.innerText = "FAILED";
            console.error(e);
        }
    }

    /* ============ Biometric Verification UI ============ */
    showBiometricPrompt(callback) {
        const modal = document.getElementById('biometric-modal');
        const progress = document.getElementById('biometric-progress');
        const btn = document.getElementById('btn-mock-scan');

        modal.style.display = 'flex';
        progress.innerText = "PLACE FINGER ON SENSOR";
        progress.className = "biometric-status";
        btn.disabled = false;
        btn.innerText = "START SCAN";

        this.onBiometricSuccess = () => {
            modal.style.display = 'none';
            callback();
        };
    }

    async verifyBiometric() {
        const progress = document.getElementById('biometric-progress');
        const btn = document.getElementById('btn-mock-scan');

        btn.disabled = true;
        btn.innerText = "SCANNING...";

        try {
            const resp = await fetch('/api/security/biometric', { method: 'POST' });
            const data = await resp.json();

            if (data.status === 'BIOMETRIC_MATCH') {
                setTimeout(() => {
                    progress.innerText = "BIOMETRIC MATCH CONFIRMED";
                    progress.className = "biometric-status biometric-success";
                    setTimeout(() => this.onBiometricSuccess(), 1000);
                }, 1500);
            }
        } catch (e) {
            progress.innerText = "HARDWARE SENSOR ERROR";
            btn.disabled = false;
        }
    }

    /* ============ Terminal & Parameters (Legacy) ============ */
    setupParameterSliders() {
        const mapping = { 'fi-severity': 'fi-severity-val', 'param-gamma': 'param-gamma-val', 'param-epsilon': 'param-epsilon-val' };
        Object.entries(mapping).forEach(([sid, vid]) => {
            const s = document.getElementById(sid);
            if (s) s.addEventListener('input', () => {
                let v = s.value;
                if (sid !== 'fi-severity') v = (v / 100).toFixed(2);
                else v += '%';
                document.getElementById(vid).textContent = v;
            });
        });
    }

    updateParameters(data) {
        if (!data.rl_stats) return;
        const rl = data.rl_stats;
        document.getElementById('rl-algorithm').textContent = (rl.algorithm || '—').toUpperCase();
        document.getElementById('rl-episodes').textContent = (rl.episodes || 0).toLocaleString();
        document.getElementById('rl-steps').textContent = (rl.total_steps || 0).toLocaleString();
        document.getElementById('rl-best-reward').textContent = rl.best_reward != null ? rl.best_reward.toFixed(2) : '—';
        document.getElementById('rl-avg-reward').textContent = (rl.avg_reward_last_50 || 0).toFixed(2);
        document.getElementById('rl-buffer').textContent = (rl.buffer_size || 0).toLocaleString();

        const badge = document.getElementById('rl-trained-badge');
        badge.textContent = rl.trained ? 'TRAINED' : 'UNTRAINED';
        badge.className = `severity-badge severity-${rl.trained ? 'nominal' : 'minor'}`;

        if (data.barriers) {
            const assetId = Object.keys(data.barriers)[0];
            const barriers = data.barriers[assetId];
            if (barriers) {
                const map = { temperature: 'temp', vibration: 'vib', pressure: 'pres' };
                for (const [name, p] of Object.entries(map)) {
                    const b = barriers[name];
                    if (!b) continue;
                    document.getElementById(`barrier-${p}-current`).textContent = b.current.toFixed(1);
                    document.getElementById(`barrier-${p}-status`).textContent = b.safe ? 'SAFE' : 'UNSAFE';
                    document.getElementById(`barrier-${p}-status`).className = `barrier-indicator ${b.safe ? 'barrier-safe' : 'barrier-unsafe'}`;
                    const margin = document.getElementById(`barrier-${p}-margin`);
                    const pct = Math.max(0, Math.min(100, b.margin * 100));
                    margin.style.width = `${pct}%`;
                    margin.style.background = pct > 20 ? '#4caf50' : '#ff4d4d';
                }
            }
        }
    }

    async injectFault() {
        const assetId = document.getElementById('fi-asset').value;
        const type = document.getElementById('fi-type').value;
        const sev = document.getElementById('fi-severity').value / 100;
        const status = document.getElementById('fi-status');
        status.textContent = "INJECTING...";
        try {
            const resp = await fetch('/api/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'inject_fault', asset_id: assetId, fault_type: type, severity: sev })
            });
            const res = await resp.json();
            status.textContent = res.status === 'FAULT_INJECTED' ? `✓ ${type.toUpperCase()} INJECTED` : `✗ ${res.status}`;
            status.style.color = res.status === 'FAULT_INJECTED' ? "#4caf50" : "#ff4d4d";
        } catch (e) { status.textContent = "✗ NETWORK ERROR"; }
    }

    async clearFaults() {
        const assetId = document.getElementById('fi-asset').value;
        const status = document.getElementById('fi-status');
        try {
            const resp = await fetch('/api/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'clear_faults', asset_id: assetId })
            });
            const res = await resp.json();
            status.textContent = res.status === 'FAULTS_CLEARED' ? "✓ ALL FAULTS CLEARED" : `✗ ${res.status}`;
            status.style.color = res.status === 'FAULTS_CLEARED' ? "#4caf50" : "#ff4d4d";
        } catch (e) { status.textContent = "✗ NETWORK ERROR"; }
    }

    setupResizer() {
        const resizer = document.getElementById('terminal-resizer');
        const output = document.getElementById('terminal-output');
        let isResizing = false;
        resizer.addEventListener('mousedown', (e) => { isResizing = true; document.body.style.cursor = 'row-resize'; e.preventDefault(); });
        document.addEventListener('mousemove', (e) => {
            if (!isResizing) return;
            const h = e.clientY - output.getBoundingClientRect().top;
            if (h > 100 && h < 800) output.style.height = `${h}px`;
        });
        document.addEventListener('mouseup', () => { isResizing = false; document.body.style.cursor = 'default'; });
    }

    setupTerminal() {
        const input = document.getElementById('terminal-input');
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                const cmd = input.value.trim();
                if (cmd) this.sendCommand(cmd);
                input.value = "";
            }
        });
    }

    logTerminal(msg, type = "info") {
        const output = document.getElementById('terminal-output');
        const line = document.createElement('div');
        line.className = `terminal-line line-${type}`;
        line.innerHTML = `<span class="prompt">\u276F</span> ${msg}`;
        output.appendChild(line);
        output.scrollTop = output.scrollHeight;
    }

    async submitPushForReview(updates) {
        try {
            const resp = await fetch('/api/security/reviews', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(updates)
            });
            const res = await resp.json();
            if (res.status === 'SUCCESS') {
                this.logTerminal(`REVIEW TICKET ${res.ticket_id} CREATED. Moderator notification sent.`, "warn");
            }
        } catch (e) { console.error("Review submission failed", e); }
    }

    async refreshReviewQueue() {
        const container = document.getElementById('reviews-list-container');
        if (!container) return;

        // Silent update if already rendered
        if (container.children.length === 0 || container.querySelector('.loading-spinner')) {
            container.innerHTML = '<div class="loading-spinner">SYNCHRONIZING REVIEWS...</div>';
        }
        try {
            const resp = await fetch('/api/security/reviews');
            const data = await resp.json();
            this.renderReviewCenter(container, data.tickets);
        } catch (e) { container.innerHTML = `<div class="err">FAILED TO LOAD REVIEWS: ${e.message}</div>`; }
    }

    renderReviewCenter(container, tickets) {
        // Filter out approved/resolved tickets from active queue
        const activeTickets = (tickets || []).filter(t => t.status !== 'approved');

        if (activeTickets.length === 0) {
            container.innerHTML = '<div class="timeline-empty">NO PENDING REVIEWS IN QUEUE</div>';
            return;
        }

        container.innerHTML = ''; // Clear only on successful fetch
        activeTickets.sort((a, b) => b.timestamp - a.timestamp).forEach(t => {
            const item = document.createElement('div');
            item.className = 'review-ticket';

            const time = new Date(t.timestamp * 1000).toLocaleTimeString();
            const statusColor = t.status === 'bug_raised' ? '#f44336' : t.status === 'approved' ? '#4caf50' : '#ffa726';

            item.innerHTML = `
                <div class="review-ticket-id">#${t.id}</div>
                <div class="review-ticket-time">${time}</div>
                <div class="review-ticket-badge">
                    <span class="tier-badge" style="background: ${statusColor}; color: white; padding: 6px 14px; font-size: 10px; border-radius: 4px; letter-spacing: 1px; border: 1px solid rgba(255,255,255,0.1); text-transform: uppercase;">${t.status}</span>
                </div>
                <div class="review-ticket-requester">
                    <strong>REQUESTER AUTHENTICATION:</strong> ${t.tier.toUpperCase()} CLEARANCE LEVEL
                </div>
            `;

            item.onclick = () => this.openReviewTicket(t);
            container.appendChild(item);
        });
    }

    openReviewTicket(t) {
        const modal = document.getElementById('review-modal');
        document.getElementById('rev-ticket-id').innerText = t.id;
        document.getElementById('rev-requester').innerText = t.tier.toUpperCase();
        document.getElementById('rev-status').innerText = t.status.toUpperCase();

        const diffContainer = document.getElementById('rev-changes-diff');
        diffContainer.innerHTML = '';
        Object.entries(t.changes).forEach(([section, values]) => {
            Object.entries(values).forEach(([key, val]) => {
                const line = document.createElement('div');
                line.innerHTML = `<span style="color:#ffa726">${section}.${key}</span> -> <span style="color:#4caf50">${JSON.stringify(val)}</span>`;
                diffContainer.appendChild(line);
            });
        });

        const feedbackList = document.getElementById('rev-feedback-list');
        feedbackList.innerHTML = t.feedback.length > 0 ? '' : '<div style="opacity:0.5; font-style:italic;">No feedback strings provided yet.</div>';
        t.feedback.forEach(f => {
            const div = document.createElement('div');
            div.style.marginBottom = '8px';
            const color = f.type === 'bug' ? '#f44336' : f.type === 'approval' ? '#4caf50' : '#2196f3';
            div.innerHTML = `<span style="color:${color}; font-weight:bold;">[${f.author.toUpperCase()}]</span>: ${f.text}`;
            feedbackList.appendChild(div);
        });

        // Permissions
        const isModerator = this.securityStatus?.tier === 'moderator' || this.securityStatus?.tier === 'senior' || this.securityStatus?.tier === 'master';
        const isRequester = this.securityStatus?.tier === t.tier;

        document.getElementById('moderator-controls').style.display = isModerator ? 'block' : 'none';
        document.getElementById('requester-controls').style.display = (isRequester && t.status === 'bug_raised') ? 'block' : 'none';

        this.currentActiveTicket = t.id;
        modal.style.display = 'flex';
    }

    async submitReviewFeedback(isBug) {
        const text = document.getElementById('rev-feedback-input').value;
        if (!text) return;
        try {
            const resp = await fetch(`/api/security/reviews/${this.currentActiveTicket}/feedback`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text, is_bug: isBug })
            });
            const res = await resp.json();
            if (res.status === 'SUCCESS') {
                document.getElementById('review-modal').style.display = 'none';
                this.refreshReviewQueue();
            }
        } catch (e) { console.error("Feedback submission failed", e); }
    }

    async resolveReviewBugs() {
        try {
            const resp = await fetch(`/api/security/reviews/${this.currentActiveTicket}/resolve`, { method: 'POST' });
            const res = await resp.json();
            if (res.status === 'SUCCESS') {
                document.getElementById('review-modal').style.display = 'none';
                this.refreshReviewQueue();
            }
        } catch (e) { console.error("Resolve failed", e); }
    }

    async approveReview() {
        try {
            const resp = await fetch(`/api/security/reviews/${this.currentActiveTicket}/approve`, { method: 'POST' });
            const res = await resp.json();
            if (res.status === 'SUCCESS') {
                document.getElementById('review-modal').style.display = 'none';
                this.refreshReviewQueue();
                this.logTerminal(`TICKET ${this.currentActiveTicket} APPROVED. Config promoted to LIVE server.`, "info");
            }
        } catch (e) { console.error("Approval failed", e); }
    }

    async sendCommand(action) {
        this.logTerminal(`USER@AURORA:~$ ${action}`, "cmd");
        try {
            const resp = await fetch('/api/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: action, asset_id: "global" })
            });
            const res = await resp.json();

            if (res.stdout) {
                this.logTerminal(res.stdout);
            }
            if (res.stderr) {
                this.logTerminal(res.stderr, "err");
            }

            if (!res.stdout && !res.stderr) {
                if (res.status === "SHELL_EXECUTION" && res.exit_code === 0) {
                    this.logTerminal("Command completed with no output.");
                } else if (res.status === "ERROR" || res.exit_code !== 0) {
                    this.logTerminal(`Command failed: ${res.message || 'Exit code ' + res.exit_code}`, "err");
                }
            }
        } catch (err) {
            this.logTerminal(`NETWORK ERROR: ${err.message}`, "err");
        }
    }

    async refreshAudit() {
        try {
            const resp = await fetch('/api/security/audit');
            const data = await resp.json();
            const body = document.getElementById('audit-body');
            body.innerHTML = "";
            (data.logs || []).forEach(log => {
                const row = document.createElement('tr');
                row.innerHTML = `<td>${new Date(log.timestamp * 1000).toLocaleString()}</td><td>${log.request_hash.substring(0, 16)}...</td><td>${log.model_name}</td><td>${log.status}</td><td>${log.latency_ms.toFixed(2)}ms</td>`;
                body.appendChild(row);
            });
        } catch (err) { }
    }

    startUptimeTicker() {
        setInterval(() => {
            const u = Math.floor((Date.now() - this.uptimeStart) / 1000);
            const h = Math.floor(u / 3600).toString().padStart(2, '0');
            const m = Math.floor((u % 3600) / 60).toString().padStart(2, '0');
            const s = (u % 60).toString().padStart(2, '0');
            document.getElementById('uptime-value').innerText = `${h}:${m}:${s}`;
        }, 1000);
    }
}

// Global scope helper for quick buttons
window.sendQuickCommand = (cmd) => { window.commander.sendCommand(cmd); };

// Initialize on Load
window.addEventListener('load', () => { window.commander = new AuroraCommander(); });
