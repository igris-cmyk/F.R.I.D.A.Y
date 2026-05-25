import { connect, StringCodec } from "nats.ws";
const { getCurrentWindow } = window.__TAURI__.window;
const { listen } = window.__TAURI__.event;

const sc = StringCodec();
let nc;

const input = document.getElementById('command-input');
const responseArea = document.getElementById('response-area');
const outputLog = document.getElementById('output-log');
const container = document.querySelector('.palette-container');
const statusIndicator = document.getElementById('nats-status');

// Ephemeral Working Context
let currentAmbientContext = null;
let contextTTL = null;
const CONTEXT_EXPIRATION_MS = 15000; // 15 seconds of validity

// UI Element for Explicit Context Visibility
const contextIndicator = document.createElement('div');
contextIndicator.className = 'context-indicator';
contextIndicator.style.fontSize = '10px';
contextIndicator.style.color = 'rgba(255, 255, 255, 0.4)';
contextIndicator.style.padding = '4px 20px';
contextIndicator.style.display = 'none';
document.querySelector('.input-wrapper').after(contextIndicator);

async function init() {
    try {
        // Connect to NATS over WebSocket (Port 9222 as configured)
        nc = await connect({ servers: "ws://localhost:9222" });
        
        statusIndicator.textContent = "NATS Connected";
        statusIndicator.className = "status-indicator online";
        console.log(`Connected to NATS ${nc.getServer()}`);
        // Subscribe to Core Health Telemetry
        const healthSub = nc.subscribe("friday.system.health");
        (async () => {
            for await (const msg of healthSub) {
                try {
                    const eventJson = sc.decode(msg.data);
                    const event = JSON.parse(eventJson);
                    const p = event.payload;
                    if (p.status === "healthy") {
                        statusIndicator.textContent = "Core Healthy";
                        statusIndicator.className = "status-indicator online";
                    } else if (p.status === "degraded") {
                        statusIndicator.textContent = "Core Degraded";
                        statusIndicator.className = "status-indicator offline";
                    } else if (p.status === "recovering") {
                        statusIndicator.textContent = "Core Recovering";
                        statusIndicator.className = "status-indicator offline";
                    }
                } catch (e) {
                    console.error("Health parse error", e);
                }
            }
        })();
        
        // Auto-focus the input whenever the Tauri window is summoned
        try {
            // Listen for global shortcut focus event
            getCurrentWindow().listen("tauri://focus", () => {
                input.focus();
            });
            
            // Listen for Ambient OS context from Rust
            listen("ambient-context", (event) => {
                currentAmbientContext = {
                    active_app: event.payload.active_app || "unknown",
                    window_title: event.payload.window_title || "unknown"
                };
                console.log("Ingested Ambient Context:", currentAmbientContext);
                
                // Explicit UI Visibility
                contextIndicator.textContent = `[Context Visible] ${currentAmbientContext.active_app}: ${currentAmbientContext.window_title}`;
                contextIndicator.style.display = 'block';
                
                // Ephemeral Safeguard: TTL Expiration
                if (contextTTL) clearTimeout(contextTTL);
                contextTTL = setTimeout(() => {
                    currentAmbientContext = null;
                    contextIndicator.style.display = 'none';
                    console.log("Ambient Context Expired (TTL)");
                }, CONTEXT_EXPIRATION_MS);
            });
        } catch (e) {
            console.warn("Tauri API unavailable (running in standard browser?)", e);
        }
        
        // Setup MVP Event Loop
        input.addEventListener('keydown', async (e) => {
            if (e.key === 'Enter' && input.value.trim() !== '') {
                const command = input.value.trim();
                input.value = '';
                
                const traceId = crypto.randomUUID();
                const streamSubject = `friday.stream.${traceId}`;
                
                // Reset UI for new stream
                outputLog.textContent = '';
                appendLine(`[TRACE: ${traceId.split('-')[0]}] Initializing cognitive pipeline...`);
                outputLog.className = "response-content";
                container.classList.add('has-response');
                responseArea.classList.add('active');
                
                try {
                    // Subscribe to the trace-bound stream first
                    const sub = nc.subscribe(streamSubject);
                    
                    // Stream Lifecycle Protection
                    let streamTTL = null;
                    const WATCHDOG_MS = 15000;
                    
                    const resetWatchdog = () => {
                        if (streamTTL) clearTimeout(streamTTL);
                        streamTTL = setTimeout(() => {
                            console.warn(`[WATCHDOG] Stream ${traceId} stalled. Forcing cleanup.`);
                            sub.unsubscribe();
                            renderError("Stream Connection Lost (Stall Detected)");
                        }, WATCHDOG_MS);
                    };
                    
                    // Process incoming stream events asynchronously
                    (async () => {
                        resetWatchdog();
                        for await (const msg of sub) {
                            try {
                                resetWatchdog();
                                const eventJson = sc.decode(msg.data);
                                const event = JSON.parse(eventJson);
                                const p = event.payload;
                                
                                if (p.stage !== undefined) {
                                    // ExecutionUpdateEvent
                                    appendTokenizedLine(p.message || "");
                                } else if (p.intent_type !== undefined) {
                                    // TaskAcknowledgedEvent
                                    appendLine(`[ACK] ${p.message}`);
                                } else if (p.status !== undefined) {
                                    // ExecutionResultEvent (Final)
                                    if (streamTTL) clearTimeout(streamTTL);
                                    renderExecutionResult(event);
                                    sub.unsubscribe(); // Close the stream
                                }
                            } catch (parseErr) {
                                console.error("Stream parse error", parseErr);
                            }
                        }
                    })();

                    // Subscribe to the trace-bound permission request stream
                    const permSub = nc.subscribe(`friday.permission.request.${traceId}`);
                    (async () => {
                        for await (const msg of permSub) {
                            try {
                                const eventJson = sc.decode(msg.data);
                                const event = JSON.parse(eventJson);
                                const p = event.payload;
                                
                                renderApprovalCard(p);
                            } catch (e) {
                                console.error("Permission request parse error", e);
                            }
                        }
                    })();

                    // Create strong typed CommandIntentEvent
                    const intentPayload = {
                        metadata: {
                            trace_id: traceId,
                            timestamp: new Date().toISOString(),
                            source_component: "ui.command_palette",
                            priority: "normal"
                        },
                        payload: {
                            raw_command: command,
                            environment: currentAmbientContext ? {
                                active_app: currentAmbientContext.active_app,
                                window_title: currentAmbientContext.window_title,
                                selected_text: null, // Future
                                ingestion_timestamp: new Date().toISOString()
                            } : null,
                            working_directory: null
                        }
                    };
                    
                    // Consume Context immediately on use to preserve ephemerality
                    if (contextTTL) clearTimeout(contextTTL);
                    currentAmbientContext = null;
                    contextIndicator.style.display = 'none';

                    // Publish (Fire-and-forget, responses return on stream)
                    nc.publish("friday.intent.command", sc.encode(JSON.stringify(intentPayload)));
                    
                } catch (err) {
                    renderError(`Stream Error: ${err.message}`);
                }
            }
        });

    } catch (err) {
        console.error("NATS connection error:", err);
        statusIndicator.textContent = "NATS Offline";
        statusIndicator.className = "status-indicator offline";
        renderError("System Offline: Unable to reach Event Bus.");
    }
}

function showResponse(text) {
    outputLog.textContent = text;
    outputLog.className = "response-content";
    container.classList.add('has-response');
    responseArea.classList.add('active');
}

function appendLine(text) {
    outputLog.append(document.createTextNode(`${text}\n`));
}

function appendTokenizedLine(text) {
    const line = document.createElement('span');
    const tokenPattern = /(\[(?:PLANNER|CAPABILITY|COMPLETED|SECURITY|BLOCKED|FAILED)\])/g;
    const parts = text.split(tokenPattern);

    for (const part of parts) {
        if (!part) continue;

        const token = document.createElement('span');
        if (part === '[PLANNER]') {
            token.style.color = '#cda869';
            token.style.fontWeight = 'bold';
            token.textContent = part;
            line.append(token);
        } else if (part === '[CAPABILITY]' || part === '[COMPLETED]') {
            token.style.color = '#8f9d6a';
            token.style.fontWeight = 'bold';
            token.textContent = part;
            line.append(token);
        } else if (part === '[SECURITY]' || part === '[BLOCKED]' || part === '[FAILED]') {
            token.style.color = '#cf6a4c';
            token.style.fontWeight = 'bold';
            token.textContent = part;
            line.append(token);
        } else {
            line.append(document.createTextNode(part));
        }
    }

    outputLog.append(line, document.createTextNode('\n'));
}

function appendResultBlock(header, content, errorMsg) {
    outputLog.append(document.createTextNode('\n'));
    appendLine(header);
    appendLine('--------------------------------------------------');
    appendLine(content || 'No output returned.');
    if (errorMsg) {
        appendLine(`Error Details: ${errorMsg}`);
    }
}

function renderError(message) {
    outputLog.textContent = `[FAILURE] ${message}`;
    outputLog.className = "response-content error";
    container.classList.add('has-response');
    responseArea.classList.add('active');
}

function renderExecutionResult(event) {
    const traceId = event.metadata.trace_id.split('-')[0]; // short trace
    const ms = event.payload.execution_time_ms;
    const status = event.payload.status; // 'success' or 'failure'
    const output = event.payload.output;
    const errorMsg = event.payload.error;
    
    const header = `[TRACE: ${traceId}] [STATUS: ${status.toUpperCase()}] [TIME: ${ms}ms]`;
    appendResultBlock(header, output, status !== "success" ? errorMsg : null);
    
    if (status === "success") {
        outputLog.className = "response-content success";
    } else {
        outputLog.className = "response-content error";
    }
    
    container.classList.add('has-response');
    responseArea.classList.add('active');
}

function createApprovalDetail(label, value) {
    const row = document.createElement('div');
    row.style.fontSize = '12px';
    row.style.marginBottom = '5px';

    const strong = document.createElement('strong');
    strong.textContent = `${label}: `;
    row.append(strong, document.createTextNode(value));
    return row;
}

function renderApprovalCard(p) {
    const cardId = `approval-card-${p.capability_id.replace(/\./g, '-')}`;
    const cardContainer = document.createElement('div');
    cardContainer.id = cardId;
    cardContainer.className = 'approval-card';
    cardContainer.style.border = '1px solid #cf6a4c';
    cardContainer.style.padding = '10px';
    cardContainer.style.marginTop = '10px';
    cardContainer.style.marginBottom = '10px';
    cardContainer.style.borderRadius = '5px';
    cardContainer.style.background = 'rgba(207, 106, 76, 0.1)';

    const title = document.createElement('div');
    title.className = 'card-title';
    title.style.color = '#cf6a4c';
    title.style.fontWeight = 'bold';
    title.style.marginBottom = '5px';
    title.textContent = `[APPROVAL REQUIRED] ${p.human_name}`;
    cardContainer.append(title);

    cardContainer.append(
        createApprovalDetail('Risk Level', p.risk_level),
        createApprovalDetail('Reason', p.reason),
        createApprovalDetail('Action', p.requested_action_summary)
    );

    const inputRow = document.createElement('div');
    inputRow.style.fontSize = '12px';
    inputRow.style.marginBottom = '10px';
    const inputLabel = document.createElement('strong');
    inputLabel.textContent = 'Input Preview:';
    const inputPreview = document.createElement('pre');
    inputPreview.style.margin = '0';
    inputPreview.style.background = '#222';
    inputPreview.style.padding = '5px';
    inputPreview.style.borderRadius = '3px';
    inputPreview.style.maxHeight = '100px';
    inputPreview.style.overflowY = 'auto';
    inputPreview.textContent = p.input_preview;
    inputRow.append(inputLabel, document.createTextNode(' '), inputPreview);
    cardContainer.append(inputRow);

    const timerDiv = document.createElement('div');
    timerDiv.id = `${cardId}-timer`;
    timerDiv.style.fontSize = '12px';
    timerDiv.style.marginBottom = '10px';
    timerDiv.style.color = '#cda869';
    timerDiv.textContent = `Timeout in ${p.timeout_seconds}s`;
    cardContainer.append(timerDiv);

    const buttonRow = document.createElement('div');
    buttonRow.style.display = 'flex';
    buttonRow.style.gap = '10px';

    const approveBtn = document.createElement('button');
    approveBtn.id = `${cardId}-approve`;
    approveBtn.style.background = '#8f9d6a';
    approveBtn.style.color = '#fff';
    approveBtn.style.border = 'none';
    approveBtn.style.padding = '5px 15px';
    approveBtn.style.borderRadius = '3px';
    approveBtn.style.cursor = 'pointer';
    approveBtn.textContent = 'Approve';

    const denyBtn = document.createElement('button');
    denyBtn.id = `${cardId}-deny`;
    denyBtn.style.background = '#cf6a4c';
    denyBtn.style.color = '#fff';
    denyBtn.style.border = 'none';
    denyBtn.style.padding = '5px 15px';
    denyBtn.style.borderRadius = '3px';
    denyBtn.style.cursor = 'pointer';
    denyBtn.textContent = 'Deny';

    buttonRow.append(approveBtn, denyBtn);
    cardContainer.append(buttonRow);
    outputLog.append(cardContainer, document.createTextNode('\n'));
    
    let timeLeft = p.timeout_seconds;
    
    const sendResponse = (approved) => {
        clearInterval(timerInterval);
        approveBtn.disabled = true;
        denyBtn.disabled = true;
        
        const decision = approved ? "approved" : "denied";
        if (approved) {
            cardContainer.style.borderColor = "#8f9d6a";
            cardContainer.style.background = "rgba(143, 157, 106, 0.1)";
            title.textContent = `[APPROVED] ${p.human_name}`;
            title.style.color = "#8f9d6a";
        } else {
            title.textContent = `[DENIED] ${p.human_name}`;
        }
        
        const responsePayload = {
            metadata: {
                trace_id: p.trace_id,
                timestamp: new Date().toISOString(),
                source_component: "ui.command_palette"
            },
            payload: {
                trace_id: p.trace_id,
                capability_id: p.capability_id,
                approved: approved,
                user_decision: decision,
                response_timestamp: new Date().toISOString(),
                source_component: "desktop_ui"
            }
        };
        
        nc.publish(`friday.permission.response.${p.trace_id}`, sc.encode(JSON.stringify(responsePayload)));
    };
    
    approveBtn.addEventListener('click', () => sendResponse(true));
    denyBtn.addEventListener('click', () => sendResponse(false));
    
    const timerInterval = setInterval(() => {
        timeLeft--;
        if (timeLeft <= 0) {
            clearInterval(timerInterval);
            approveBtn.disabled = true;
            denyBtn.disabled = true;
            timerDiv.textContent = "Timed Out";
            title.textContent = `[TIMED OUT] ${p.human_name}`;
        } else {
            timerDiv.textContent = `Timeout in ${timeLeft}s`;
        }
    }, 1000);
}

// Initialize on load
init();
