//Simple helper to log JSON nicely
function logResponse(obj){
    const box = document.getElementById("responseBox");
    if (!box) return;
    box.textContent = JSON.stringify(obj, null, 2);
}

// Generate or reuse a sessionID for this browser session
let sessionID = localStorage.getItem("bd_session_id");
if (!sessionID){
    sessionID = "sess-" + Math.random().toString(36).slice(2, 10);
    localStorage.setItem("bd_session_id", sessionID)
}

//In-memory buffers
const signalsBuffer = {
    mouseMoves: [],
    clicks: [],
    keys:   [],
};

//Collect mouse moves (throttled)
let lastMoveTime = 0;
document.addEventListener("mousemove", (e) => {
    const now = Date.now();
    if (now - lastMoveTime < 100) return; //record at most every 100 ms
    lastMoveTime = now;

    signalsBuffer.mouseMoves.push({
        x: e.clientX,
        y: e.clientY,
        ts: now,
    });
});

//Collect clicks
document.addEventListener("click", (e) => {
    signalsBuffer.clicks.push({
        x: e.clientX,
        y: e.clientY,
        button: e.button,
        ts: Date.now(),
    });
});

//Collect key presses
document.addEventListener("keydown", (e) =>{
    signalsBuffer.keys.push({
        key: e.key,
        code: e.code,
        ts: Date.now(),
    });
});

 //Send a batch to backend
 async function sendBatch(){
    //if nothing collected, skip
    if(signalsBuffer.mouseMoves.length === 0 && signalsBuffer.clicks.length === 0 &&  signalsBuffer.keys.length === 0){
        return;
    }
    
    //Copy and clear buffer
    const payloadSignals = {
        mouseMoves: signalsBuffer.mouseMoves,
        clicks: signalsBuffer.clicks,
        keys: signalsBuffer.keys,
    };

    signalsBuffer.mouseMoves = [];
    signalsBuffer.clicks = [];
    signalsBuffer.keys = [];

    const payload = {
        sessionID: sessionID,
        signals: payloadSignals,
        metadata: {
            userAgent: navigator.userAgent,
            viewportWidth: window.innerWidth,
            viewportHeight: window.innerHeight,
        },
    };

    try {
        const res = await fetch("/api/signals", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload),
        });

        const data = await res.json();
        logResponse(data);
    } catch(err) { 
        logResponse({ error: err.message });
    }
}

// Manual trigger (button)
async function sendTestBatch(){
    await sendBatch();
}

// Set up UI and interval
document.addEventListener("DOMContentLoaded", () =>{
    const btn = document.getElementById("sendTest");
    if (btn) {
        btn.addEventListener("click", sendTestBatch);
    }

    //autosend every 3 seconds
    setInterval(sendBatch, 3000);
});
    