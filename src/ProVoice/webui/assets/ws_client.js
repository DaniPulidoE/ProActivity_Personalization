console.log("WS client loaded");

// Helper function to set color based on value
function highlightAlerts(el) {
    if (!el) return;
    const text = el.innerText || '';
    if (text === 'YES' || text === 'Detected') {
        el.style.color = 'red';
    } else {
        el.style.color = '';
    }
}


document.addEventListener('DOMContentLoaded', () => {

    const socket = io('http://localhost:8001', {
        transports: ['websocket'],  // force WebSocket only
        reconnection: true
    });


    socket.on('new_data', function(data) {
        console.log("has bpm_history?", "bpm_history" in data, data.bpm_history);
        console.log("has rr_history?", "rr_history" in data, data.rr_history);

        if(data.frame) document.getElementById('live-image').src = 'data:image/jpeg;base64,' + data.frame;
        const banner = document.getElementById('calibration-banner');
        if (banner) banner.style.display = data.calibrating ? 'block' : 'none';
        document.getElementById('blink-count').innerText = data.blink_count;
        document.getElementById('yawn-count').innerText = data.yawn_count;
        document.getElementById('perclos-score').innerText = data.perclos;
        document.getElementById('drowsiness-status').innerText = data.drowsiness_alert ? 'YES' : 'NO';
        highlightAlerts(document.getElementById('drowsiness-status'));

        document.getElementById('gaze-score').innerText = data.gaze_score;
        document.getElementById('gaze-distracted').innerText = data.gaze_distracted ? 'YES' : 'NO';
        highlightAlerts(document.getElementById('gaze-distracted'));

        document.getElementById('emotion-label').innerText = data.emotion;
        document.getElementById('emotion-prob').innerText = data.emotion_prob;

        document.getElementById('phone-status').innerText = data.lab.includes('phone') ? 'Detected' : 'No';
        highlightAlerts(document.getElementById('phone-status'));
        document.getElementById('smoke-status').innerText = data.lab.includes('smoke') ? 'Detected' : 'No';
        highlightAlerts(document.getElementById('smoke-status'));
        document.getElementById('drink-status').innerText = data.lab.includes('drink') ? 'Detected' : 'No';
        highlightAlerts(document.getElementById('drink-status'));
        let others = data.lab.filter(x => !['phone','smoke','drink'].includes(x));
        document.getElementById('lab-status').innerText = others.length ? others.join(', ') : 'None';

        document.getElementById('eye-ar').innerText = data.eye_ar;
        document.getElementById('mouth-ar').innerText = data.mar;

        // Heart rate trend
        if(data.bpm_history) {
            let fig = {
                data: [{y: data.bpm_history, type:'scatter', mode:'lines+markers'}],
                layout: {margin:{l:20,r:20,t:20,b:20}, yaxis:{title:"BPM"}, xaxis:{title:"Frame"}, height:230}
            };
            Plotly.react('hr-trend', fig.data, fig.layout);
        }

        // Respiratory rate trend
        if(data.rr_history) {
            let fig = {
                data: [{y: data.rr_history, type:'scatter', mode:'lines+markers'}],
                layout: {margin:{l:20,r:20,t:20,b:20}, yaxis:{title:"BPM"}, xaxis:{title:"Frame"}, height:230}
            };
            Plotly.react('rr-trend', fig.data, fig.layout);
        }

        // Action report
        if(data.last_action) {
            let a = data.last_action;
            document.getElementById('action-report').innerText =
                `LoA: ${a.LoA||'--'}, Action: ${a.action||'--'}, Level: ${a.level||'--'}, Message: ${a.message||'--'}`;
        }

        document.getElementById('update-timestamp').innerText = data.timestamp;
    });
    socket.on('connect', () => { console.log('WS connected'); });
    socket.on('disconnect', (reason) => { console.log('WS disconnected', reason); });
    socket.on('reconnect', (attempt) => { console.log('WS reconnected', attempt); });

})