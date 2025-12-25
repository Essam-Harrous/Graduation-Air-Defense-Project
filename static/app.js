async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: body ? JSON.stringify(body) : "{}"
  });
  return {ok: res.ok, data: await res.json()};
}

function renderLog(items) {
  const el = document.getElementById("log");
  el.innerHTML = items.map(x => `<div class="row"><b>${x.t}</b> — ${x.msg}</div>`).join("");
}

let armed = false;

async function refresh() {
  let res;
  try {
      res = await fetch("/api/status");
  } catch (e) {
      document.getElementById("loading").classList.remove("hidden");
      return;
  }
  const s = await res.json();

  const loadingOverlay = document.getElementById("loading");
  
  // connection check
  if (s.state && s.state.includes("ONLINE")) {
      loadingOverlay.classList.add("hidden");
  } else {
      loadingOverlay.classList.remove("hidden");
  }

  document.getElementById("st").textContent = s.state;
  document.getElementById("rad").textContent = `ANG ${s.radar.angle} | ${s.radar.dist}cm | HIT ${s.radar.hit}`;
  document.getElementById("pt").textContent = `${s.pan} / ${s.tilt}`;
  document.getElementById("det").textContent = `${s.lastDetection.name} (${(s.lastDetection.conf||0).toFixed(2)})`;
  document.getElementById("aa").textContent = `${s.autoAlarmRemaining.toFixed(1)}s`;

  armed = s.manualArmed;
  const armBtn = document.getElementById("arm");
  armBtn.textContent = `ARM MANUAL FIRE: ${armed ? "ON" : "OFF"}`;

  const fireBtn = document.getElementById("fire");
  fireBtn.disabled = !armed;

  // Render Components Status
  if (s.components) {
      const compDiv = document.getElementById("comp-status") || createCompStatus();
      let html = "<table><tr><th>COMP</th><th>PIN</th><th>STATUS</th><th>VAL</th></tr>";
      for (const [key, val] of Object.entries(s.components)) {
          let statusClass = val.status === "ONLINE" ? "c-on" : "c-off";
          html += `<tr>
            <td style="text-transform:uppercase">${key.replace("_", " ")}</td>
            <td>${val.pin}</td>
            <td class="${statusClass}">${val.status}</td>
            <td>${val.val}</td>
          </tr>`;
      }
      html += "</table>";
      compDiv.innerHTML = "<div class='ptitle'>SYSTEM HEALTH</div>" + html;
  }

  renderLog(s.log);
  updateRadarData(s.radar);
}

function createCompStatus() {
    const div = document.createElement("div");
    div.id = "comp-status";
    div.className = "panel";
    div.style.marginTop = "14px";
    // Insert after video Panel
    const videoPanel = document.querySelector(".panel");
    videoPanel.parentElement.insertBefore(div, videoPanel.nextSibling);
    // Actually layout might break, let's just append to side panel or make it floating?
    // User asked for dashboard component status. Let's put it in the "side" column for better layout.
    const side = document.querySelector(".side");
    side.appendChild(div);
    return div;
}

document.getElementById("center").onclick = async () => {
  await postJSON("/api/center");
};

document.getElementById("arm").onclick = async () => {
  armed = !armed;
  await postJSON("/api/arm", {armed});
  await refresh();
};

document.getElementById("fire").onclick = async () => {
  const fireBtn = document.getElementById("fire");
  fireBtn.disabled = true;
  fireBtn.textContent = "FIRING...";
  await postJSON("/api/fire");
  setTimeout(async () => {
    fireBtn.textContent = "FIRE (MANUAL)";
    fireBtn.disabled = !armed;
    await refresh();
  }, 900);
};

// Radar Logic
const radarCanvas = document.getElementById("radar");
const ctx = radarCanvas.getContext("2d");

// State for smooth animation
let radarState = {
    angle: 90,
    dist: 0,
    hit: false,
    timestamp: 0
};
let displayedAngle = 90;

// Blip persistence - store detected objects with timestamps
const blipHistory = [];
const BLIP_LIFETIME_MS = 2000; // Blips fade over 2 seconds

function drawRadarFrame() {
  const w = radarCanvas.width;
  const h = radarCanvas.height;
  const cx = w / 2;
  const cy = h - 2; 
  const radius = w / 2 - 4; 

  // Clear
  ctx.clearRect(0, 0, w, h);

  // --- Grid & Ticks ---
  // (Simplified for performance, but keeping style)
  ctx.strokeStyle = "rgba(85, 221, 136, 0.4)";
  ctx.fillStyle = "rgba(85, 221, 136, 0.8)";
  ctx.lineWidth = 1;

  // Main Arcs
  ctx.beginPath();
  [1.0, 0.75, 0.5, 0.25].forEach(scale => {
      ctx.arc(cx, cy, radius * scale, Math.PI, 0); 
  });
  ctx.stroke();

  // Spokes & Ticks (every 30 deg)
  for (let a = Math.PI; a <= Math.PI * 2; a += (Math.PI / 6)) {
      const x1 = cx + Math.cos(a) * radius;
      const y1 = cy + Math.sin(a) * radius;
      const x2 = cx + Math.cos(a) * (radius - 10);
      const y2 = cy + Math.sin(a) * (radius - 10);
      
      ctx.beginPath();
      ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
      ctx.stroke();
      
      // Spoke (faint)
      if (Math.abs(a - Math.PI) < 0.01 || Math.abs(a - 1.5*Math.PI) < 0.01 || Math.abs(a - 2*Math.PI) < 0.01) {
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(x1, y1);
          ctx.strokeStyle = "rgba(85, 221, 136, 0.2)";
          ctx.stroke();
          ctx.strokeStyle = "rgba(85, 221, 136, 0.4)";
      }
      
      // Text labels
      if (a % (Math.PI/4) === 0) {
        let deg = Math.round((a - 1.5*Math.PI) * (180/Math.PI));
        ctx.font = "10px monospace";
        ctx.fillText(`${deg}°`, x1 * 0.95 + cx*0.05 - 10, y1 * 0.95 + cy*0.05);
      }
  }

  // --- Smooth Sweep Line ---
  // Interpolate displayedAngle towards radarState.angle
  // If difference is large (e.g. wrap around), jump. Else slide.
  let diff = radarState.angle - displayedAngle;
  // If moving, smooth it. 
  // Note: Arduino sends integer angles.
  // We apply simple easing:
  displayedAngle += diff * 0.1; 
  if (Math.abs(diff) < 0.5) displayedAngle = radarState.angle;
  
  // Mapping:
  // let radarRad = Math.PI + ((180 - displayedAngle) / 180) * Math.PI; // 180->Left, 0->Right
  // Wait, check standard servo: usually 0 is Right (2PI) and 180 is Left (PI)?
  // Previous code: radarRad = Math.PI + ((180 - angleDeg) / 180) * Math.PI;
  // Let's stick to that mapping.
  
  let radarRad = Math.PI + ((180 - displayedAngle) / 180) * Math.PI;

  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(radarRad);
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.arc(0, 0, radius, -0.05, 0.05); // Tiny wedge
  ctx.lineTo(0,0);
  ctx.fillStyle = "rgba(0, 255, 0, 0.5)";
  ctx.fill();
  ctx.restore();

  // --- Object Blips with Persistence ---
  const now = Date.now();
  
  // Add new blip to history if object detected
  if (radarState.dist > 0 && radarState.dist <= 50) {
      // Check if we already have a recent blip at similar angle (avoid duplicates)
      const existingBlip = blipHistory.find(b => Math.abs(b.angle - radarState.angle) < 5 && now - b.timestamp < 200);
      if (!existingBlip) {
          blipHistory.push({
              angle: radarState.angle,
              dist: radarState.dist,
              hit: radarState.hit,
              timestamp: now
          });
      }
  }
  
  // Remove old blips
  while (blipHistory.length > 0 && now - blipHistory[0].timestamp > BLIP_LIFETIME_MS) {
      blipHistory.shift();
  }
  
  // Draw all blips with fading
  for (const blip of blipHistory) {
      const age = now - blip.timestamp;
      const alpha = Math.max(0, 1 - (age / BLIP_LIFETIME_MS));
      
      const pixDist = (blip.dist / 50) * radius;
      const blipRad = Math.PI + ((180 - blip.angle) / 180) * Math.PI;
      
      const blipX = cx + Math.cos(blipRad) * pixDist;
      const blipY = cy + Math.sin(blipRad) * pixDist;

      ctx.beginPath();
      ctx.arc(blipX, blipY, 5 + alpha * 2, 0, Math.PI * 2); 
      
      if (blip.hit) {
          ctx.fillStyle = `rgba(255, 50, 50, ${alpha})`;
          ctx.shadowBlur = 15 * alpha;
          ctx.shadowColor = "red";
      } else {
          ctx.fillStyle = `rgba(255, 255, 0, ${alpha * 0.9})`;
          ctx.shadowBlur = 10 * alpha;
          ctx.shadowColor = "yellow";
      }
      
      ctx.fill();
      ctx.shadowBlur = 0;
      
      // Only show distance label for fresh blips
      if (alpha > 0.7) {
          ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
          ctx.font = "10px monospace";
          ctx.fillText(`${blip.dist}cm`, blipX + 10, blipY);
      }
  }
  
  // Digital text
  ctx.fillStyle = "rgba(85, 221, 136, 0.6)";
  ctx.font = "10px monospace";
  ctx.fillText("RADAR LINK: ONLINE", 10, h-5);
  ctx.fillText("RANGE: 50cm", w - 70, h-5);
  
  requestAnimationFrame(drawRadarFrame);
}

// Start animation loop
requestAnimationFrame(drawRadarFrame);

// Update state function (called by refresh)
function updateRadarData(data) {
    if (data.angle !== undefined) {
        radarState.angle = data.angle;
        radarState.dist = data.dist;
        radarState.hit = data.hit;
        radarState.timestamp = Date.now();
    }
}

setInterval(refresh, 50); // ~20fps polling
refresh();
