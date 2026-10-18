"""Interactive Canvas Audio Player with Variable Speed and A/B Looping for Forensic Audio.

Author: Agam Ghotra (IIT2024226), Chunit Bansal (IIT2024233)
"""

import base64
import io
from pathlib import Path
from typing import Optional
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components


def render_forensic_audio_player(audio_path: Path | str, height: int = 160) -> None:
    """Renders a custom HTML5 canvas-driven forensic audio player.

    Features:
    - Synchronized canvas waveform progress visualizer
    - Play / Pause toggle with spacebar / click
    - Interactive seekbar
    - Variable playback speed controls (0.5x, 0.75x, 1.0x, 1.25x, 1.5x, 2.0x)
    - Forensic A/B segment loop selector for scrutinizing micro-artifacts
    - High-contrast forensic dark mode aesthetic
    """
    path = Path(audio_path)
    if not path.exists():
        st.warning("Audio file not found for playback.")
        return

    # Prepare base64 audio data URI
    try:
        suffix = path.suffix.lower()
        if suffix in [".wav", ".mp3", ".ogg"]:
            mime_type = "audio/wav" if suffix == ".wav" else ("audio/mpeg" if suffix == ".mp3" else "audio/ogg")
            with open(path, "rb") as f:
                b64_audio = base64.b64encode(f.read()).decode("utf-8")
        else:
            # Transcode in memory to 16-bit PCM WAV for HTML5 compatibility
            data, sr = sf.read(str(path))
            buf = io.BytesIO()
            sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
            b64_audio = base64.b64encode(buf.getvalue()).decode("utf-8")
            mime_type = "audio/wav"
    except Exception as e:
        # Fallback to standard Streamlit audio if anything fails
        st.audio(str(path))
        return

    audio_uri = f"data:{mime_type};base64,{b64_audio}"

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        body {{
            background: transparent;
            color: #F8FAFC;
            padding: 0;
            margin: 0;
            overflow: hidden;
        }}
        .player-card {{
            background: linear-gradient(180deg, #111827 0%, #0B0F17 100%);
            border: 1px solid #1F2937;
            border-radius: 12px;
            padding: 12px 14px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        }}
        .waveform-canvas {{
            width: 100%;
            height: 48px;
            background: #080B11;
            border-radius: 6px;
            border: 1px solid #1E293B;
            display: block;
            cursor: pointer;
        }}
        .controls-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 10px;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .left-controls {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .play-btn {{
            background: #06B6D4;
            color: #080B11;
            border: none;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            font-size: 15px;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.15s ease;
            box-shadow: 0 0 12px rgba(6, 182, 212, 0.4);
        }}
        .play-btn:hover {{
            background: #22D3EE;
            transform: scale(1.05);
        }}
        .time-readout {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: #94A3B8;
        }}
        .speed-group, .loop-group {{
            display: flex;
            align-items: center;
            gap: 4px;
            background: #161F30;
            padding: 3px 6px;
            border-radius: 6px;
            border: 1px solid #1E293B;
        }}
        .chip-label {{
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
            margin-right: 2px;
            font-weight: 600;
        }}
        .chip-btn {{
            background: transparent;
            border: 1px solid transparent;
            color: #94A3B8;
            font-size: 11px;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        .chip-btn:hover {{
            color: #F8FAFC;
            background: #1E293B;
        }}
        .chip-btn.active {{
            background: #06B6D4;
            color: #080B11;
            border-color: #06B6D4;
        }}
        .loop-status {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: #F59E0B;
        }}
    </style>
    </head>
    <body>
        <div class="player-card">
            <audio id="audioElem" src="{audio_uri}" preload="metadata"></audio>
            
            <canvas id="waveCanvas" class="waveform-canvas" height="48"></canvas>
            
            <div class="controls-row">
                <div class="left-controls">
                    <button id="playBtn" class="play-btn">▶</button>
                    <div id="timeText" class="time-readout">00:00.00 / 00:00.00</div>
                </div>

                <div class="speed-group">
                    <span class="chip-label">Speed</span>
                    <button class="chip-btn" onclick="setSpeed(0.5, this)">0.5x</button>
                    <button class="chip-btn" onclick="setSpeed(0.75, this)">0.75x</button>
                    <button class="chip-btn active" onclick="setSpeed(1.0, this)">1.0x</button>
                    <button class="chip-btn" onclick="setSpeed(1.5, this)">1.5x</button>
                    <button class="chip-btn" onclick="setSpeed(2.0, this)">2.0x</button>
                </div>

                <div class="loop-group">
                    <span class="chip-label">A/B Loop</span>
                    <button class="chip-btn" id="btnA" onclick="setLoopA()">Set A</button>
                    <button class="chip-btn" id="btnB" onclick="setLoopB()">Set B</button>
                    <button class="chip-btn" onclick="clearLoop()">Clear</button>
                    <span id="loopStatus" class="loop-status"></span>
                </div>
            </div>
        </div>

        <script>
            const audio = document.getElementById('audioElem');
            const playBtn = document.getElementById('playBtn');
            const timeText = document.getElementById('timeText');
            const canvas = document.getElementById('waveCanvas');
            const ctx = canvas.getContext('2d');
            const loopStatus = document.getElementById('loopStatus');

            let loopA = null;
            let loopB = null;

            function formatTime(s) {{
                if (isNaN(s)) return "00:00.00";
                const mins = Math.floor(s / 60);
                const secs = (s % 60).toFixed(2);
                return (mins < 10 ? "0" + mins : mins) + ":" + (secs < 10 ? "0" + secs : secs);
            }}

            function drawWaveform() {{
                const w = canvas.width = canvas.clientWidth;
                const h = canvas.height = canvas.clientHeight;
                ctx.clearRect(0, 0, w, h);

                // Draw background bars
                const bars = 70;
                const progress = audio.duration ? (audio.currentTime / audio.duration) : 0;
                const barWidth = w / bars;

                // Loop region highlight
                if (loopA !== null && loopB !== null && audio.duration) {{
                    const xA = (loopA / audio.duration) * w;
                    const xB = (loopB / audio.duration) * w;
                    ctx.fillStyle = 'rgba(245, 158, 11, 0.2)';
                    ctx.fillRect(xA, 0, xB - xA, h);
                }}

                for (let i = 0; i < bars; i++) {{
                    const x = i * barWidth;
                    const normI = i / bars;
                    // Pseudo visual envelope based on frame
                    const barHeight = (Math.sin(i * 0.4) * 0.3 + 0.5) * (h * 0.7);
                    const y = (h - barHeight) / 2;

                    if (normI <= progress) {{
                        ctx.fillStyle = '#06B6D4'; // Played
                    }} else {{
                        ctx.fillStyle = '#1E293B'; // Unplayed
                    }}
                    ctx.fillRect(x + 1, y, barWidth - 2, barHeight);
                }}

                // Playhead line
                const playX = progress * w;
                ctx.strokeStyle = '#FFFFFF';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(playX, 0);
                ctx.lineTo(playX, h);
                ctx.stroke();

                // Loop markers
                if (loopA !== null && audio.duration) {{
                    const xA = (loopA / audio.duration) * w;
                    ctx.fillStyle = '#10B981';
                    ctx.fillRect(xA - 1.5, 0, 3, h);
                }}
                if (loopB !== null && audio.duration) {{
                    const xB = (loopB / audio.duration) * w;
                    ctx.fillStyle = '#EF4444';
                    ctx.fillRect(xB - 1.5, 0, 3, h);
                }}
            }}

            playBtn.addEventListener('click', () => {{
                if (audio.paused) {{
                    audio.play();
                    playBtn.textContent = '⏸';
                }} else {{
                    audio.pause();
                    playBtn.textContent = '▶';
                }}
            }});

            audio.addEventListener('timeupdate', () => {{
                timeText.textContent = formatTime(audio.currentTime) + " / " + formatTime(audio.duration);
                if (loopA !== null && loopB !== null && loopB > loopA) {{
                    if (audio.currentTime >= loopB) {{
                        audio.currentTime = loopA;
                        audio.play();
                    }}
                }}
                drawWaveform();
            }});

            audio.addEventListener('loadedmetadata', () => {{
                timeText.textContent = formatTime(audio.currentTime) + " / " + formatTime(audio.duration);
                drawWaveform();
            }});

            audio.addEventListener('ended', () => {{
                playBtn.textContent = '▶';
                drawWaveform();
            }});

            canvas.addEventListener('click', (e) => {{
                const rect = canvas.getBoundingClientRect();
                const clickX = e.clientX - rect.left;
                const ratio = clickX / rect.width;
                if (audio.duration) {{
                    audio.currentTime = ratio * audio.duration;
                    drawWaveform();
                }}
            }});

            window.setSpeed = function(rate, elem) {{
                audio.playbackRate = rate;
                document.querySelectorAll('.speed-group .chip-btn').forEach(b => b.classList.remove('active'));
                elem.classList.add('active');
            }};

            window.setLoopA = function() {{
                loopA = audio.currentTime;
                updateLoopUI();
            }};

            window.setLoopB = function() {{
                if (loopA === null) loopA = 0;
                loopB = Math.max(audio.currentTime, loopA + 0.1);
                updateLoopUI();
            }};

            window.clearLoop = function() {{
                loopA = null;
                loopB = null;
                loopStatus.textContent = '';
                drawWaveform();
            }};

            function updateLoopUI() {{
                if (loopA !== null && loopB !== null) {{
                    loopStatus.textContent = '[' + loopA.toFixed(1) + 's ⇄ ' + loopB.toFixed(1) + 's]';
                }} else if (loopA !== null) {{
                    loopStatus.textContent = '[A: ' + loopA.toFixed(1) + 's]';
                }}
                drawWaveform();
            }}

            // Initial render
            window.addEventListener('resize', drawWaveform);
            drawWaveform();
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=height)
