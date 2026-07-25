import asyncio
from contextlib import asynccontextmanager

import socketio
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

from ProVoice.data_collector import DataCollector


@asynccontextmanager
async def lifespan(_: FastAPI):
    asyncio.create_task(emit_data_periodically())
    print("Emitter task started")
    yield

app = FastAPI(lifespan=lifespan)

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=["http://localhost:8001", "http://127.0.0.1:8001"]
)

sio_app = socketio.ASGIApp(sio, socketio_path="/socket.io")
app.mount("/socket.io", sio_app)

# Connected dashboard clients; the emitter idles when this is empty.
_connected_sids: set[str] = set()


@sio.event
async def connect(sid, environ):
    _connected_sids.add(sid)


@sio.event
async def disconnect(sid):
    _connected_sids.discard(sid)

external_scripts = [
    "https://cdn.socket.io/4.7.2/socket.io.min.js",
    "https://cdn.plot.ly/plotly-3.3.1.min.js",
]

dash_app = Dash(
    __name__,
    url_base_pathname="/",
    # requests_pathname_prefix="/dash/",
    # routes_pathname_prefix="/dash/",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    external_scripts=external_scripts,
)

data_collector: DataCollector | None = None

dash_app.layout = dbc.Container([
    html.H2("ProVoice Driver State Dashboard with WebSocket", className="mt-4 mb-4 text-center"),
    dbc.Alert(
        "Calibrating sensors... please look at the camera.",
        id="calibration-banner",
        color="warning",
        className="text-center",
        style={"display": "none"},   # hidden by default
    ),
    dbc.Row([
        # Left side: live image
        dbc.Col(
            [
                html.Img(id="live-image",
                         style={"width": "100%", "height": "auto", "border": "1px solid #333",
                                "minHeight": "480px", "display": "block"}),
                html.Div(id="action-report", className="mt-2",
                         style={"fontSize": "1.1rem", "color": "#246"}),
            ],
            width=6,
        ),
        # Right side: data panel
        dbc.Col([
            html.Div([
                html.Div([
                    html.Span("Last Updated: ", className='fw-bold'),
                    html.Span(id='update-timestamp', className='me-3'),
                ], className="mb-2"),

                html.Div("Fatigue Detection", className="fw-bold mb-2"),
                html.Div([
                    "Blink Rate: ", html.Span(id='blink-rate', className='me-3'),
                    "Yawn Rate: ", html.Span(id='yawn-rate', className='me-3'),
                    "PERCLOS: ", html.Span(id='perclos-score', className='me-3'),
                    "Fatigue: ", html.Span(id='drowsiness-status', className='me-3'),
                ], className="mb-3"),
                html.Hr(),

                html.Div("Gaze Detection", className="fw-bold mb-2"),
                html.Div([
                    "Gaze Score: ", html.Span(id='gaze-score', className='me-3'),
                    "Looking Away: ", html.Span(id='gaze-distracted', className='me-3'),
                ], className="mb-3"),
                html.Hr(),

                html.Div("Emotion Detection", className="fw-bold mb-2"),
                html.Div([
                    "Emotion: ", html.Span(id='emotion-label', className='me-3'),
                    "Confidence: ", html.Span(id='emotion-prob', className='me-3'),
                ], className="mb-3"),
                html.Hr(),

                html.Div("Distraction Detection", className="fw-bold mb-2"),
                html.Div([
                    html.Span("Phone: ", className='fw-bold'),
                    html.Span(id='phone-status', className='me-3'),
                    html.Span("Smoking: ", className='fw-bold'),
                    html.Span(id='smoke-status', className='me-3'),
                    html.Span("Drinking: ", className='fw-bold'),
                    html.Span(id='drink-status', className='me-3'),
                ], className="mb-2"),
                html.Div([
                    html.Span("Detected: ", className='fw-bold'),
                    html.Span(id='lab-status', className='me-3'),
                ], className="mb-3"),
                html.Hr(),

                html.Div("Heart Rate Trend", className="fw-bold mb-2", style={"fontSize": "1.1rem"}),
                dcc.Graph(id='hr-trend', style={'height': '250px'}),
                html.Hr(),

                html.Div("Respiratory Rate Trend", className="fw-bold mb-2", style={"fontSize": "1.1rem"}),
                dcc.Graph(id='rr-trend', style={'height': '250px'}),
                html.Hr(),

                html.Div([
                    html.Div("EYE Aspect Ratio:", className='fw-bold'),
                    html.Div(id='eye-ar', className='mb-2'),
                    html.Div("MOUTH Aspect Ratio:", className='fw-bold'),
                    html.Div(id='mouth-ar', className='mb-2'),
                ]),
            ], className='p-2')
        ], width=6)
    ]),
], fluid=True)

app.mount("/", WSGIMiddleware(dash_app.server))

# =====================================================
# Async data emitter (20 Hz)
# =====================================================
async def emit_data_periodically():
    last_emitted_ts = None
    while True:
        try:
            if data_collector is None or not _connected_sids:
                # Nobody listening (or nothing to read): skip the JPEG encode
                # entirely, don't just skip the emit.
                await asyncio.sleep(0.05)
                continue

            # Numeric values (cheap) first — bail before the frame encode
            # (expensive) if this collector tick was already emitted.
            latest_data = data_collector.get_latest_data()

            if latest_data is None:
                await asyncio.sleep(0.05)
                continue

            ts = latest_data.get("timestamp")
            if ts is not None and ts == last_emitted_ts:
                # Collector runs ~4 Hz, this loop 20 Hz: same frame, don't
                # re-encode/re-send it.
                await asyncio.sleep(0.05)
                continue

            payload = latest_data.copy()
            payload["frame"] = data_collector.get_latest_frame()
            payload["calibrating"] = not data_collector.calibrated

            await sio.emit("new_data", payload)
            last_emitted_ts = ts

        except Exception as e:
            print(f"[emitter] error: {e}")

        await asyncio.sleep(0.05) # 0.05 to match 20Hz of data collection