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
    cors_allowed_origins="*"
)

sio_app = socketio.ASGIApp(sio, socketio_path="/socket.io")
app.mount("/socket.io", sio_app)

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
        # 左侧：实时图像
        dbc.Col(
            [
                html.Img(id="live-image",
                         style={"width": "720px", "height": "auto", "border": "1px solid #333", "minHeight": "480px",
                                "display": "block", "margin": "0 auto"}),
                html.Div(id="action-report", className="mt-3",
                         style={"fontSize": "1.15rem", "color": "#246", "minHeight": "50px"}),
            ],
            width=6,
            className="d-flex align-items-center",
            style={"minHeight": "760px"}
        ),
        # 右侧：数据面板
        dbc.Col([
            html.Div([
                html.Div([
                    html.Span("Last Updated: ", className='fw-bold'),
                    html.Span(id='update-timestamp', className='me-3'),
                ], className="mb-2"),

                html.Div("Fatigue Detection", className="fw-bold mb-2"),
                html.Div([
                    "Blink Count: ", html.Span(id='blink-count', className='me-3'),
                    "Yawn Count: ", html.Span(id='yawn-count', className='me-3'),
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
# Async data emitter (20 ms)
# =====================================================
async def emit_data_periodically():
    while True:
        if data_collector is None:
            await asyncio.sleep(0.02)
            continue
        
        
        # 图像流
        frame_b64 = data_collector.get_latest_frame()
        # 数值
        latest_data = data_collector.get_latest_data()

        if latest_data is None:
            await asyncio.sleep(0.02)
            continue

        payload = latest_data.copy()
        payload["frame"] = frame_b64
        payload["calibrating"] = not data_collector.calibrated

        await sio.emit("new_data", payload)
        await asyncio.sleep(0.02)
