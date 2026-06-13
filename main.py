#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EEG 生体認証 リアルタイム・ダッシュボード (main.py)

ESP32 から送られてくる EEG の TCP バイナリストリームを受信し、`EEGBiometricPipeline`
に流し込みながら、8ch の生波形と認証結果をリアルタイムに可視化するデスクトップ
アプリケーションです。

スレッド構成
------------
* TCP 受信      : ``threading.Thread``（``EEGTCPServer``）— GUI をブロックしない。
* 推論          : ``QThread``（``InferenceWorker``）— 1 秒ごとにバッファから窓を切り出し推論。
* 描画/GUI      : メインスレッド（``EEGDashboard`` + ``QTimer``）。
共有データはスレッドセーフな ``RingBuffer``（``threading.Lock``）経由でやり取りします。

パケット仕様
------------
36 バイト・リトルエンディアン = ``uint32`` タイムスタンプ + ``float32`` × 8ch（``"<I8f"``）。

備考
----
* モデルは未学習想定のため、推論が失敗した場合は **ダミー判定** にフォールバックします。
* ESP32 が無くても動作確認できるよう、内蔵 **シグナルシミュレータ** が
  ``127.0.0.1:8888`` へ実際に TCP 送信します（受信経路をそのまま検証できます）。

必要パッケージ: ``pyqtgraph`` と ``PyQt5``（または ``PySide6``）、``numpy``。
    pip install pyqtgraph PyQt5 numpy
"""
from __future__ import annotations

import os
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# 同階層の eeg_biometric パッケージを解決できるようにする。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
except Exception as exc:  # pragma: no cover
    sys.stderr.write(
        "pyqtgraph と Qt バインディングが必要です。\n"
        "  pip install pyqtgraph PyQt5 numpy\n"
        f"詳細: {exc}\n"
    )
    raise SystemExit(1)


# =========================================================================== #
# 定数
# =========================================================================== #
HOST = "0.0.0.0"
PORT = 8888
N_CHANNELS = 8
# 8ch の名称。Liveness 検知が前頭(Fp1/Fp2)を参照するため先頭に配置する。
CHANNELS: List[str] = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "O1", "O2"]
FS = 250.0                       # サンプリング周波数 [Hz]
BUFFER_SECONDS = 4.0             # リングバッファ長
PLOT_SECONDS = 4.0               # 表示窓
WINDOW_SECONDS = 3.0             # 推論窓
INFER_INTERVAL_MS = 1000         # 推論間隔
PACKET_FORMAT = "<I8f"           # uint32 + float32 x8
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)  # = 36
SUBJECT_ID = "S001"

PALETTE = ["#e6194B", "#3cb44b", "#4363d8", "#f58231",
           "#911eb4", "#22d3ee", "#f032e6", "#bfef45"]

# Qt5/Qt6 双方で動くよう列挙体を解決する。
try:
    _ALIGN_CENTER = QtCore.Qt.AlignCenter
except AttributeError:  # PyQt6
    _ALIGN_CENTER = QtCore.Qt.AlignmentFlag.AlignCenter
try:
    _HLINE = QtWidgets.QFrame.HLine
except AttributeError:  # PyQt6
    _HLINE = QtWidgets.QFrame.Shape.HLine


# =========================================================================== #
# スレッドセーフ・リングバッファ
# =========================================================================== #
class RingBuffer:
    """固定長の循環バッファ（``threading.Lock`` で保護）。

    形状 ``(n_channels, capacity)`` を事前確保し、最新サンプルを末尾とする
    時系列窓を ``get_latest`` で取り出す。
    """

    def __init__(self, n_channels: int, capacity: int) -> None:
        self.n_channels = int(n_channels)
        self.capacity = int(capacity)
        self._buf = np.zeros((self.n_channels, self.capacity), dtype=np.float32)
        self._idx = 0           # 次に書き込む位置
        self._count = 0         # 充填済みサンプル数
        self._total = 0         # 累積受信サンプル数（レート計測用）
        self._lock = threading.Lock()

    def append(self, sample: np.ndarray) -> None:
        """1 サンプル ``(n_channels,)`` を追加する。"""
        with self._lock:
            self._buf[:, self._idx] = sample
            self._idx = (self._idx + 1) % self.capacity
            self._count = min(self._count + 1, self.capacity)
            self._total += 1

    def get_latest(self, n: int) -> Optional[np.ndarray]:
        """最新 ``n`` サンプルを古い→新しい順で ``(n_channels, n)`` で返す。"""
        with self._lock:
            n = min(n, self._count)
            if n <= 0:
                return None
            out = np.empty((self.n_channels, n), dtype=np.float32)
            start = (self._idx - n) % self.capacity
            if start + n <= self.capacity:
                out[:] = self._buf[:, start:start + n]
            else:
                k = self.capacity - start
                out[:, :k] = self._buf[:, start:]
                out[:, k:] = self._buf[:, : n - k]
            return out

    @property
    def total(self) -> int:
        with self._lock:
            return self._total

    @property
    def filled(self) -> int:
        with self._lock:
            return self._count


# =========================================================================== #
# TCP 受信サーバ（バックグラウンドスレッド）
# =========================================================================== #
class EEGTCPServer(threading.Thread):
    """``0.0.0.0:8888`` で待ち受け、36B パケットを解析してバッファへ書き込む。

    部分受信のバッファリング、クライアント切断・再接続、応答性のある停止
    （``settimeout`` ベース）に対応する。GUI とは ``RingBuffer`` と
    スレッドセーフな状態辞書のみを共有する。
    """

    def __init__(self, buffer: RingBuffer, host: str = HOST, port: int = PORT) -> None:
        super().__init__(daemon=True, name="EEGTCPServer")
        self.buffer = buffer
        self.host = host
        self.port = port
        self._running = threading.Event()
        self._running.set()
        self._lock = threading.Lock()
        self._status: Dict[str, object] = {
            "state": "stopped", "client": None, "packets": 0, "error": None,
        }
        self._server_sock: Optional[socket.socket] = None

    # ---- status -----------------------------------------------------------
    def _set_status(self, **kwargs) -> None:
        with self._lock:
            self._status.update(kwargs)

    def status(self) -> Dict[str, object]:
        with self._lock:
            return dict(self._status)

    # ---- lifecycle --------------------------------------------------------
    def stop(self) -> None:
        self._running.clear()
        try:
            if self._server_sock is not None:
                self._server_sock.close()
        except OSError:
            pass

    def run(self) -> None:
        while self._running.is_set():
            try:
                srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind((self.host, self.port))
                srv.listen(1)
                srv.settimeout(1.0)
                self._server_sock = srv
                self._set_status(state="listening", client=None, error=None)
            except Exception as exc:  # bind 失敗等
                self._set_status(state="error", error=str(exc))
                time.sleep(1.5)
                continue

            try:
                while self._running.is_set():
                    try:
                        conn, addr = srv.accept()
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    self._handle_client(conn, addr)
            finally:
                try:
                    srv.close()
                except OSError:
                    pass
        self._set_status(state="stopped", client=None)

    # ---- client handling --------------------------------------------------
    def _handle_client(self, conn: socket.socket, addr) -> None:
        conn.settimeout(1.0)
        self._set_status(state="connected", client=f"{addr[0]}:{addr[1]}")
        pending = b""
        packets = 0
        with conn:
            while self._running.is_set():
                try:
                    chunk = conn.recv(8192)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:          # 相手が切断
                    break
                pending += chunk
                while len(pending) >= PACKET_SIZE:
                    raw = pending[:PACKET_SIZE]
                    pending = pending[PACKET_SIZE:]
                    try:
                        values = struct.unpack(PACKET_FORMAT, raw)
                    except struct.error:
                        continue
                    sample = np.asarray(values[1:1 + N_CHANNELS], dtype=np.float32)
                    self.buffer.append(sample)
                    packets += 1
                    if packets % 50 == 0:
                        self._set_status(packets=packets)
        self._set_status(state="listening", client=None, packets=packets)


# =========================================================================== #
# 内蔵シグナルシミュレータ（ESP32 が無いときのテスト用）
# =========================================================================== #
class SignalSimulator(threading.Thread):
    """合成 EEG を ``127.0.0.1:PORT`` へ 250Hz で TCP 送信する。

    実際の受信・解析経路を通してダッシュボードを検証できる。前頭(Fp1/Fp2)に
    数秒おきに瞬目(EOG)を重畳するので、Liveness の合否変化も観察できる。
    """

    def __init__(self, port: int = PORT, fs: float = FS, seed: int = 1) -> None:
        super().__init__(daemon=True, name="SignalSimulator")
        self.port = port
        self.fs = float(fs)
        self._running = threading.Event()
        self._running.set()
        self._rng = np.random.default_rng(seed)

    def stop(self) -> None:
        self._running.clear()

    def run(self) -> None:
        sock: Optional[socket.socket] = None
        n = 0
        phase = self._rng.uniform(0, 2 * np.pi, size=N_CHANNELS)
        alpha_f = self._rng.uniform(9.5, 11.5, size=N_CHANNELS)
        gains = self._rng.uniform(0.7, 1.3, size=N_CHANNELS)
        next_blink = time.time() + self._rng.uniform(2.0, 5.0)
        blink_until = 0.0
        blink_t0 = 0.0
        dt = 1.0 / self.fs
        while self._running.is_set():
            if sock is None:
                try:
                    sock = socket.create_connection(("127.0.0.1", self.port), timeout=2.0)
                except OSError:
                    time.sleep(0.5)
                    continue
            now = time.time()
            if now >= next_blink:           # 瞬目を発火
                blink_t0 = now
                blink_until = now + 0.25
                next_blink = now + self._rng.uniform(3.0, 6.0)
            t = n / self.fs
            sample = np.empty(N_CHANNELS, dtype=np.float32)
            for c in range(N_CHANNELS):
                v = gains[c] * 8.0 * np.sin(2 * np.pi * alpha_f[c] * t + phase[c])
                v += self._rng.standard_normal() * 4.0           # 背景ノイズ
                sample[c] = v
            if now < blink_until and self.fs > 0:                # 瞬目を前頭に重畳
                env = np.exp(-0.5 * ((now - (blink_t0 + 0.1)) / 0.06) ** 2)
                sample[0] += 120.0 * env
                sample[1] += 120.0 * env
            try:
                sock.sendall(struct.pack(PACKET_FORMAT, n & 0xFFFFFFFF, *sample.tolist()))
            except OSError:
                try:
                    sock.close()
                except OSError:
                    pass
                sock = None
                continue
            n += 1
            time.sleep(dt)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


# =========================================================================== #
# 推論エンジン（EEGBiometricPipeline 統合 + ダミーフォールバック）
# =========================================================================== #
@dataclass
class InferenceResult:
    """GUI へ渡す推論結果。"""
    source: str = "waiting"          # "model" | "dummy" | "waiting"
    decision: bool = False
    label: str = "—"
    score: Optional[float] = None
    ocsvm: Optional[float] = None
    lgbm: Optional[float] = None
    live_pass: Optional[bool] = None
    blinks: int = 0
    threshold: float = 0.5
    reason: str = ""
    note: str = ""

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class InferenceEngine:
    """`EEGBiometricPipeline` をラップし、窓データから認証結果を返す。

    起動時に 8ch 合成データで自動登録(enroll)し、以降は受信窓に対して
    Liveness と生体スコアを算出する。パイプラインが利用不可、または推論が
    例外を投げた場合は、信号ヒューリスティックに基づくダミー判定を返す。
    """

    def __init__(self, channels: List[str], sfreq: float, subject_id: str = SUBJECT_ID) -> None:
        self.channels = list(channels)
        self.sfreq = float(sfreq)
        self.subject_id = subject_id
        self.available = False
        self.info = "未初期化"
        self._pipeline = None
        self._Challenge = None

    # ---- setup（ワーカスレッドから一度だけ呼ぶ）---------------------------
    def setup(self) -> str:
        """パイプラインを構築し合成データで登録する。失敗時はダミーモード。"""
        try:
            from eeg_biometric.pipeline import EEGBiometricPipeline, PipelineConfig
            from eeg_biometric.data import EEGDataSource
            from eeg_biometric.liveness import Challenge

            cfg = PipelineConfig(
                data_source="synthetic", sfreq=self.sfreq, trial_seconds=WINDOW_SECONDS,
                n_bootstrap=6, max_channels=6, encoder_prefer="auto",
            )
            pipe = EEGBiometricPipeline(cfg)
            # ストリーミングでは「プロンプト前は瞬目なし」制約は無意味なので緩和する。
            pipe.liveness.require_clean_pre_prompt = False

            src = EEGDataSource(source="synthetic", sfreq=self.sfreq,
                                trial_seconds=WINDOW_SECONDS, montage=self.channels, seed=0)
            genuine = src.get_subject_trials(self.subject_id, n_trials=16, base_seed=1)
            background = []
            for k, bid in enumerate(["B01", "B02", "B03"]):
                background += src.get_subject_trials(bid, n_trials=6, base_seed=100 + 10 * k)
            calib_imp = src.get_subject_trials("B02", n_trials=6, base_seed=300)
            pipe.enroll(self.subject_id, genuine[:12], background,
                        calib_genuine=genuine[12:], calib_impostor=calib_imp)

            self._pipeline = pipe
            self._Challenge = Challenge
            self.available = True
            self.info = "モデル準備完了 (合成データで自動登録)"
        except Exception as exc:           # パッケージ未導入・登録失敗など
            self.available = False
            self.info = f"フォールバック動作 (理由: {exc})"
        return self.info

    # ---- 推論 -------------------------------------------------------------
    def infer(self, window: np.ndarray, sfreq: float) -> InferenceResult:
        if not self.available or self._pipeline is None:
            return self._dummy(window)
        try:
            return self._infer_model(window, sfreq)
        except Exception as exc:           # 未学習・形状不一致などは握りつぶしてダミーへ
            res = self._dummy(window)
            res.note = f"推論エラーのためダミー: {exc}"
            return res

    def _infer_model(self, window: np.ndarray, sfreq: float) -> InferenceResult:
        from eeg_biometric.data import EEGTrial

        pipe = self._pipeline
        trial = EEGTrial(data=window.astype(float).copy(), channels=self.channels, sfreq=sfreq)
        duration = window.shape[1] / sfreq

        # Liveness は raw 信号に対して評価（窓全体をチャレンジ窓とする）。
        challenge = self._Challenge(nonce="live", prompt_time=0.0,
                                    window=(0.0, duration), expected_blinks=1, tolerance=1)
        live = pipe.liveness.verify(trial, challenge)

        # 生体スコアは ATAR → チャネル選択 → 埋め込み → OC-SVM⊕LightGBM。
        enr = pipe.enrollments[self.subject_id]
        cleaned = pipe.atar.transform(trial)
        idx = enr.selector.selected_indices_for(cleaned)
        emb = enr.encoder.embed(cleaned, channel_idx=idx)
        accept, scores = enr.recognizer.verify(emb, threshold=enr.threshold)

        decision = bool(live.passed and accept)
        return InferenceResult(
            source="model",
            decision=decision,
            label="本人" if decision else "他人",
            score=float(scores["fused"]),
            ocsvm=float(scores["ocsvm_p"]),
            lgbm=float(scores["lgbm_p"]),
            live_pass=bool(live.passed),
            blinks=int(live.observed_in_window),
            threshold=float(enr.threshold),
            reason=("一致 (liveness+生体)" if decision
                    else ("ライブネス不成立" if not live.passed else "生体スコア不足")),
        )

    # ---- ダミー判定 -------------------------------------------------------
    def _dummy(self, window: Optional[np.ndarray]) -> InferenceResult:
        """信号ヒューリスティックに基づく仮判定（モデル未使用時）。"""
        if window is None or window.shape[1] < 8:
            return InferenceResult(source="dummy", label="—", reason="データ待機中")
        # 前頭(0,1)に高振幅の瞬目があるか（ロバスト z）。
        frontal = window[:2].mean(axis=0)
        med = np.median(frontal)
        mad = np.median(np.abs(frontal - med)) + 1e-6
        z = np.abs((frontal - med) / (1.4826 * mad))
        blink = int(np.sum(z > 5.0) > 0)
        # 振幅の安定度から擬似スコア（窓ごとに大きくは振れない値）。
        amp = float(np.median(np.std(window, axis=1)))
        score = float(np.clip(0.45 + 0.2 * np.tanh((amp - 10.0) / 15.0), 0.0, 0.95))
        decision = bool(blink and score >= 0.5)
        return InferenceResult(
            source="dummy",
            decision=decision,
            label="本人" if decision else "他人",
            score=score,
            live_pass=bool(blink),
            blinks=blink,
            threshold=0.5,
            reason="ダミー判定 (モデル未学習/未接続)",
        )


# =========================================================================== #
# 推論ワーカ（QThread）
# =========================================================================== #
class InferenceWorker(QtCore.QThread):
    """一定間隔でバッファから窓を切り出し、推論結果をシグナルで通知する。"""

    result_ready = QtCore.Signal(object)     # InferenceResult
    engine_status = QtCore.Signal(str)

    def __init__(self, buffer: RingBuffer, engine: InferenceEngine,
                 sfreq: float, window_samples: int, interval_ms: int = INFER_INTERVAL_MS) -> None:
        super().__init__()
        self.buffer = buffer
        self.engine = engine
        self.sfreq = float(sfreq)
        self.window_samples = int(window_samples)
        self.interval_ms = int(interval_ms)
        self._running = True
        self._min_samples = int(0.6 * sfreq)

    def run(self) -> None:
        self.engine_status.emit("モデル初期化中…")
        info = self.engine.setup()
        self.engine_status.emit(info)
        while self._running:
            window = self.buffer.get_latest(self.window_samples)
            if window is not None and window.shape[1] >= self._min_samples:
                result = self.engine.infer(window, self.sfreq)
            else:
                result = InferenceResult(source="waiting", label="—", reason="データ待機中")
            self.result_ready.emit(result)
            # 停止に素早く反応するため小刻みに待つ。
            waited = 0
            while self._running and waited < self.interval_ms:
                self.msleep(50)
                waited += 50

    def stop(self) -> None:
        self._running = False
        self.wait(2000)


# =========================================================================== #
# GUI ダッシュボード
# =========================================================================== #
class EEGDashboard(QtWidgets.QMainWindow):
    """8ch オシロスコープ + 認証ステータスパネルのメインウィンドウ。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EEG 生体認証 ダッシュボード")
        self.resize(1240, 720)

        self.buffer = RingBuffer(N_CHANNELS, int(BUFFER_SECONDS * FS))
        self.server = EEGTCPServer(self.buffer)
        self.simulator: Optional[SignalSimulator] = None
        self.engine = InferenceEngine(CHANNELS, FS)
        self.worker = InferenceWorker(self.buffer, self.engine, FS, int(WINDOW_SECONDS * FS))

        self._plot_samples = int(PLOT_SECONDS * FS)
        self._last_total = 0
        self._last_time = time.time()
        self._meas_hz = 0.0

        self._build_ui()
        self._connect_signals()

        # スレッド起動：受信サーバ → 推論ワーカ。
        self.server.start()
        self.worker.start()

        # 描画タイマ（GUI スレッド）。
        self._plot_timer = QtCore.QTimer(self)
        self._plot_timer.timeout.connect(self._update_plots)
        self._plot_timer.start(40)        # ~25 fps

    # ---- UI 構築 ----------------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # 左: オシロスコープ
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("#0b0f14")
        root.addWidget(self.glw, stretch=4)

        self.curves = []
        self.plots = []
        x0 = -self._plot_samples
        for c in range(N_CHANNELS):
            p = self.glw.addPlot(row=c, col=0)
            p.setMenuEnabled(False)
            p.setMouseEnabled(x=False, y=False)
            p.hideButtons()
            p.showGrid(x=False, y=True, alpha=0.15)
            p.setXRange(x0, 0, padding=0)
            p.getAxis("left").setWidth(46)
            p.getAxis("left").setLabel(CHANNELS[c])
            if c < N_CHANNELS - 1:
                p.getAxis("bottom").setStyle(showValues=False)
            else:
                p.getAxis("bottom").setLabel("最新 4 秒 (右端=現在)")
            curve = p.plot(pen=pg.mkPen(PALETTE[c], width=1.2))
            self.curves.append(curve)
            self.plots.append(p)

        # 右: ステータスパネル
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(330)
        root.addWidget(panel, stretch=0)
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(10)

        title = QtWidgets.QLabel("認証ステータス")
        title.setStyleSheet("font-size:15px; font-weight:600; color:#cbd5e1;")
        v.addWidget(title)

        self.verdict_label = QtWidgets.QLabel("—")
        self.verdict_label.setAlignment(_ALIGN_CENTER)
        self.verdict_label.setFixedHeight(120)
        self._set_verdict_style("#1f2937", "#e5e7eb")
        v.addWidget(self.verdict_label)

        self.source_badge = QtWidgets.QLabel("初期化中")
        self.source_badge.setAlignment(_ALIGN_CENTER)
        self.source_badge.setStyleSheet(
            "background:#374151; color:#e5e7eb; border-radius:6px; padding:4px; font-size:12px;")
        v.addWidget(self.source_badge)

        self.live_label = QtWidgets.QLabel("Liveness: —")
        self.live_label.setStyleSheet("font-size:16px; color:#e5e7eb;")
        v.addWidget(self.live_label)

        self.score_label = QtWidgets.QLabel("スコア: —")
        self.score_label.setStyleSheet("font-size:16px; color:#e5e7eb;")
        v.addWidget(self.score_label)

        self.score_bar = QtWidgets.QProgressBar()
        self.score_bar.setRange(0, 100)
        self.score_bar.setValue(0)
        self.score_bar.setTextVisible(True)
        v.addWidget(self.score_bar)

        self.detail_label = QtWidgets.QLabel("OC-SVM / LightGBM: —")
        self.detail_label.setStyleSheet("font-size:12px; color:#94a3b8;")
        v.addWidget(self.detail_label)

        self.reason_label = QtWidgets.QLabel("")
        self.reason_label.setWordWrap(True)
        self.reason_label.setStyleSheet("font-size:12px; color:#94a3b8;")
        v.addWidget(self.reason_label)

        v.addSpacing(8)
        line = QtWidgets.QFrame()
        line.setFrameShape(_HLINE)
        line.setStyleSheet("color:#334155;")
        v.addWidget(line)

        self.conn_label = QtWidgets.QLabel("接続: —")
        self.conn_label.setStyleSheet("font-size:12px; color:#cbd5e1;")
        v.addWidget(self.conn_label)

        self.rate_label = QtWidgets.QLabel("受信レート: 0.0 Hz")
        self.rate_label.setStyleSheet("font-size:12px; color:#cbd5e1;")
        v.addWidget(self.rate_label)

        v.addStretch(1)

        self.server_btn = QtWidgets.QPushButton("サーバ停止")
        self.server_btn.clicked.connect(self._toggle_server)
        v.addWidget(self.server_btn)

        self.sim_btn = QtWidgets.QPushButton("シミュレータ開始")
        self.sim_btn.clicked.connect(self._toggle_simulator)
        v.addWidget(self.sim_btn)

        hint = QtWidgets.QLabel(f"TCP 待受: {HOST}:{PORT}  /  {int(FS)}Hz × {N_CHANNELS}ch")
        hint.setStyleSheet("font-size:11px; color:#64748b;")
        v.addWidget(hint)

    def _connect_signals(self) -> None:
        self.worker.result_ready.connect(self._on_result)
        self.worker.engine_status.connect(self._on_engine_status)

    # ---- スタイルヘルパ ---------------------------------------------------
    def _set_verdict_style(self, bg: str, fg: str) -> None:
        self.verdict_label.setStyleSheet(
            f"background:{bg}; color:{fg}; border-radius:10px;"
            f"font-size:42px; font-weight:800;")

    # ---- 描画更新（QTimer） ----------------------------------------------
    def _update_plots(self) -> None:
        window = self.buffer.get_latest(self._plot_samples)
        if window is not None and window.shape[1] >= 2:
            m = window.shape[1]
            x = np.arange(-m, 0, dtype=np.float32)
            for c in range(N_CHANNELS):
                y = window[c]
                self.curves[c].setData(x, y)
                med = float(np.median(y))
                pad = max(8.0, 4.0 * float(np.std(y)))
                self.plots[c].setYRange(med - pad, med + pad, padding=0)

        # 接続状態と受信レート
        st = self.server.status()
        state = st.get("state")
        client = st.get("client")
        if state == "connected":
            self.conn_label.setText(f"接続: {client}")
            self.conn_label.setStyleSheet("font-size:12px; color:#34d399;")
        elif state == "listening":
            self.conn_label.setText(f"接続: 待受中 ({HOST}:{PORT})")
            self.conn_label.setStyleSheet("font-size:12px; color:#fbbf24;")
        elif state == "error":
            self.conn_label.setText(f"接続: エラー {st.get('error')}")
            self.conn_label.setStyleSheet("font-size:12px; color:#f87171;")
        else:
            self.conn_label.setText("接続: 停止")
            self.conn_label.setStyleSheet("font-size:12px; color:#94a3b8;")

        now = time.time()
        total = self.buffer.total
        dt = now - self._last_time
        if dt >= 0.5:
            self._meas_hz = (total - self._last_total) / dt
            self._last_total = total
            self._last_time = now
            self.rate_label.setText(f"受信レート: {self._meas_hz:5.1f} Hz")

    # ---- 推論結果の反映 ---------------------------------------------------
    def _on_result(self, result: InferenceResult) -> None:
        if result.source == "waiting":
            self.verdict_label.setText("待機中")
            self._set_verdict_style("#1f2937", "#9ca3af")
            self.reason_label.setText(result.reason)
            return

        if result.decision:
            self.verdict_label.setText("本人\nACCEPT")
            self._set_verdict_style("#064e3b", "#34d399")
        else:
            self.verdict_label.setText("他人\nREJECT")
            self._set_verdict_style("#4c0519", "#fb7185")

        if result.source == "model":
            self.source_badge.setText("判定ソース: MODEL")
            self.source_badge.setStyleSheet(
                "background:#1d4ed8; color:#e5e7eb; border-radius:6px; padding:4px; font-size:12px;")
        else:
            self.source_badge.setText("判定ソース: DUMMY (フォールバック)")
            self.source_badge.setStyleSheet(
                "background:#92400e; color:#fde68a; border-radius:6px; padding:4px; font-size:12px;")

        if result.live_pass is None:
            self.live_label.setText("Liveness: —")
        else:
            ok = result.live_pass
            self.live_label.setText(
                f"Liveness: {'PASS' if ok else 'FAIL'}  (瞬目 {result.blinks} 回)")
            self.live_label.setStyleSheet(
                f"font-size:16px; color:{'#34d399' if ok else '#fb7185'};")

        if result.score is None:
            self.score_label.setText("スコア: —")
            self.score_bar.setValue(0)
        else:
            self.score_label.setText(f"スコア(融合): {result.score:.3f}  (閾値 {result.threshold:.2f})")
            self.score_bar.setValue(int(round(result.score * 100)))

        if result.ocsvm is not None and result.lgbm is not None:
            self.detail_label.setText(
                f"OC-SVM(SVDD): {result.ocsvm:.3f}   LightGBM: {result.lgbm:.3f}")
        else:
            self.detail_label.setText("OC-SVM / LightGBM: —")

        self.reason_label.setText((result.reason + ("  /  " + result.note if result.note else "")))

    def _on_engine_status(self, info: str) -> None:
        self.source_badge.setText(info)

    # ---- ボタン操作 -------------------------------------------------------
    def _toggle_server(self) -> None:
        st = self.server.status().get("state")
        if st in ("listening", "connected"):
            self.server.stop()
            self.server_btn.setText("サーバ開始")
        else:
            if not self.server.is_alive():
                self.server = EEGTCPServer(self.buffer)
                self.server.start()
            self.server_btn.setText("サーバ停止")

    def _toggle_simulator(self) -> None:
        if self.simulator is None or not self.simulator.is_alive():
            self.simulator = SignalSimulator()
            self.simulator.start()
            self.sim_btn.setText("シミュレータ停止")
        else:
            self.simulator.stop()
            self.simulator = None
            self.sim_btn.setText("シミュレータ開始")

    # ---- 終了処理 ---------------------------------------------------------
    def closeEvent(self, event) -> None:
        try:
            self._plot_timer.stop()
            self.worker.stop()
            if self.simulator is not None:
                self.simulator.stop()
            self.server.stop()
        finally:
            super().closeEvent(event)


# =========================================================================== #
# エントリポイント
# =========================================================================== #
def main() -> None:
    pg.setConfigOptions(antialias=True)
    app = pg.mkQApp("EEG 生体認証 ダッシュボード")
    win = EEGDashboard()
    win.show()
    exec_fn = getattr(app, "exec", None) or app.exec_
    sys.exit(exec_fn())


if __name__ == "__main__":
    main()
