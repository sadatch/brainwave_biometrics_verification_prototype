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
* セキュリティ: 待受は既定で **127.0.0.1**（全 IF 公開は ``EEG_HOST=0.0.0.0`` を明示）。
  ``EEG_TOKEN`` を設定すると接続時に共有トークン・ハンドシェイクを要求します。
* Liveness は **能動的チャレンジ&レスポンス**: ランダム時刻に「今すぐ N 回まばたき」を
  提示し、その応答窓のみを nonce 拘束で評価。最終判定 = 生体一致 AND 直近の Liveness 成功。

必要パッケージ: ``pyqtgraph`` と ``PyQt5``（または ``PySide6``）、``numpy``。
    pip install pyqtgraph PyQt5 numpy
"""
from __future__ import annotations

import hmac
import os
import socket
import struct
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

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
# 既定はループバックのみ。全 IF に公開する場合のみ環境変数 EEG_HOST=0.0.0.0 を明示。
HOST = os.environ.get("EEG_HOST", "127.0.0.1")
PORT = int(os.environ.get("EEG_PORT", "8888"))
# 任意の共有トークン(接続時ハンドシェイク)。空なら無効。環境変数 EEG_TOKEN で設定。
SHARED_TOKEN = os.environ.get("EEG_TOKEN", "")
N_CHANNELS = 8
# 8ch の名称。Liveness 検知が前頭(Fp1/Fp2)を参照するため先頭に配置する。
CHANNELS: List[str] = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "O1", "O2"]
FS = 250.0                       # サンプリング周波数 [Hz]
BUFFER_SECONDS = 4.0             # リングバッファ長
PLOT_SECONDS = 4.0               # 表示窓
WINDOW_SECONDS = 3.0             # 推論窓(生体)
INFER_INTERVAL_MS = 1000         # 生体推論間隔
PACKET_FORMAT = "<I8f"           # uint32 + float32 x8
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)  # = 36
SUBJECT_ID = "S001"

# --- 能動的チャレンジ&レスポンス(Liveness) ---
CHALLENGE_EVERY = (6.0, 11.0)    # チャレンジ発行間隔(秒, ランダム)
CHALLENGE_LEAD = 1.2             # 「まもなく」表示〜プロンプトまでの猶予(秒)
CHALLENGE_WIDTH = 2.2            # 応答(まばたき)受付窓(秒)
CHALLENGE_BLINKS = 2             # 要求まばたき回数
LIVENESS_FRESH_SEC = 10.0        # 最終認証で要求する liveness 成功の鮮度(秒)

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
        self._t = np.zeros(self.capacity, dtype=np.float64)   # 各サンプルの到着時刻(epoch秒)
        self._idx = 0           # 次に書き込む位置
        self._count = 0         # 充填済みサンプル数
        self._total = 0         # 累積受信サンプル数（レート計測用）
        self._lock = threading.Lock()

    def append(self, sample: np.ndarray) -> None:
        """1 サンプル ``(n_channels,)`` を追加する。"""
        now = time.time()
        with self._lock:
            self._buf[:, self._idx] = sample
            self._t[self._idx] = now
            self._idx = (self._idx + 1) % self.capacity
            self._count = min(self._count + 1, self.capacity)
            self._total += 1

    def get_window_by_time(self, t0: float, t1: float) -> Optional[np.ndarray]:
        """到着時刻が ``[t0, t1]`` に入るサンプルを古い→新しい順で返す。

        チャレンジ&レスポンスで「指定した応答窓に対応する区間」だけを切り出すのに使う。
        """
        with self._lock:
            if self._count <= 0:
                return None
            idx = (np.arange(self._idx - self._count, self._idx) % self.capacity)
            ts = self._t[idx]
            mask = (ts >= t0) & (ts <= t1)
            if not np.any(mask):
                return None
            cols = idx[mask]
            return self._buf[:, cols].copy()

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
    """``HOST:PORT``（既定 ``127.0.0.1:8888``）で待ち受け、36B パケットを解析する。

    部分受信のバッファリング、クライアント切断・再接続、応答性のある停止
    （``settimeout`` ベース）、任意の共有トークン・ハンドシェイク（``SHARED_TOKEN``）
    に対応する。全 IF への公開は ``EEG_HOST=0.0.0.0`` を明示した場合のみ。GUI とは
    ``RingBuffer`` とスレッドセーフな状態辞書のみを共有する。
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
        token_bytes = SHARED_TOKEN.encode("utf-8")
        token_ok = (SHARED_TOKEN == "")        # トークン未設定なら検証不要
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
                if not token_ok:       # 接続時の共有トークン・ハンドシェイク
                    if len(pending) < len(token_bytes):
                        continue
                    if not hmac.compare_digest(pending[:len(token_bytes)], token_bytes):
                        self._set_status(state="listening", client=None, error="auth_failed")
                        return         # 認証失敗 → 切断
                    pending = pending[len(token_bytes):]
                    token_ok = True
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

    def __init__(self, port: int = PORT, fs: float = FS, seed: int = 1,
                 cooperative: bool = True) -> None:
        super().__init__(daemon=True, name="SignalSimulator")
        self.port = port
        self.fs = float(fs)
        self.cooperative = bool(cooperative)
        self._running = threading.Event()
        self._running.set()
        self._rng = np.random.default_rng(seed)
        self._sched_lock = threading.Lock()
        self._scheduled: List[float] = []     # 協力的応答の瞬目中心時刻(epoch秒)

    def stop(self) -> None:
        self._running.clear()

    def request_blinks(self, centers: Sequence[float]) -> None:
        """協力的ユーザを模し、指定 wall-clock 時刻にまばたきを発火するよう予約する。"""
        with self._sched_lock:
            self._scheduled.extend(float(c) for c in centers)

    def _blink_envelope(self, now: float, center: float, width: float = 0.06) -> float:
        return float(np.exp(-0.5 * ((now - center) / width) ** 2))

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
                    if SHARED_TOKEN:                  # 接続時ハンドシェイク
                        sock.sendall(SHARED_TOKEN.encode("utf-8"))
                except OSError:
                    time.sleep(0.5)
                    continue
            now = time.time()
            if now >= next_blink:           # 自発的な瞬目
                blink_t0 = now
                blink_until = now + 0.25
                next_blink = now + self._rng.uniform(3.0, 6.0)
            t = n / self.fs
            sample = np.empty(N_CHANNELS, dtype=np.float32)
            for c in range(N_CHANNELS):
                v = gains[c] * 8.0 * np.sin(2 * np.pi * alpha_f[c] * t + phase[c])
                v += self._rng.standard_normal() * 4.0           # 背景ノイズ
                sample[c] = v
            env = 0.0
            if now < blink_until:                                # 自発瞬目
                env += self._blink_envelope(now, blink_t0 + 0.1)
            with self._sched_lock:                               # 協力的応答(チャレンジ)
                self._scheduled = [c for c in self._scheduled if c > now - 1.0]
                for center in self._scheduled:
                    env += self._blink_envelope(now, center)
            if env > 0:                                          # 前頭に EOG を重畳
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
            # チャレンジ&レスポンス: nonce を単回使用とし失効も有効化(echo は簡易 sim では無効)。
            pipe.liveness.track_nonce = True
            pipe.liveness.max_age_seconds = 30.0
            pipe.liveness.require_nonce_echo = False

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
        """生体(identity)のみを評価する。Liveness は別系統(チャレンジ&レスポンス)。"""
        from eeg_biometric.data import EEGTrial

        pipe = self._pipeline
        trial = EEGTrial(data=window.astype(float).copy(), channels=self.channels, sfreq=sfreq)
        enr = pipe.enrollments[self.subject_id]
        cleaned = pipe.atar.transform(trial)
        missing = [ch for ch in enr.selected_channels if cleaned.channel_index(ch) is None]
        if missing:
            raise RuntimeError(f"montage mismatch: {missing}")
        idx = enr.selector.selected_indices_for(cleaned)
        emb = enr.encoder.embed(cleaned, channel_idx=idx)
        accept, scores = enr.recognizer.verify(emb)   # 較正済みのモード別閾値を使用
        return InferenceResult(
            source="model",
            decision=bool(accept),
            label="本人" if accept else "他人",
            score=float(scores["fused"]),
            ocsvm=float(scores["ocsvm_p"]),
            lgbm=float(scores["lgbm_p"]),
            threshold=float(enr.threshold),
            reason=("生体一致" if accept
                    else f"生体スコア不足(oc={scores['ocsvm_p']:.2f},lgbm={scores['lgbm_p']:.2f})"),
        )

    # ---- Liveness(能動的チャレンジ&レスポンス) ---------------------------
    def check_liveness(self, window: Optional[np.ndarray], sfreq: float,
                       nonce: str, expected_blinks: int = CHALLENGE_BLINKS) -> dict:
        """応答窓のスライスにオンキューまばたきが含まれるかを評価する。"""
        if window is None or window.shape[1] < 8:
            return {"source": "none", "passed": False, "blinks": 0, "reason": "応答窓のデータ不足"}
        if not self.available or self._pipeline is None or self._Challenge is None:
            return self._dummy_liveness(window, expected_blinks)
        try:
            from eeg_biometric.data import EEGTrial
            duration = window.shape[1] / sfreq
            trial = EEGTrial(data=window.astype(float).copy(), channels=self.channels, sfreq=sfreq)
            ch = self._Challenge(nonce=nonce, prompt_time=0.0, window=(0.0, duration),
                                 expected_blinks=expected_blinks, tolerance=1)
            res = self._pipeline.liveness.verify(trial, ch)
            return {"source": "model", "passed": bool(res.passed),
                    "blinks": int(res.observed_in_window), "reason": "; ".join(res.reasons) or "ok"}
        except Exception as exc:
            d = self._dummy_liveness(window, expected_blinks)
            d["reason"] += f" (err:{exc})"
            return d

    # ---- ダミー判定 -------------------------------------------------------
    def _dummy(self, window: Optional[np.ndarray]) -> InferenceResult:
        """信号ヒューリスティックに基づく仮の生体判定（モデル未使用時）。"""
        if window is None or window.shape[1] < 8:
            return InferenceResult(source="dummy", label="—", reason="データ待機中")
        amp = float(np.median(np.std(window, axis=1)))
        score = float(np.clip(0.45 + 0.2 * np.tanh((amp - 10.0) / 15.0), 0.0, 0.95))
        decision = bool(score >= 0.5)
        return InferenceResult(
            source="dummy", decision=decision, label="本人" if decision else "他人",
            score=score, threshold=0.5, reason="ダミー生体判定 (モデル未学習/未接続)",
        )

    @staticmethod
    def _dummy_liveness(window: np.ndarray, expected_blinks: int) -> dict:
        """前頭の高振幅遷移の数からまばたきを概算するダミー liveness。"""
        frontal = window[:2].mean(axis=0)
        med = np.median(frontal)
        mad = np.median(np.abs(frontal - med)) + 1e-6
        above = (np.abs((frontal - med) / (1.4826 * mad)) > 5.0).astype(int)
        blinks = int(np.sum(np.diff(above) == 1))
        return {"source": "dummy", "passed": bool(blinks >= 1), "blinks": blinks,
                "reason": "ダミーliveness"}


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

        # チャレンジ&レスポンス(Liveness)状態
        self._active_challenge: Optional[dict] = None
        self._last_live: Optional[dict] = None
        self._last_live_pass_ts = 0.0
        self._last_bio: Optional[InferenceResult] = None
        self._closing = False

        self._build_ui()
        self._connect_signals()

        # スレッド起動：受信サーバ → 推論ワーカ。
        self.server.start()
        self.worker.start()

        # 描画タイマ（GUI スレッド）。
        self._plot_timer = QtCore.QTimer(self)
        self._plot_timer.timeout.connect(self._update_plots)
        self._plot_timer.start(40)        # ~25 fps

        # 最初のチャレンジは少し待ってから（モデル初期化・データ蓄積の猶予）。
        QtCore.QTimer.singleShot(4000, self._schedule_next_challenge)

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

        # 能動的チャレンジのプロンプト（ランダム時刻に「今すぐまばたき！」を表示）
        self.prompt_label = QtWidgets.QLabel("チャレンジ: 待機中")
        self.prompt_label.setAlignment(_ALIGN_CENTER)
        self.prompt_label.setFixedHeight(52)
        self._set_prompt_style("#1f2937", "#9ca3af")
        v.addWidget(self.prompt_label)

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
            # Liveness の鮮度失効を最終判定へ反映するため定期的に再評価。
            self._update_combined()

    # ---- 推論結果の反映 ---------------------------------------------------
    def _on_result(self, result: InferenceResult) -> None:
        """生体(identity)結果の反映。最終判定は Liveness 鮮度と AND して決める。"""
        self._last_bio = result
        if result.source == "waiting":
            self.verdict_label.setText("待機中")
            self._set_verdict_style("#1f2937", "#9ca3af")
            self.reason_label.setText(result.reason)
            return

        if result.source == "model":
            self.source_badge.setText("判定ソース: MODEL (生体ブランチAND)")
            self.source_badge.setStyleSheet(
                "background:#1d4ed8; color:#e5e7eb; border-radius:6px; padding:4px; font-size:12px;")
        else:
            self.source_badge.setText("判定ソース: DUMMY (フォールバック)")
            self.source_badge.setStyleSheet(
                "background:#92400e; color:#fde68a; border-radius:6px; padding:4px; font-size:12px;")

        if result.score is None:
            self.score_label.setText("生体スコア: —")
            self.score_bar.setValue(0)
        else:
            self.score_label.setText(f"生体スコア(融合): {result.score:.3f}")
            self.score_bar.setValue(int(round(result.score * 100)))

        if result.ocsvm is not None and result.lgbm is not None:
            self.detail_label.setText(
                f"OC-SVM(SVDD): {result.ocsvm:.3f}   LightGBM: {result.lgbm:.3f}")
        else:
            self.detail_label.setText("OC-SVM / LightGBM: —")

        self._update_combined()

    def _on_engine_status(self, info: str) -> None:
        self.source_badge.setText(info)

    # ---- 統合判定（生体 AND 直近の Liveness 成功）-------------------------
    def _update_combined(self) -> None:
        bio = self._last_bio
        if bio is None or bio.source == "waiting":
            return
        live_fresh = (time.time() - self._last_live_pass_ts) <= LIVENESS_FRESH_SEC
        biom_ok = bool(bio.decision)
        if biom_ok and live_fresh:
            self.verdict_label.setText("本人\nACCEPT")
            self._set_verdict_style("#064e3b", "#34d399")
            self.reason_label.setText("一致 (生体 AND 直近の Liveness 成功)")
        else:
            self.verdict_label.setText("他人\nREJECT")
            self._set_verdict_style("#4c0519", "#fb7185")
            if not biom_ok:
                self.reason_label.setText(f"生体不一致: {bio.reason}")
            elif not live_fresh:
                self.reason_label.setText("Liveness 未成立/失効 — チャレンジに応答してください")
            else:
                self.reason_label.setText(bio.reason)

    def _update_liveness_label(self, res: dict) -> None:
        ok = bool(res.get("passed"))
        self.live_label.setText(
            f"Liveness: {'PASS' if ok else 'FAIL'}  "
            f"(まばたき {res.get('blinks', 0)} 回 / {res.get('source')})")
        self.live_label.setStyleSheet(
            f"font-size:16px; color:{'#34d399' if ok else '#fb7185'};")

    # ---- 能動的チャレンジ&レスポンス -------------------------------------
    def _set_prompt_style(self, bg: str, fg: str) -> None:
        self.prompt_label.setStyleSheet(
            f"background:{bg}; color:{fg}; border-radius:8px; font-size:20px; font-weight:700;")

    def _schedule_next_challenge(self) -> None:
        if self._closing:
            return
        delay = float(np.random.uniform(*CHALLENGE_EVERY))
        QtCore.QTimer.singleShot(int(delay * 1000), self._issue_challenge)

    def _issue_challenge(self) -> None:
        if self._closing:
            return
        now = time.time()
        w0 = now + CHALLENGE_LEAD
        w1 = w0 + CHALLENGE_WIDTH
        self._active_challenge = {"nonce": uuid.uuid4().hex, "w0": w0, "w1": w1}
        self.prompt_label.setText("まもなくチャレンジ…")
        self._set_prompt_style("#374151", "#fde68a")
        # 協力的シミュレータは応答窓内に所定回数まばたきする(実機/人なら本人が応答)。
        if self.simulator is not None and self.simulator.is_alive():
            centers = [w0 + 0.5 + 0.7 * i for i in range(CHALLENGE_BLINKS)]
            self.simulator.request_blinks(centers)
        QtCore.QTimer.singleShot(int(CHALLENGE_LEAD * 1000), self._show_prompt_now)
        QtCore.QTimer.singleShot(int((CHALLENGE_LEAD + CHALLENGE_WIDTH + 0.4) * 1000),
                                 self._evaluate_challenge)

    def _show_prompt_now(self) -> None:
        if self._closing or self._active_challenge is None:
            return
        self.prompt_label.setText(f"今すぐ {CHALLENGE_BLINKS} 回まばたき！")
        self._set_prompt_style("#7c2d12", "#fdba74")

    def _evaluate_challenge(self) -> None:
        if self._closing:
            return
        ac = self._active_challenge
        self._active_challenge = None
        if ac is not None:
            window = self.buffer.get_window_by_time(ac["w0"], ac["w1"])
            res = self.engine.check_liveness(window, FS, ac["nonce"], CHALLENGE_BLINKS)
            self._last_live = res
            if res.get("passed"):
                self._last_live_pass_ts = time.time()
            self._update_liveness_label(res)
            self._update_combined()
        self.prompt_label.setText("チャレンジ: 待機中")
        self._set_prompt_style("#1f2937", "#9ca3af")
        self._schedule_next_challenge()

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
        self._closing = True
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
