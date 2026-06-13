# EEG 生体認証システム（研究プロトタイプ）

脳波（EEG）を用いた **1:1 防御的生体認証**の研究プロトタイプです。  
ESP32 から送信される EEG をリアルタイムで受信し、なりすましとプレゼンテーション攻撃を検知しながら本人照合を行います。

> **倫理・スコープ**：本実装は学術目的のプロトタイプです。MNE-Python の公開サンプルデータまたは NumPy で合成した波形のみを使用します。実在の個人データ収集・本番運用は対象外です。

---

## システム概要

```
ESP32 (ADS1299, 8ch, 250Hz)
    │  TCP バイナリストリーム (36B/packet)
    ▼
main.py  ─── EEGTCPServer ──► RingBuffer
              │                （get_window_by_time でチャレンジ窓も切り出し）
              InferenceWorker (QThread)
              │
              InferenceEngine
              │  ├─ check_liveness(応答窓, nonce) ←─ 能動的チャレンジ&レスポンス
              │  └─ infer(window) ─► EEGBiometricPipeline
              │
              eeg_biometric.pipeline.EEGBiometricPipeline
              │
              ┌──────────────────────────────────────────┐
              │ RAW  ─► LivenessDetector (pre-ATAR)      │
              │  │pass               │fail → REJECT       │
              │  └─► ATAR ─► Elastic-Net ─► MAEEG/手作り │
              │              埋め込み ─► OC-SVM⊕LightGBM  │
              │                     ANDゲート ─► ACCEPT  │
              └──────────────────────────────────────────┘
              │
              生体OK AND 直近Liveness成功 → 最終ACCEPT
              │
              EEGDashboard (PyQtGraph GUI)
              8ch オシロスコープ + 認証ステータスパネル
```

### システムフロー図（Mermaid）

```mermaid
flowchart TD
    ESP32["ESP32\nADS1299, 8ch, 250Hz\nTCP バイナリストリーム 36B/packet"]

    subgraph main["main.py"]
        TCP["EEGTCPServer\nTCP受信・再接続\nトークン認証対応"]
        BUF["RingBuffer\n固定長循環バッファ\nget_window_by_time()"]
        IW["InferenceWorker\nQThread 1秒周期"]
        IE["InferenceEngine\ninfer() / check_liveness()"]
        TCP --> BUF --> IW --> IE
    end

    subgraph pipeline["EEGBiometricPipeline"]
        RAW["RAW EEG"]

        subgraph liveness_check["ライブネス検査（ISO/IEC 30107）"]
            LD["LivenessDetector\n瞬目チャレンジ&レスポンス"]
        end

        ATAR["ATARPreprocessor\nウェーブレット アーティファクト除去"]
        EN["ElasticNetChannelSelector\n安定チャネル選択"]

        subgraph encoder["特徴抽出"]
            MAEEG["MAEEGEncoder / GMAEEGEncoder\n深層特徴量"]
            HC["HandcraftedSpectralEncoder\n手作りスペクトル特徴量"]
        end

        subgraph recognizer["識別（オープンセット）"]
            OCSVM["OC-SVM\n未知他人対応"]
            LGBM["LightGBM\n既知他人識別"]
            FUSE["Platt 融合 AND ロジック"]
            OCSVM & LGBM --> FUSE
        end

        REJECT["❌ REJECT\nリプレイ / スプーフ検出"]
        ACCEPT["✅ ACCEPT\n本人照合成功"]

        RAW --> LD
        LD -- fail --> REJECT
        LD -- pass --> ATAR --> EN --> encoder
        encoder --> recognizer
        FUSE -- ACCEPT --> ACCEPT
        FUSE -- REJECT --> REJECT
    end

    GUI["EEGDashboard\n8ch オシロスコープ\n認証ステータスパネル"]

    ESP32 --> TCP
    IE --> RAW
    pipeline --> GUI
```

---

## リポジトリ構成

```
.
├── main.py                  # リアルタイムダッシュボード（PyQtGraph GUI）
├── tests/
│   └── test_smoke.py        # スモークテスト（enroll→verify・NumPy2対応・anti-replay等）
├── .github/
│   └── workflows/
│       └── ci.yml           # CI（NumPy 1.x / 2.x 両行列 + eeg_biometric デモ実行）
└── eeg_biometric/           # サーバ側推論パイプライン（Pythonパッケージ）
    ├── __init__.py
    ├── dsp.py               # 共通DSP（PSD・帯域パワー・ロバストz・ピーク検出）
    ├── data.py              # EEGDataSource / EEGTrial（MNE + 合成フォールバック）
    ├── preprocess.py        # ATARPreprocessor（ウェーブレットアーティファクト除去）
    ├── channels.py          # ElasticNetChannelSelector / PerChannelFeatureExtractor（安定チャネル選択）
    ├── features.py          # MAEEGEncoder / GMAEEGEncoder / HandcraftedSpectralEncoder
    ├── recognition.py       # OpenSetRecognizer（OC-SVM/SVDD ⊕ LightGBM）
    ├── liveness.py          # LivenessDetector（ISO/IEC 30107 能動チャレンジ&レスポンス）
    ├── adversarial.py       # EEG-GAN / サロゲート生成 + なりすましレッドチーム
    ├── pipeline.py          # EEGBiometricPipeline 統合 + デモ
    ├── requirements.txt     # 依存パッケージ
    ├── README.md            # サブパッケージ詳細（英語）
    └── README.ja.md         # サブパッケージ詳細（日本語）
```

---

## インストール

```bash
# 最小構成（NumPy のみ）
pip install numpy pyqtgraph PyQt5

# 推奨（全機能）
pip install -r eeg_biometric/requirements.txt
pip install pyqtgraph PyQt5
```

### 依存関係とフォールバック

| パッケージ | 有効になる機能 | 無い場合のフォールバック |
|---|---|---|
| `numpy` | （必須） | — |
| `scipy` | Butterworth/Welch/`find_peaks` | FFT / 局所最大 |
| `PyWavelets` | 本来の ATAR（WPD/DWT） | ロバスト振幅減衰 |
| `scikit-learn` | Elastic-Net + OC-SVM + スケーリング | NumPy ロジスティック + マハラノビス |
| `lightgbm` | LightGBM ブースティング枝 | 勾配ブースティング / ロジスティック回帰 |
| `torch` | MAEEG/GMAEEG + EEG-GAN | 手作りエンコーダ + 位相ランダム化サロゲート |
| `mne` | PhysioNet EEGBCI 公開データ | NumPy 合成波形 |

---

## 実行方法

### 1. リアルタイムダッシュボード（`main.py`）

```bash
python main.py
```

- 既定で `127.0.0.1:8888` で TCP 待受を開始します（全 IF 公開は `EEG_HOST=0.0.0.0` を明示）。
- ESP32 が接続されていなくてもGUI内の **「シミュレータ開始」** ボタンで内蔵シグナルシミュレータが起動し、合成EEGをリアルタイム受信できます。
- 起動時に `InferenceEngine` が `eeg_biometric.pipeline` を使って自動登録（enroll）を実行します。失敗した場合は信号ヒューリスティックによるダミー判定にフォールバックします。
- 数秒おきに能動的チャレンジ（「今すぐ N 回まばたき」）を発行し、Liveness を評価します。最終判定 = 生体一致 AND 直近の Liveness 成功。

#### 環境変数

| 変数 | 既定値 | 説明 |
|---|---|---|
| `EEG_HOST` | `127.0.0.1` | TCP 待受アドレス（`0.0.0.0` で全 IF 公開） |
| `EEG_PORT` | `8888` | TCP 待受ポート |
| `EEG_TOKEN` | （空）| 接続時の共有トークン。設定すると接続ハンドシェイクを要求 |

#### パケット仕様（ESP32 側）

```
36バイト・リトルエンディアン: uint32 タイムスタンプ + float32 × 8ch
フォーマット文字列: "<I8f"
ポート: 8888
```

### 2. パイプライン単体デモ（`eeg_biometric/pipeline.py`）

```bash
# 親ディレクトリから
python -m eeg_biometric.pipeline

# または
cd eeg_biometric && python pipeline.py
```

5つのシナリオ（本人受理・他人拒否・リプレイ拒否・タイミング不一致拒否・GAN スプーフ拒否）を実行し、FAR / FRR / ACC を表示します。

### 3. スモークテスト

```bash
python -m pytest -q tests
```

enroll→verify・NumPy 2.0 対応（`np.trapz` → `np.trapezoid`）・montage 不一致検出・anti-replay・FAR 閾値を CI でチェックします（NumPy 1.x / 2.x の両行列）。

---

## 主要コンポーネント

### `eeg_biometric` パッケージ

| コンポーネント | 説明 |
|---|---|
| **ATAR前処理** | ウェーブレット（WPD/DWT）で瞬目・筋電を除去。単一チャネル・低遅延でストリーミングに適合 |
| **Elastic-Net チャネル選択** | `ElasticNetChannelSelector` + `PerChannelFeatureExtractor` + `build_selection_dataset`。Stability selection で容積伝導の相関を考慮しつつ安定した電極セットを抽出 |
| **MAEEG / GMAEEG** | 6層conv → 8層Transformer（192→64次元）のマスク自己符号化器。事前学習重み未同梱のため、デフォルトは HandcraftedSpectralEncoder |
| **OC-SVM / SVDD ⊕ LightGBM** | One-Class SVM（SVDD、未知他人対応オープンセット）と LightGBM（既知他人の識別）を Platt 融合。`recognizer_mode="and"` がデフォルト（両ブランチ通過必須） |
| **LivenessDetector** | ISO/IEC 30107 準拠。ランダムな時間窓への瞬目をチャレンジとして、リプレイ・スプライス攻撃を拒否。nonce 追跡・有効期限・モンタージュ付きチャレンジに対応 |
| **EEG-GAN / サロゲート** | 登録データ拡張とレッドチーム用（防御目的のみ）。`use_gan_augmentation` / `gan_backend` で切替 |

### `main.py` ダッシュボード

| コンポーネント | 説明 |
|---|---|
| **EEGTCPServer** | TCP 受信スレッド。部分受信バッファリング・再接続・`EEG_TOKEN` による共有トークン認証に対応 |
| **RingBuffer** | `threading.Lock` で保護された固定長循環バッファ。`get_latest()` に加え `get_window_by_time()` でチャレンジ応答窓を時刻指定で切り出せる |
| **InferenceResult** | 推論結果を保持するデータクラス（source / decision / score / ocsvm / lgbm / live_pass 等） |
| **InferenceEngine** | `EEGBiometricPipeline` をラップし `infer()`（生体）と `check_liveness()`（Liveness）を提供。未学習・例外時はダミー判定にフォールバック |
| **InferenceWorker** | 1秒ごとにバッファから窓を切り出し `InferenceEngine` に渡す QThread |
| **SignalSimulator** | ESP32 不在時のテスト用 TCP 合成シグナル送信スレッド。`request_blinks()` でチャレンジへの協力的応答をシミュレート |
| **EEGDashboard** | 8ch オシロスコープ + 能動的チャレンジプロンプト + Liveness / スコア / 最終判定のステータスパネル |

---

## 詳細ドキュメント

- [eeg_biometric/README.md](eeg_biometric/README.md) — 設計判断・アーキテクチャ詳細（英語）
- [eeg_biometric/README.ja.md](eeg_biometric/README.ja.md) — 設計判断・アーキテクチャ詳細（日本語）

---

## ライセンス

本リポジトリは研究・教育目的のプロトタイプです。実運用・商用利用・実在個人データへの適用は想定していません。
