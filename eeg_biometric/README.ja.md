# eeg_biometric — 脳波(EEG)による防御的1:1生体認証（研究プロトタイプ）

主張された本人性をEEGから**照合(verify)**し、なりすまし(他人)とプレゼンテーション攻撃を**拒否**する、モジュール式のサーバ側推論パイプラインです。本リポジトリは**防御的**な学術プロトタイプであり、正規ユーザを認証し、なりすましを検知することを目的とします。検証には **MNE-Python の公開サンプルデータ**または **NumPy で合成した波形**のみを使用し、実在個人からのデータ収集や本番運用は対象外です。

> 添付の研究レポート（仮説1〜4および Plan A/Plan B の結論）を設計の根拠としています。本実装は **Plan B（ESP32-S3 でのエッジ収集 → セキュアトンネル → GPU サーバで GMAEEG＋OC-SVM＋LightGBM 推論）** のうち、**サーバ側推論パイプライン**に相当します。

## アーキテクチャ

```
RAW trial ─┬─ 前頭チャネル ──► LivenessDetector (ATARの前)  ──失敗──► REJECT
           │                                   │成功
           └─ 全チャネル ───► ATAR ─► Elastic-Net チャネル選択 ─► MAEEG/手作り
                                       埋め込み ─► OC-SVM ⊕ LightGBM ─┐
                                                                       │
                              ACCEPT  ◄────────────── AND ─────────────┘
```

ライブネス検知は **raw（ATAR前）** 信号を見ます（瞬目/EOG の証拠が ATAR で消える前に検査するため）。生体特徴の経路は **ATAR でクリーン化した** 信号を見ます（同じ瞬目に支配されないように）。最終的な受理は**両ステージの通過**を要求します。

## モジュール

| ファイル | クラス | 役割 |
|------|-------|------|
| `dsp.py` | — | 共通DSP（PSD・帯域パワー・ロバストz・ピーク検出）。SciPy→NumPy フォールバック |
| `data.py` | `EEGDataSource`, `EEGTrial` | MNE EEGBCI ローダ（合成フォールバック）＋被験者ごとのシグネチャ |
| `preprocess.py` | `ATARPreprocessor` | ウェーブレットによるアーティファクト除去。可変・単一チャネル・低遅延（WPD/DWT） |
| `channels.py` | `ElasticNetChannelSelector` | 安定なL1/L2チャネル・特徴選択（stability selection） |
| `features.py` | `MAEEGEncoder`, `GMAEEGEncoder`, `HandcraftedSpectralEncoder` | 凍結埋め込み：6層conv→8層Transformer(192→64)＋手作りフォールバック |
| `recognition.py` | `OpenSetRecognizer` | One-Class SVM (SVDD) ⊕ LightGBM、キャリブレーション融合 |
| `liveness.py` | `LivenessDetector`, `Challenge` | ISO/IEC 30107 能動的チャレンジ＆レスポンス PAD |
| `adversarial.py` | `EEGGAN`, `SurrogateEEGGenerator`, `PresentationAttackSimulator` | GAN/サロゲートによる拡張＋なりすましレッドチーム（防御目的） |
| `pipeline.py` | `EEGBiometricPipeline`, `main()` | 統合＋一気通貫デモ |

## 主要な設計判断

**ICA ではなく ATAR。** ICA は全多チャネルブロックと比較的高コストな分離行列推定を要し、低遅延・チャネルストリーミング推論には不向きです。ATAR は短い重なり窓で**単一チャネルずつ**処理するため、ESP32→サーバのストリーミングに適し、遅延はおよそ1窓に収まります。その中核は古典的なウェーブレットデノイズの**逆**で、アーティファクト（瞬目・EOG・筋電）は**高振幅のウェーブレット係数**だと仮定し、*大きい*係数を抑制して小さな神経リズムを残します。1つのノブ `beta` が動作点（穏やか↔積極的）を決め、モードは `soft`/`linatten`/`elim`。分解は既定で**ウェーブレットパケット分解(WPD)**（資料が指定する変種。近似・詳細の両分岐を分解し均一な周波数分解能を得る）で、軽量な多レベル**DWT**も選択可。

**チャネル選択に Elastic Net。** 容積伝導により近接電極は強く相関します。純L1(Lasso)は相関クラスタから恣意的に1本だけ残し、その選択が再標本ごとに揺れます。L2項が Elastic Net の*グルーピング効果*を与え、容積伝導クラスタをまとめて採否します。さらに **stability selection**（ブートストラップ再学習で頻出特徴を残す）で包み、1回のノイジーな当て嵌めを再現性ある順位付けに変えます。資料の EN-CSP は32chから最も情報量の多い4〜8chを抽出（マクロF1≈0.889）しており、`max_channels` で同様の挙動を狙えます。

**MAEEG/GMAEEG は正直なフォールバック付き。** 本来の特徴抽出器は事前学習済みMAEEGを**凍結**して使います。Chien らに倣い `MAEEGEncoder` は6層の畳み込みフロントエンド（GroupNorm＋GELU＋Dropout）→64次元トークン→`model_dim=192`の8層Transformer→64次元コンテキスト埋め込みという構成で、`MaskedReconstructionPretrainer` はガウシアンノイズマスキング＋コサイン類似度再構成損失の目的関数を示します。`GMAEEGEncoder`（Fu ら）は**学習可能な動的隣接行列**を論文どおり `A = ReLU(tanh(W₂·ELU(W₁·Ã_init)))`（`Ã_init = E·Eᵀ`、自己ループ＋行正規化）で構成し、電極間にグラフ畳み込み `ELU(Â·X)` を適用します。これにより「どの電極間が強く結合しているか」という*結合トポロジー*自体が署名になります。事前学習重みを同梱しないため乱数初期化の Transformer は**本人識別性を持ちません**。そこでファクトリの既定は、デモデータ上で実際に被験者を分離できる `HandcraftedSpectralEncoder`（帯域パワー＋Hjorth＋スペクトラルエッジ）です。深層モジュールは依然として本物で実行可能（デモはフォワードとパラメータ数を出力）であり、`load_pretrained(path)` または `prefer="deep"` でスコアリング用エンコーダへ昇格します。学習による識別性を捏造せずアーキの正当性を保つ設計です。

**オープンセットに OC-SVM ⊕ LightGBM。** 1:1照合は登録時に見たことのない無限の他人を拒否せねばなりません。**One-Class SVM(SVDD)** は本人の埋め込み*のみ*で学習し、未知の攻撃者に対する誤受入率(FAR)を抑えます（オープンセット項）。**LightGBM** は本人 vs 背景コホートで学習し、*既知*の他人分布に対し境界を鋭くします（クローズドセット項）。両スコアは Platt キャリブレーション後に融合します。より厳格・低FARな代替として `and` ゲート（両方通過必須）も用意。デモは背景コホートを評価用 impostor と**分離**し、オープンセット主張を誠実に保ちます。

**ライブネスは ATAR の前。** ATAR はライブネスが依拠する瞬目/EOG をまさに除去するため、検知器は raw 信号をタップします。能動チャレンジは nonce とランダムな時間窓を持ち、瞬目の**有無・回数**、**窓内のタイミング**、**プロンプト前に瞬目が無いこと**を確認します。これらが静的リプレイやスプライス断片を拒否します。

**GAN は防御的デュアルユース。** `adversarial.py` は生成器（小型 `EEGGAN`、または NumPy の位相ランダム化 `SurrogateEEGGenerator` フォールバック）を2目的で提供します。(1) 少数の登録試行の**拡張**、(2) スプーフ EEG を合成してパイプラインが拒否できるかを確かめる**レッドチーム**。核心は、生成器は*安静時*EEG 統計は模倣できても、ランダムなチャレンジに同期した*正当な瞬目*は生成できないため、ライブネスがスペクトル的リアルさに関係なくスプーフを拒否する点です（デモシナリオ S5）。本モジュールは信号を生成し自前のパイプラインに対してスコアリングするだけで、注入機能を持たず、実在個人を標的にしません。

## 依存関係 / フォールバック表

NumPy だけで全て動作します。各オプションは段を強化します:

- **SciPy** → Butterworth/Welch/`find_peaks`。無ければ FFT/ピリオドグラム/局所最大。
- **PyWavelets** → 本来の ATAR（WPD/DWT）。無ければロバスト振幅減衰。
- **scikit-learn** → Elastic-Net ロジスティック＋One-Class SVM＋スケーリング。無ければ NumPy ロジスティック＋マハラノビス一クラス。
- **LightGBM** → ブースティング枝。無ければ勾配ブースティング/ロジスティック回帰。
- **PyTorch** → MAEEG/GMAEEG エンコーダ＋EEG-GAN。無ければ手作りエンコーダ＋位相ランダム化サロゲート。
- **MNE** → PhysioNet EEGBCI データ。無ければ合成波形。

## 実行

```bash
pip install -r requirements.txt          # 最小なら `pip install numpy` のみでも可
python -m eeg_biometric.pipeline         # 親ディレクトリから
# または:  cd eeg_biometric && python pipeline.py
```

デモは有効なバックエンドを表示し、被験者 `S001` を登録した後、5つのシナリオを実行します — 本人＋瞬目（受理）、他人＋瞬目（本人性で拒否）、本人リプレイ瞬目なし（ライブネスで拒否）、本人だが瞬目ミスタイミング（ライブネスで拒否）、GAN/サロゲートのスプーフ瞬目なし（ライブネスで拒否） — 続いて生体経路の FAR/FRR/ACC を表示します。登録時の GAN 拡張は `PipelineConfig(use_gan_augmentation=True)` で有効化できます。

## 主な設定（`PipelineConfig`）

- `data_source`：`"auto"`（MNE→合成自動フォールバック）/ `"synthetic"` / `"mne"`
- `atar_decomposition`：`"wpd"`（既定）/ `"dwt"`、`atar_beta`：動作点（0=穏やか〜1=積極的）
- `max_channels`：選択チャネル上限（資料の4〜8chに対応）
- `encoder_prefer`：`"auto"`（手作り）/ `"deep"`・`"gmaeeg"`（深層）、`pretrained_path`：重み読込
- `nu` / `fusion_weight` / `recognizer_mode`（`"fusion"`/`"and"`）/ `target_far`
- `use_gan_augmentation` / `gan_backend`（`"surrogate"`/`"gan"`/`"auto"`）

## 制限と今後

これは骨組みです。MAEEG をスコアリング用にするには実際の事前学習重みが必要で、しきい値や `nu`/`beta` は実データでの調整が前提です。ライブネスの `require_clean_pre_prompt` は自発的な瞬目が含まれるデータでは緩和が必要な場合があります。合成生成器は実在の被験者間EEG変動のモデルではなく代替物です。

資料準拠で**任意に有効化できる細部**として、(1) 資料の「2回瞬き」例に合わせた `make_challenge(n_blinks=2)`、(2) Phase 1 のエッジ前処理順序に合わせた **0.5–45Hz 帯域通過 → ATAR** の前段バンドパス、が挙げられます。自然な拡張：テンプレートの経年・再登録、ホールドアウトコホートでのスコア融合キャリブレーション、マルチセッション評価、ESP32 収集経路に合わせたストリーミング（ブロック単位）ATAR＋ライブネスのフロントエンド。

**スコープ外**：Plan A/Phase 1 のファームウェア層（ESP32-S3/ADS1299 の C++ 実装、WireGuard/TLS 転送）。本プロトタイプはサーバ側 Python 推論に限定しています。
