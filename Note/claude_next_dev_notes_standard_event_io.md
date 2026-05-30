# Claude向け開発メモ: 標準化Event入出力アーキテクチャ

作成日: 2026-05-30

## 目的

このモデルの設計方針は、入力元の種類に縛られない共通形式で情報を扱うことである。

テキスト、視覚、聴覚、触覚、温度、時間などを、それぞれ専用の生データ形式のままモデルへ渡すのではなく、いったん標準化されたEvent形式へ変換する。

同様に、モデルの出力も標準化されたEvent形式にし、各出力装置側がそのEvent形式を自分の出力形式へ変換する。

目指す流れ:

```text
Sensor / Input Device
  -> modality adapter
  -> standardized EventSeq
  -> model
  -> standardized EventSeq
  -> output adapter
  -> Output Device
```

モデル本体は、可能な限り「これはテキスト」「これは画像」「これは温度」といった生の形式に依存しない。

## 現在の到達点

現在の実装では、`src/event.h` / `src/event.c` により、標準化Event列の入口が作られている。

現在あるもの:

- `EventSeq`
- `Modality`
- `event_seq_new`
- `event_seq_del`
- `event_seq_clear`
- `event_append_text`
- `event_append_scalar`

これは良い方向である。

ただし、現状のモデル本体はまだ基本的に以下の形である。

```text
int token id sequence
  -> token embedding
  -> Transformer
  -> token logits
```

つまり、EventSeqの箱はでき始めているが、まだEventSeqがTransformerの主入力・主出力になっていない。

## 次に実現したい中核

次に作るべき中核は以下の3つ。

1. `EventSeq -> D次元ベクトル列` へのevent embedding
2. `Transformer出力 -> EventSeq` へのprediction head
3. 入力adapter / 出力adapterのinterface

これにより、以下の流れへ移行する。

```text
EventSeq
  -> event embedding
  -> Transformer
  -> event prediction head
  -> EventSeq
```

## 推奨する段階的実装

### Step 1: EventSeqの安全性を上げる

現在の `EventSeq` は容量チェックがないため、append時に範囲外書き込みが起きる可能性がある。

まず以下を実装する。

- `EventSeq` に `capacity` を追加する。
- `event_append_text` / `event_append_scalar` の前に容量チェックを入れる。
- 超過時はエラーを返すか、自動拡張する。

推奨API:

```c
typedef struct {
    int   *modality;
    int   *channel;
    int   *time_index;
    float *value;
    int   *token_id;
    int    n;
    int    capacity;
} EventSeq;
```

可能ならappend関数は `void` ではなく `int` を返す。

```c
int event_append_text(EventSeq *s, const int *ids, int len, int start_time);
int event_append_scalar(EventSeq *s, const float *vals, int len,
                        int modality, int channel, int start_time);
```

### Step 2: Event embeddingを作る

EventをTransformerの `D` 次元へ変換する層を追加する。

初期案:

```text
event_embedding =
    modality_embedding
  + channel_embedding
  + time_embedding
  + token_embedding
  + value_projection
```

まずは単純に以下でよい。

- text event:
  - `token_embedding[token_id]`
  - `modality_embedding[MOD_TEXT]`
  - `time/position embedding`

- scalar event:
  - `value * learned_value_vector`
  - `modality_embedding[MOD_TEMPERATURE]` など
  - `channel_embedding[channel]`
  - `time/position embedding`

候補API:

```c
void event_embed_fwd(Model *m, const EventSeq *ev, Mat *out);
```

`out` は `(ev->n x D)`。

### Step 3: 最初は離散token化で進める

連続値をいきなり回帰で扱うと、loss設計と出力headが複雑になる。

最初は、温度や触覚などのscalar値もbin化して、離散tokenとして扱うのがよい。

例:

```text
TEMP: 0.42 -> TEMP_BIN_42
TOUCH_PRESSURE: 0.81 -> TOUCH_BIN_81
TIME: 0.15 -> TIME_BIN_15
```

この方式なら、現在の「次token予測」に近いまま進められる。

後で拡張する場合:

```text
event type: classification
channel   : classification
value     : regression or quantized classification
confidence: regression
```

### Step 4: Event prediction headを作る

モデル出力も標準化Event形式に戻す。

初期段階では、すべてを離散tokenとして扱い、既存の `proj` / `proj_b` に近い出力でよい。

将来的には、複数headに分ける。

```text
hidden
  -> modality logits
  -> channel logits
  -> token/value-bin logits
  -> optional continuous value
```

候補API:

```c
void event_predict_fwd(Model *m, const Mat *hidden, Mat *event_logits);
```

または、最初は既存 `logits` をEvent vocabulary logitsとして再利用してもよい。

### Step 5: Adapter interfaceを分離する

各センサーや出力装置は、標準Event形式との相互変換だけを担当する。

入力adapter:

```text
raw text / sensor / time
  -> EventSeq
```

出力adapter:

```text
EventSeq
  -> text / actuator / display / audio / etc.
```

モデル本体には、センサー固有処理を入れない。

## 注意点

### 画像や音声をそのままEvent化しすぎない

画像の全pixelや音声waveformの全sampleをそのままEventSeqにすると、系列長が非常に長くなる。

最初からraw dataを直接入れるのではなく、以下のような中間表現をadapter側で作るのが現実的。

画像:

```text
patch / edge / feature / object / region
```

音声:

```text
frame / spectrogram bin / feature
```

触覚:

```text
pressure / location / duration
```

温度:

```text
normalized scalar / quantized bin
```

### まずは text + scalar sensor の小さな実験から始める

推奨する最初のデモ:

```text
[TEXT: BOS]
[TEMP_BIN: 42]
[TIME_BIN: 15]
[TEXT: token_7]
[TOUCH_BIN: 81]
```

これをEvent vocabularyのtoken列として学習し、次Eventを予測する。

## レビュー時に見てほしい観点

次回レビューでは、特に以下を確認する。

1. `EventSeq` が容量安全になっているか。
2. Event形式がモデル本体に過度に漏れていないか。
3. adapter層とmodel層の責務が分かれているか。
4. Event embeddingが `D` 次元へ一貫して変換されているか。
5. 出力も標準Event形式に戻す方向になっているか。
6. text専用実装へ逆戻りしていないか。
7. KV cache実装とEvent入力設計が衝突していないか。
8. ビルドと既存sequence copy taskが壊れていないか。

## まとめ

今の方向性は実現可能である。

ただし現在は、標準化Event入力構想の足場ができた段階であり、まだモデル本体はEventSeqを主入力・主出力として扱っていない。

次の開発では、以下を優先する。

```text
EventSeq safety
  -> event embedding
  -> event prediction head
  -> adapter interface
```

この順序で進めると、テキスト、視覚、聴覚、触覚、温度、時間を同じ標準表現に変換して扱う設計へ近づける。

## 2026-05-30 追加レビュー結果

Claudeによる追加実装後、以下を確認した。

追加された主な要素:

- `EventSeq.capacity`
- `EventEmbed`
- `EventHead`
- `TextAdapter`
- `ScalarBinAdapter`
- `event_task.c` による text + temperature の混合Eventデモ
- `test.c` による基本テスト

検証結果:

```text
mingw32-make test
-> 42 passed, 0 failed

mingw32-make
.\transformer.exe
-> LASTEXITCODE=0
```

通常のsequence copy task、KV cache decode、EventEmbed demoはいずれも実行完了した。

ただし、設計上の重要な修正点が残っている。

### 最優先: Eventの次要素予測が未来を見ている

現在の `event_task.c` は、次Event予測の形に見えるが、実際には未来情報を見られる構造になっている。

問題の構造:

```text
EventSeq全体
  -> EventEmbed
  -> Encoder self-attention causal=0
  -> EventHead
  -> 各位置で次Event labelを予測
```

`encoder.c` の `el_fwd` は現在以下を呼ぶ。

```c
attn_fwd(&xi_s, &xi_s, &w->sa, cfg->H, 0, &c->sa);
```

最後の `0` はcausal maskなしを意味する。

そのため、位置 `t` のhidden stateが、正解である位置 `t+1` のEventを見ることができる。

これはnext-event predictionとしては情報リークである。

#### 修正候補

いずれかを選ぶ。

1. Event用にcausal self-attentionを持つEncoder相当のlayerを作る。
2. Decoder-only TransformerとしてEvent列を扱う。
3. 学習時にprefixのみを入力し、次Eventを1つだけ予測する。
4. 既存Decoderを流用し、EventSeqをtarget列としてcausal self-attentionで扱う。

短期的には、まず以下が最も単純。

```text
EventSeq prefix
  -> causal self-attention
  -> last hidden
  -> EventHead
  -> next Event label
```

または、既存のEncoder layerに無理に混ぜず、Event専用の `event_lm_*` 実装を分けるとよい。

### High: KV cacheの境界チェックがまだ未対応

`model_decode_step` では、以下の範囲チェックが必要。

- `pos >= 0`
- `pos < m->c.T`
- `token >= 0`
- `token < m->c.V`
- `lkv->self.len < kv->max_len`
- `lkv->cross.len > 0`

現状のままだと、長い生成やAPI誤用で `pos`, `K`, `V` が範囲外参照になる。

推奨:

```c
int model_decode_step(...);
```

のように戻り値を `int` にし、成功時 `0`、失敗時 `-1` を返す。

### High: EventEmbedがEvent値を信頼しすぎている

`event_embed_fwd` / `event_embed_bwd` は、以下をそのまま配列offsetに使っている。

- `modality`
- `channel`
- `time_index`
- `token_id`

標準化Eventを公開形式にするなら、adapter外から壊れたEventSeqが来ても即メモリ破壊しないようにしたい。

最低限のチェック:

```text
0 <= modality < EVENT_MAX_MOD
0 <= channel < EVENT_MAX_CHAN
0 <= time_index < max_time
token_id == -1 または 0 <= token_id < V
```

対応案:

- `event_seq_validate(const EventSeq *s, const EventEmbed *e)` を追加する。
- `event_embed_fwd` の先頭でvalidateする。
- または、debug buildでは `assert`、release buildではエラーコードを返す。

可能なら `event_embed_fwd` も `void` ではなく `int` を返す。

```c
int event_embed_fwd(const EventSeq *s, const EventEmbed *e, Mat *out);
```

### Medium: ScalarBinAdapterの `n_bins <= 1` が未防御

`sba_encode`, `sba_decode`, `sba_label` は `a->n_bins - 1` で割る。

`n_bins <= 1` のadapterが渡ると壊れる。

最低限:

```c
if (a->n_bins < 2) return error;
```

を入れる。

### Medium: Makefile cleanとNULファイル問題が残っている

`Makefile` はまだ以下の形式。

```make
del /Q $(TARGET).exe 2>NUL || true
```

この環境では過去にプロジェクト直下へ実ファイル `NUL` が作られている。

MSYS系makeで進めるなら、以下へ戻すほうが安全。

```make
clean:
	rm -f $(TARGET) $(TARGET).exe $(TEST_TARGET) $(TEST_TARGET).exe
```

Windows `cmd` 前提にするなら、明示的に `cmd /C` を使う。

```make
clean:
	cmd /C "del /Q $(TARGET).exe 2>NUL"
```

どちらかに統一すること。

### 良かった点

- `EventSeq.capacity` によって前回レビューの容量問題は改善された。
- `EventEmbed` と `EventHead` が入り、EventSeqをモデル入出力へつなぐ骨格ができた。
- adapter層が `TextAdapter` / `ScalarBinAdapter` として分離されている。
- `make test` が追加され、基本的な行列演算・LayerNorm・Attention・EventEmbedの検証ができる。
- sequence copyの既存動作とKV cache decodeは維持されている。

### 次の作業優先順位

1. Event next predictionの未来情報リークを直す。
2. `model_decode_step` に境界チェックを入れ、可能なら `int` 戻り値にする。
3. `event_embed_fwd/bwd` の入力validateを追加する。
4. `ScalarBinAdapter` の `n_bins >= 2` を保証する。
5. `Makefile clean` と `NUL` ファイル問題を解消する。
6. Eventタスクに「未来を見ないこと」を確認するテストを追加する。
7. EventHead出力をEventSeqへ復元する関数を追加する。

### 次回レビューで特に見ること

- Eventモデルが本当にcausal / prefix-onlyになっているか。
- EventSeq標準形式が入力と出力の両方で使われているか。
- adapter層がモデル本体から独立しているか。
- 既存のsequence copy taskとKV cache decodeが壊れていないか。
- `make test` が失敗せず、Event関連テストが増えているか。
