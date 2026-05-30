# Review: KV cache / EventSeq実装

作成日: 2026-05-30

## 対象

今回のレビューでは、以下の追加・変更を中心に確認した。

- `src/inference.c`
  - `DecodeCache`
  - `model_encode`
  - `decode_cache_precompute_cross`
  - `model_decode_step`
- `src/event.h`
- `src/event.c`
- `src/main.c`
  - baseline greedy decode
  - KV cache greedy decode
  - 出力一致確認
- `Makefile`
  - `inference.c` / `event.c` の追加
  - `clean` ターゲット変更

## 検証結果

以下のコマンドでビルドと実行を確認した。

```powershell
mingw32-make
.\transformer.exe
```

結果:

```text
KV cache match: OK
LASTEXITCODE=0
```

今回の実行では、baseline decodeとKV cache decodeの生成結果は一致した。

## Findings

### High: EventSeq appendに容量チェックがない

対象:

- `src/event.c`
  - `event_append_text`
  - `event_append_scalar`

現状、`EventSeq` は確保時に `capacity` を受け取るが、構造体内に `capacity` を保持していない。

そのため、以下のようなappendで確保済み配列の範囲外へ書き込む可能性がある。

```c
int k = s->n + i;
s->modality[k] = MOD_TEXT;
```

リスク:

- メモリ破壊
- 不定動作
- 学習や推論中のクラッシュ
- 壊れたデータによる原因特定しにくい誤動作

推奨対応:

1. `EventSeq` に `capacity` を追加する。
2. append前に `s->n + len <= s->capacity` を確認する。
3. 超過時はエラー終了、または自動拡張する。

例:

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

### High: DecodeCacheに上限チェックがない

対象:

- `src/inference.c`
  - `model_decode_step`

現状、KV cacheに新しいtokenのK/Vを追加する際、`self.len` が `max_len` を超えないか確認していない。

該当箇所:

```c
Mat Kr = {lkv->self.K.d + len*D, 1, D};
Mat Vr = {lkv->self.V.d + len*D, 1, D};
```

`len >= cfg->T` の状態で呼ばれると、`K/V` の範囲外へ書き込む。

また、`pos >= cfg->T` の場合も以下が範囲外参照になる。

```c
float *penc = m->pos + pos * D;
```

リスク:

- メモリ破壊
- 長い生成時のクラッシュ
- 入力長が変化したときの不定動作

推奨対応:

1. `model_decode_step` の先頭で `pos < m->c.T` を確認する。
2. layerごとに `lkv->self.len < kv->max_len` を確認する。
3. APIの戻り値を `int` にして、成功/失敗を返せるようにする。

例:

```c
int model_decode_step(...);
```

### High: `attn_1q` が `len == 0` に対応していない

対象:

- `src/inference.c`
  - `attn_1q`

現状、`len == 0` の場合でも以下を読む。

```c
float mx = scores[0];
```

現在の `main.c` では、self-attentionは現在tokenのK/Vを追加してから呼び、cross-attentionは事前計算後に呼ぶため、通常は `len > 0` になる。

ただし、関数単体としては危険である。

リスク:

- 空のsource sequence
- cross cache未初期化
- API誤用時のクラッシュ

推奨対応:

```c
if (len <= 0) {
    memset(out, 0, (size_t)D * sizeof(float));
    return;
}
```

または、呼び出し側でエラーにする。

### Medium: `Makefile clean` が `NUL` ファイルを生成する可能性

対象:

- `Makefile`

現状:

```make
clean:
	del /Q $(TARGET).exe 2>NUL || true
	del /Q $(TARGET) 2>NUL || true
```

プロジェクト直下に実ファイル `NUL` が残っている。

```text
NUL 44 bytes
```

これはWindows予約デバイス名と衝突し、PowerShellから通常の `Get-Item .\NUL` で扱えない場合がある。

推定原因:

- `mingw32-make` の実行環境で `2>NUL` がWindowsデバイスではなく通常ファイルとして解釈された可能性がある。

推奨対応:

MSYS系makeで動かすなら、`rm` を使う。

```make
clean:
	rm -f $(TARGET) $(TARGET).exe
```

PowerShell/cmd前提にするなら、実行シェルを明示する。

```make
clean:
	cmd /C "del /Q $(TARGET).exe 2>NUL"
	cmd /C "del /Q $(TARGET) 2>NUL"
```

## 良かった点

- 学習用forward/backwardを壊さず、推論用に `model_encode` / `model_decode_step` を分離している。
- `DecodeCache` を学習用 `AC` / `EC` / `DC` と混ぜていないため、責務が分かりやすい。
- baseline decodeとKV cache decodeの結果一致確認が `main.c` に入っている。
- `EventSeq` はparallel arrays形式で、C実装として扱いやすい方向になっている。

## 次の推奨作業

優先順:

1. `EventSeq` に `capacity` を追加し、append時の範囲チェックを入れる。
2. `model_decode_step` に `pos` と `self.len` の上限チェックを入れる。
3. `attn_1q` の `len <= 0` を防御する。
4. `Makefile clean` をMSYSまたはcmdのどちらかに寄せる。
5. `NUL` ファイルを削除する。
6. KV cache一致確認を固定seed・複数サンプルで回す。
7. 可能なら `model_decode_step` を `int` 戻り値にして、APIとして失敗を返せるようにする。

## 補足

現在のKV cache実装は、短いsequence copy taskでは速度差が目立ちにくい。

ただし設計としては、将来の長い系列生成や会話形式の推論に必要な方向であり、今の段階で入れる価値は高い。
