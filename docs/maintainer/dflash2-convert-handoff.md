# 转模型：产出带 DFlash2 草稿的 artifact（在有源的机器上执行）

这份文档只讲**转模型**这一件事。分支、运行时改动、基准测试都不在这里。

## 目标

在有 Qwen3.8-27B GGUF 源和 HF 访问的机器上，把官方 DFlash2 草稿包转成一个
**原生 v2 容器**，容器里带 DFlash2 草稿（66 个对象），供 `feat/mmflash-dflash1`
分支的引擎以 `--spec dflash --draft-tokens 7` 加载。

## 为什么是 DFlash2 而不是 MMFlash

MMFlash 的移植卡在一个未定位的特征捕获问题上（见文末），而 DFlash2 是**这个引擎
原生支持、官方发布、在同样硬件上实测过**的宽块草稿：

- 引擎的 reader 已实现 `components/dflash2` 的校验与映射（`src/artifact/reader.cpp`）
- 引擎的运行时已实现 DFlash2 的动态卷积与候选选择器
- 上游在 **2 × V100-SXM2 16 GB** 上实测：DFlash7 默认 **122.65 tok/s**，
  开启两个实验档位 **168.15 tok/s**（85K 输入的 fixture）

要对比的基线（这台机器，2026-11-26 实测）：

| 配置 | 验收率 | decode |
| --- | --- | --- |
| 无投机 | — | 51.3 tok/s |
| MTP draft=4, 256K | 2.56 tok/round (39.0%) | **104–107 tok/s** |
| MTP draft=5, 256K | 2.63 tok/round (32.6%) | 95–102 tok/s |

MTP 的草稿宽度上限是 5（`kMaximumMtpDraftTokens`），加宽反而更慢，所以宽块这条
路只能靠 DFlash2 走。

## 代码位置

仓库：`chenjxdev/ninfer-v100xN`，分支 **`feat/mmflash-dflash1`**。

DFlash2 的转换器在这个分支上**已经可用**，不需要任何改动：
`tools/convert/qwen3_8_27b/convert_gguf.py` 的 `--dflash2-model` 参数。

## 前置：输入

### 1. 主源 GGUF（必须）

- 仓库：`lmstudio-community/Qwen3.8-27B-GGUF`
- 需要两个文件：

  | 文件 | 大小 | 说明 |
  | --- | ---: | --- |
  | `Qwen3.8-27B-Q4_K_M.gguf` | 16,810,714,336 | 目标模型，转换器只认这个量化档 |
  | `mmproj-Qwen3.8-27B-BF16.gguf` | 931,145,856 | 视觉塔，转换器要求它存在 |

  转换器会严格校验源的清单：`general.architecture` 为 `qwen35`、`block_count` 65、
  `embedding_length` 5120、`file_type` 15，张量类型计数必须恰好是
  `{0: 360, 12: 439, 14: 67}`，视觉 GGUF 必须有 334 个张量。**换成别的量化档会被拒。**

  下载（服务器连 `huggingface.co` 不通，走镜像站）：

  ```bash
  SRC=<SRC_DIR>
  mkdir -p "$SRC"
  BASE="https://hf-mirror.com/lmstudio-community/Qwen3.8-27B-GGUF/resolve/main"
  curl -L --retry 8 -C - -o "$SRC/Qwen3.8-27B-Q4_K_M.gguf"      "$BASE/Qwen3.8-27B-Q4_K_M.gguf"
  curl -L --retry 8 -C - -o "$SRC/mmproj-Qwen3.8-27B-BF16.gguf"  "$BASE/mmproj-Qwen3.8-27B-BF16.gguf"
  ```

### 2. DFlash2 草稿包（必须）

- 仓库：`incoai/Qwen3.8-27B-DFlash2`
- revision：`main`（该仓库只有一个提交，`lastModified` 2026-09-17）
- 文件：

  | 文件 | 大小 |
  | --- | ---: |
  | `config.json` | 1,239 |
  | `model.safetensors` | 3,848,817,896 |

  ```bash
  DF2=<DFLASH2_DIR>
  mkdir -p "$DF2"
  BASE="https://hf-mirror.com/incoai/Qwen3.8-27B-DFlash2/resolve/main"
  curl -L --retry 8 -C - -o "$DF2/config.json"          "$BASE/config.json"
  curl -L --retry 8 -C - -o "$DF2/model.safetensors"    "$BASE/model.safetensors"
  ```

  这个包的 `config.json` 声明了转换器会校验的每一项：

  ```json
  "dflash_config": {
    "block_size": 8,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "mask_token_id": 248070,
    "selector_rank": 256,
    "selector_top_k": 16,
    "target_layer_ids": [5, 19, 33, 47, 61]
  }
  ```

  `block_size` 是 8，所以草稿宽度是 **7**（`--draft-tokens 7`）。

## 执行转换

```bash
cd /home/chenjx/ninfer-mmflash
PY=/home/chenxin/… # 用带 torch 的解释器，见下
```

用哪个 python 取决于那台机器；这台服务器上带 torch 的是
`/home/chenjx/ninfer-conv-venv/bin/python3`（torch 2.15 CPU 版），转换只需要 CPU。

```bash
$PY -m tools.convert.qwen3_8_27b.convert_gguf \
    --model  "$SRC/Qwen3.8-27B-Q4_K_M.gguf" \
    --mmproj "$SRC/mmproj-Qwen3.8-27B-BF16.gguf" \
    --dflash2-model "$DF2" \
    --out /data/models/qwen3_8_27b_q4km_dflash2.ninfer
```

`--ranking` 用仓库自带的默认值即可
（`tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64`）。

## 验证产物

```bash
OUT=/data/models/qwen3_8_27b_q4km_dflash2.ninfer

# 1) 容器是原生 v2：magic 为 'NINFER' + 00 + 02
od -An -tx1 -N8 "$OUT"
# 期望: 4e 49 4e 46 45 52 00 02

# 2) 草稿对象齐了：66 个 dflash/ 对象
$PY - <<'EOF'
import json, struct
p = "/data/models/qwen3_8_27b_q4km_dflash2.ninfer"
with open(p, "rb") as f:
    magic, length = struct.unpack("<8sQ", f.read(16))
    d = json.loads(f.read(length).decode("utf-8"))
names = [o.get("id") or o.get("name") for o in d["objects"]]
df = [n for n in names if n.startswith("dflash/")]
print("objects:", len(names), "dflash objects:", len(df))
print("the selector is present:", any("candidate_selector" in n for n in df))
print("the convolution is present:", any("conv" in n for n in df))
EOF
```

期望：`dflash objects: 66`，选择器和卷积都在。

## 拷回

产物大约 **20 GB**（Q4_K_M 目标 + BF16 草稿）。拷回 `172.16.3.88` 的 `/data/models/`。

拷回后这样跑：

```bash
export LD_LIBRARY_PATH=/home/chenjx/ninfer-jack/pkg/lib
/home/chenjx/ninfer-mmflash/build-v100/apps/ninfer-serve \
  /data/models/qwen3_8_27b_q4km_dflash2.ninfer \
  --host 172.16.3.88 --port 18099 \
  --tp 2 --devices 1,0 --max-concurrency 1 \
  --max-context 131072 --kv-dtype int8 \
  --spec dflash --draft-tokens 7
```

### 上下文该给多少

**先给 128K**。这台机器上（2 × 16 GB）：

| | 每卡占用 |
| --- | ---: |
| 目标模型（Q4_K_M 路径更小，但草稿是 BF16 且在两张卡上各存一份） | ~10.8 GB |
| DFlash2 草稿（BF16，复制到每张卡） | ~3.6 GB |
| 文本 KV（INT8，128K） | ~3.6 GB |
| 草稿上下文 cache | ~0.2 GB |
| 工作区与图 | ~0.3 GB |

128K 大约用掉 18.5 GB，能放下；256K 的文本 KV 要 6.3 GB，装不下。

注意 DFlash2 与 Vision 在这个引擎里**互斥**（`--spec dflash` 和 `--vision` 不能同时开），
所以 DFlash2 这一档没有视觉塔。

## 与 MTP 的对比口径

用同一台机器、同一个 prompt 长度、同一个 `max_tokens` 测。参考基线是
MTP draft=4、128K 上下文：**101.8 tok/s，2.63 tok/round（32.6%）**。

上游 DFlash2 在同样硬件、85K 输入的 fixture 上是 122.65 tok/s，所以这一档
**有可能超过 MTP**——但要在这台机器上实测才能确认。

## 附：MMFlash 那条路的状态

MMFlash（16 宽块的第一代 DFlash）的移植停在一个未定位的问题上：

- artifact 已修好（原本整个 payload 错位 413,696 字节，草稿对象内容全错）
- 草稿权重与 checkpoint 校验一致（W8 误差 0.0039）
- 草稿前向已实现：每层独立的上下文 cache、无卷积、用目标 output_head 的 argmax 出 token
- **验收率仍只有 5.3%**（对比 MTP 的 39%）

已用逐环节数值对照定位到：**decode 阶段捕获的 5 个目标层特征里，只有第 1 个有数据，
其余 4 个是零**。层号捕获正确（层 1/16/31/46/61 各自填各自的槽位）、压缩算子、
scatter 的寻址与步长逐条核对过都正确，但数据在写完到读出之间消失。参考实现的
对照脚本在 `/home/chenjx/mmflash-work/`，引擎的 trace 用 `NINFER_MMFLASH_TRACE` 打开。

MFFlash 这条路要接着做，先看 `/home/chenjx/mmflash-work/compare4.py` 和
`find_taps.py` 的对照方法；DFlash2 则是可以直接用的。
