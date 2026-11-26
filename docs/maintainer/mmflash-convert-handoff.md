# 转模型：在有源的机器上产出带 MMFlash 草稿的 v3 artifact

这份文档只讲**转模型**这一件事。分支、运行时改动、基准测试都不在这里。

## 目标

在有 QAD 源 checkpoint 的机器上，把 `Qwen3.8-27B-NVFP4-QAD` 转成一个 **v3 容器**，
容器里带 `components/mmflash` 草稿包（43 个对象，W8 + BF16，1.71 GiB），
供 `feat/mmflash-dflash1` 分支的引擎加载。

## 代码位置

仓库：`chenjxdev/ninfer-v100xN`，分支 **`feat/mmflash-dflash1`**。

关键提交：

| 提交 | 内容 |
| --- | --- |
| `1dd0b47` | `feat(convert): carry the MMFlash draft package in a v3 artifact` |
| `dfc21f6` | `feat(engine): run the first-generation DFlash draft (MMFlash) alongside DFlash2` |

在服务器上的 worktree 是 `/home/chenjx/ninfer-mmflash`。

## 前置：输入

### 1. 主源 checkpoint（必须）

- 仓库：`nvidia/Qwen3.8-27B-NVFP4-QAD`
- revision：`modelopt-0.46.0rc1`
- 落盘目录（下文用 `<QAD_DIR>` 表示），必须是完整的 HF checkpoint 目录：

  ```
  <QAD_DIR>/
    model.safetensors.index.json      # weight_map 覆盖的 shard 集合要与目录里的 *.safetensors 完全一致
    *.safetensors                     # 所有 shard，缺一不可
    config.json
    hf_quant_config.json              # producer.name 必须是 "modelopt"
    tokenizer.json
    tokenizer_config.json
    chat_template.jinja
    generation_config.json
    preprocessor_config.json
    video_preprocessor_config.json
  ```

  转换器会校验：index 的 `weight_map` 非空、shard 集合与目录一致、每个 shard 非空、
  `hf_quant_config.json` 的 producer 是 modelopt、六个前端资源齐全且名字完全匹配。
  少任何一个都会在 preflight 阶段报错。

### 2. MTP 辅助 checkpoint（QAD profile 必需）

- 路径：`/data/models/KatterMobile/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-NVFP4-QAT`
- 用途：QAD 导出缺少四个 MTP 张量（`mtp.fc.weight`、`mtp.pre_fc_norm_embedding.weight`、
  `mtp.pre_fc_norm_hidden.weight`、`mtp.norm.weight`），从这份辅助 checkpoint 原样读取。
- 如果目标机器用的是 **daisyrose** profile，则不需要它（该 profile 自带 MTP 层）。

### 3. MMFlash 草稿 checkpoint（必须）

- 仓库：`FSCCS/Qwen3.8-27B-MMFlash`
- revision：`4b28ae0bc95ceed328bbceec74a30481f2b86b4b`
- 落盘目录（下文用 `<MMFLASH_DIR>`）：

  ```
  <MMFLASH_DIR>/
    config.json
    model.safetensors        # 3,460,432,504 字节
  ```

- 校验值：
  - `sha256 = 4f04793676efd7624465101a4498244fa2eac7563017f35cbff3dd451dd0a78d`
  - 58 个张量，1,730,213,120 参数，全部 BF16

  下载（服务器连 `huggingface.co` 不通，走镜像站）：

  ```bash
  DIR=<MMFLASH_DIR>
  REV=4b28ae0bc95ceed328bbceec74a30481f2b86b4b
  BASE="https://hf-mirror.com/FSCCS/Qwen3.8-27B-MMFlash/resolve/$REV"
  mkdir -p "$DIR"
  curl -sS -L --retry 8 -C - -o "$DIR/config.json"   "$BASE/config.json"
  curl -sS -L --retry 8 -C - -o "$DIR/model.safetensors" "$BASE/model.safetensors"
  sha256sum "$DIR/model.safetensors"
  # 必须是 4f04793676efd7624465101a4498244fa2eac7563017f35cbff3dd451dd0a78d
  stat -c '%s' "$DIR/model.safetensors"   # 必须是 3460432504
  ```

## 两处必须改的配置

### 1. 让 profile 产出 v3

`tools/convert/qwen3_8_27b/source_profile.py` 里 `SourceProfile.container_version`
默认是 `"v2"`。v3 才带 `bindings` / `uses` / `components`，`components/mmflash` 才有地方放。

给要用到的 profile 加上 `container_version="v3"`：

```python
QAD = SourceProfile(
    name="qad",
    ...
    container_version="v3",
)
```

### 2. 输出文件名

转换器有硬校验：`output.name` 必须等于 profile 的 `OUTPUT_BASENAME`
（`tools/convert/qwen3_8_27b/convert_nvfp4.py`）。默认是：

| profile | OUTPUT_BASENAME |
| --- | --- |
| qad | `qwen3_8_27b_q4km_v2.ninfer` |
| daisyrose | `daisyrose_27b_nvfp4_q6emb_q6head_q4mtp.ninfer` |

名字里带 `_v2` 是**旧产物遗留的命名**，和容器版本无关；用默认名即可，不必改。
如果想换个名字，就同步改 `OUTPUT_BASENAMES` 里对应的那一项。

## 执行转换

```bash
cd /home/chenjx/ninfer-mmflash

# 需要带 torch 的解释器。这台机器上是：
PY=/home/chenjx/ninfer-conv-venv/bin/python3

# 用哪个源 profile，由 NINFER_NVFP4_SOURCE 在导入前决定
export NINFER_NVFP4_SOURCE=qad        # 或 daisyrose

$PY -m tools.convert.qwen3_8_27b.convert_nvfp4 \
    --model <QAD_DIR> \
    --out /data/models/qwen3_8_27b_q4km_v2.ninfer \
    --mtp-source /data/models/KatterMobile/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-NVFP4-QAT \
    --mmflash-model <MMFLASH_DIR> \
    --device cpu
```

几点说明：

- **`--device cpu` 是安全的**。W8 量化在 CPU 上会正确回退（`pick_device` 检测到
  CUDA 不可用就退回 CPU），只是慢一些。量化只做一次，用 CPU 避免占用别人的 GPU。
- **`--mtp-source` 只有 QAD profile 需要**；daisyrose profile 传默认即可。
- **`--mmflash-model` 是新增参数**，指向 MMFlash checkpoint 目录。
- 输出目录要有约 20 GB 空间。

## 验证产物

转换完成后检查：

```bash
OUT=/data/models/qwen3_8_27b_q4km_v2.ninfer

# 1) 容器版本必须是 v3
od -An -tx1 -N8 "$OUT"
# 期望: 4e 49 4e 46 45 52 00 03   ('NINFER' + 00 + 03)

# 2) 目录里必须有 components/mmflash，且带 config
$PY - <<'EOF'
import json, struct
p = "/data/models/qwen3_8_27b_q4km_v2.ninfer"
with open(p, "rb") as f:
    magic, dlen = struct.unpack("<8sQ", f.read(16))
    d = json.loads(f.read(dlen).decode("utf-8"))
print("magic:", magic)
print("root members:", sorted(d.keys()))
comp = d["components"]
print("components:", sorted(comp.keys()))
cfg = comp["mmflash"]["config"]
print("taps   :", cfg["dflash_config"]["target_layer_ids"])   # [1, 16, 31, 46, 61]
print("layers :", cfg["layer_types"])                          # 5 x full_attention
print("block  :", cfg["block_size"])                           # 16
print("objects:", len(d["objects"]))
dflash = [o["name"] for o in d["objects"] if o["name"].startswith("dflash/")]
print("dflash objects:", len(dflash))                          # 43
EOF
```

期望：

- magic 结尾是 `03`（v3）
- 根成员含 `components`、`bindings`、`uses`
- `components` 含 `mmflash`，其 `config` 的目标层是 `[1, 16, 31, 46, 61]`，
  `layer_types` 是 5 个 `full_attention`，`block_size` 是 16
- `dflash/` 开头的对象恰好 43 个
- **对象总数是 1259**。这个数字是 reader 的硬校验：
  1012 个结构性对象 + 204 个 NVFP4 投影 + 43 个 MMFlash 草稿对象。
  差一个都会在引擎加载时被拒（"projection produced N objects; expected 1259"）。

3) 转换报告：

```bash
$PY -c "import json;d=json.load(open('$OUT.conversion.json'));print(d['arguments'])"
```

`mmflash_source` 应该指向 `<MMFLASH_DIR>`。

4) 用引擎实际加载一次（最直接的验证）：

```bash
$W/build-v100/apps/ninfer-serve "$OUT" \
    --host 127.0.0.1 --port 18099 --tp 2 --devices 1,0 \
    --max-context 4096 --kv-dtype int8 \
    --spec dflash --draft-tokens 15
```

能打印出 `model loaded` 并开始监听，就说明 v3 目录、对象计数、bindings 和
`components/mmflash` 的校验全部通过了。这一步需要两个 GPU 和约 16 GB 显存。

（`--spec dflash --draft-tokens 15` 是 MMFlash 的块宽：16 宽块减掉锚点 token。
如果报 "DFlash draft window must be in [1,7]" 之类，说明加载的是旧引擎，
需要用本分支编译的二进制。）

## 已知约束

- **v3 读取路径是 `Qwen38Nvfp4V3Adapter`**（`src/artifact/reader.cpp`）。它对目录结构有
  严格要求：`metadata.name` 必须是 `qwen3.8-27b` 或 `qwen3.8-27b-<variant>`，
  `files` 恰好一项且 `path` 为 null，`objects` / `bindings` / `uses` / `components`
  齐全。移植过来的 `v3_plan.py` 生成的目录满足这些。
- **MMFlash 与 DFlash2 不能共存**。reader 检测到两个草稿组件会报错
  "carries more than one DFlash draft"。一个容器只放一个草稿。
- **转换器目前只写 v3 的 DFlash2 兼容目录**。若目标机器上 `v3_plan.py` 缺失，
  从 `4b8afc6` 取这两个文件：

  ```bash
  git checkout 4b8afc6 -- tools/artifact/container_v3.py tools/convert/qwen3_8_27b/v3_plan.py
  ```

  但要注意：`4b8afc6` 里的 `v3_plan.py` 不含 `components/mmflash`，
  本分支的版本才有（提交 `1dd0b47`），**以本分支的为准**，不要覆盖。

## 交回

产物 `/data/models/qwen3_8_27b_q4km_v2.ninfer`（约 17.5 GB，含 MMFlash 草稿包）
拷回 `172.16.3.88` 的 `/data/models/` 即可。

拷回后需要一起给我的还有：

- 上面「验证产物」的三项输出（贴文本即可）
- 转换报告 `.conversion.json`
