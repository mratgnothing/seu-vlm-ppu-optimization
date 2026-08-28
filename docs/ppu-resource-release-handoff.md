# PPU 资源释放前快照与恢复手册

更新时间：2026-08-28

## 1. 当前恢复原则与历史快照

自 2026-08-29 起，工程文件采用以下唯一权威链路：

1. 本机 `5070ti` 工作区保存完整开发状态；
2. 每个可恢复节点提交并 push 到 GitHub `5070ti`；
3. PPU 服务器只从 GitHub 临时 clone、编译和运行，服务器/CPFS 上的工程副本、venv
   和 `.so` 均不作为交付或下一次恢复来源；
4. 新实例使用官方镜像和仓库内 `scripts/bootstrap_ppu_env.sh` 重建环境，不依赖自定义
   镜像；
5. 模型、数据和不可提交的大结果另存本机，按各自许可证与比赛规则管理。

下文的 CPFS 路径和归档是此前实验的历史证据，不再代表当前工程恢复方案。不要因
切换方案而自动删除其中数据；删除属于单独的破坏性操作，必须人工确认。

### Git 可公开层

- 分支：`5070ti`；
- gate-prep 开发前的已推送基线：`a04c3b6676ed227c66e7e228d45d0fba4889d9c6`；
- 最新 gate-prep 提交以 `5070ti` 分支的 `git rev-parse HEAD` 为准；通过 GitHub
  SSH-over-443 推送并用 `git ls-remote` 核验，不依赖不稳定的 HTTPS；
- 源码、脚本、小型结果与实验说明进入 Git；密钥、模型、数据和原始巨型 trace 不进入。

### 本地 ignored artifact 层

目录：`artifacts/ppu-snapshot-20260827/`。它被 `.gitignore` 排除，但已在本机保存：

| 文件 | 内容 | SHA-256 |
|---|---|---|
| `ppu-progress-snapshot-20260827-final.tar.gz` | 远端最终实验源码、编译产物、全部小型结果（含 SwiGLU 负实验）、pip freeze、设备/编译器信息；排除巨型 trace | `fcd3e21b4d474bf9061cc926ba58fdc890150e36d7c8edf2978d15b9c282b9b2` |
| `ppu-profile-traces-20260827.tar.gz` | 全部原始 PyTorch profiler trace | `6e8f3f317a7721e5cfc47b17b7df434950b84c33d1cdce6bde421cd8b8d8eceb` |
| `mmbench-local-copy-20260827.tar.gz` | 远端 `datasets/mmbench` 完整副本 | `7e27032f5b5cccc75371bd1c7d7115cad59ccd5be449f3ea011088c3bd19af01` |

2026-08-28 gate-prep/GEMM 迭代另在本地 ignored
`artifacts/ppu-snapshot-20260828/` 保存最终共享库、4029 条原始 paired A/B、四份
acBLASLt heuristic JSONL 和四份原始 profile trace；源码、小型聚合 JSON、memcheck
与实验说明进入 Git。新增 CPFS/本地双副本归档：

| 文件 | 内容 | SHA-256 |
|---|---|---|
| `ppu-final-evidence-20260828.tar.gz` | 实验源码、构建产物、完整集原始结果和小型证据；排除巨型 trace/log | `b581bbc957ddbedc1ab4ab08e0a8e8efd84a1fb02bda3c190f0db39c839cdfcb` |
| `ppu-acblaslt-traces-20260828.tar.gz` | acBLASLt 方阵负实验两份原始 trace | `ba237680986179ea14d95ea0aa970693fdc404133e347e46cce23fc76589705c` |
| `ppu-acblas-packed-mlp-evidence-20260828.tar.gz` | 最终 extension、两份 packed-MLP 原始 profile trace、4029 原始结果/检查点及最终 smoke/memcheck | `8451b9392faec9702c6d30a9003474f8989985bf65321a57e2cb0158592e7c15` |

逐文件哈希见 `artifacts/ppu-snapshot-20260828/SHA256SUMS-final.txt`；本地值已与服务器
源文件逐项匹配，不覆盖上表 2026-08-27 三份已核验快照。

本地模型已在 `models/Qwen3.5-2B/`；锁定 revision 为
`15852e8c16360a2fea060d615a32b45270f8a8fc`，主权重 SHA-256 为
`aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`。

### CPFS 持久化层

本次实例的 `/mnt/workspace` 与 `/mnt/cpfs` 是同一个智算 CPFS 文件系统，而非容器
临时层。当前关键目录：

```text
/mnt/workspace/seu/Qwen3.5-2B
/mnt/workspace/seu/datasets/mmbench
/mnt/workspace/seu/envs/seu-vlm-ppu-20260826
/mnt/workspace/seu/acblas-extension-work-20260827
/mnt/workspace/seu/ppu-progress-snapshot-20260827-final.tar.gz
/mnt/workspace/seu/ppu-profile-traces-20260827.tar.gz
/mnt/workspace/seu/mmbench-local-copy-20260827.tar.gz
/mnt/workspace/seu/ppu-final-evidence-20260828.tar.gz
/mnt/workspace/seu/ppu-acblaslt-traces-20260828.tar.gz
/mnt/workspace/seu/ppu-acblas-packed-mlp-evidence-20260828.tar.gz
/mnt/workspace/seu/submission-source-20260828.zip
```

2026-08-29 无镜像权限收尾新增以下清单到 `/mnt/workspace/seu/archives/`：

- `ppu-python-freeze-20260829.txt`：171 个 Python 分发包；
- `ppu-runtime-manifest-20260829.txt`：OS、Python、PPU 编译器、`ppu-smi` 与关键目录大小；
- `ppu-build-result-sha256-20260829.txt`：最新扩展和 CN100 结果校验值。

| 对象 | SHA-256 |
|---|---|
| `seu_acblas_linear_ext.so` | `aaf6993f4598bf9b86cfd59e0301d9b5aa70414cf46c8da1423cfbb1ea71c461` |
| `libseu_ppu_gdn.so` | `e742c999fd5f9df4197b864a8ce90a95efef2ed213978cebc2b42d66ae0c80fb` |
| `seu_acblas_packed_mlp_ext.so` | `86aaf036ad80b02e7c6b183fe3cf7fb6da95f3f2ec9a4b4b7bc05ae8f0a72c8d` |
| b/a-GEMV CN100 JSON | `c0d0250cd54666dbbc2a2867c995a36982fa4093adc9fde45bddfe5ce88e14cf` |

最终源码白名单包在所有文档与代码冻结后重新生成并验证；最终文件数和 SHA-256
记录在本地 ignored 的 `artifacts/ppu-snapshot-20260828/SHA256SUMS-final.txt`，并以
CPFS 上对同名文件执行 `sha256sum` 的结果交叉核验。这里不内嵌源码包自身哈希，避免
包内文档自引用导致哈希随记录值再次改变。

释放计算实例前不要删除 CPFS 文件系统；创建下一台 DSW 时把同一 CPFS 根目录重新
挂载到 `/mnt/workspace`。自定义镜像可加速系统环境恢复；无权限时使用同版本官方 PPU
镜像，并以 CPFS 中的 venv、代码、模型、数据、结果和环境清单恢复。

## 2. 在 PAI DSW 控制台制作镜像

阿里云官方步骤见：[制作 DSW 实例镜像](https://help.aliyun.com/zh/pai/create-a-dsw-instance-image)
和 [访问与管理 DSW 实例](https://help.aliyun.com/zh/pai/access-dsw-instance)。

> 2026-08-29 当前比赛账号没有制作 DSW 镜像/写入 ACR 的权限，因此本次不把镜像作为
> 释放前置条件。恢复基线改为“同版本官方 PPU 镜像 + 原 CPFS + CPFS venv + Git 提交 +
> 已保存源码/构建快照”。禁止因没有镜像权限而删除 CPFS；若后续获得权限，再按下列步骤
> 补做镜像。

1. 保持 DSW 实例状态为 **运行中**；停止后“制作镜像”按钮会变灰。
2. 在与当前 DSW **完全相同地域**的容器镜像服务 ACR 中，先创建实例、命名空间和
   镜像仓库。个人版适合临时备份；企业版同 VPC 内网推送更稳定。
3. 回到 `PAI > 工作空间 > 交互式建模（DSW）`，在实例右侧选择 **制作镜像**。
4. 选择 ACR 类型、命名空间与仓库；建议 tag：`seu-ppu-qwen35-20260827`。
5. 排除大文件和挂载路径。官方默认已经排除 `/mnt/workspace`，本实例建议确认或额外
   排除：

   ```text
   /mnt/workspace
   /mnt/cpfs
   /tmp
   /root/.cache
   ```

   模型和数据已经在 CPFS 与本地备份中，不应打进镜像。官方限制单层镜像不超过
   10 GiB；超过会构建失败。
6. 单击保存，等待实例从“保存中”恢复；在 PAI 镜像管理或 ACR 仓库确认 tag 和镜像
   地址已经生成。
7. **先验证再释放**：用该自定义镜像创建一台同型号 PPU DSW，并重新挂载同一 CPFS。
8. 新实例验证通过后，再停止/释放旧实例。若镜像名以 `_accelerated` 结尾而平台拒绝
   保存，按官方 FAQ 改用可保存的基础镜像；同时本项目仍可用“同版 PPU 官方镜像 +
   CPFS venv + pip-freeze + 源码快照”恢复。

## 3. 新实例恢复与最小验收

```bash
cd /tmp
git clone --branch 5070ti --single-branch \
  https://github.com/mratgnothing/seu-vlm-ppu-optimization.git
cd seu-vlm-ppu-optimization
bash scripts/bootstrap_ppu_env.sh --check-only
bash scripts/bootstrap_ppu_env.sh
source scripts/activate_ppu_env.sh
```

脚本已经重编译 recurrent GDN、acBLAS Linear、单入口 packed-MLP 三个扩展，并完成
设备与三条扩展路径的短 smoke。它不会安装 `torch`，且会拒绝 venv 覆盖官方镜像的
PPU 定制 Torch。需要离线部署时传入 `--wheelhouse PATH`；只重建不跑设备 smoke 时
传入 `--skip-smoke`。

随后给出模型和数据的外部路径，先运行正式 wrapper 单样本；禁止直接以完整公开集
开局：

```bash
scripts/run_ppu_first_validation.sh \
  --model-path /external/path/Qwen3.5-2B \
  --dataset-path /external/path/mmbench_dev_cn.tsv \
  --run-device-smoke --run-model-load --run-single-sample
```

若官方镜像中的 Torch/SDK ABI 变化，部署脚本会在编译或 smoke 阶段失败。此时保留
完整日志并修正 GitHub 中的源码/脚本；不要从旧服务器复制 `.so` 或系统库绕过重编译。

## 4. 释放前人工确认清单

- [!] ACR 镜像：当前账号无制作权限，已记录为外部权限限制；使用官方镜像 + CPFS 恢复；
- [x] 工程权威副本改为本机 + GitHub；服务器/CPFS 不再承担工程恢复职责；
- [x] 本地 2026-08-27 三份归档和 2026-08-28 三份归档 SHA-256 全部匹配；
- [ ] 本地 Qwen 权重哈希匹配；
- [ ] `5070ti` 本地分支提交存在，已 push 并核对远端哈希；
- [x] `SEU_PPU_GDN_GATE_PREP_ENABLE=1` 时 meta 记录 18 个 gate-prep module；
  最终正式 smoke 的完整计数为 `18/18/49/18/6/24/24/18/24/18`；
- [ ] 用新镜像 + 同一 CPFS 创建的新实例通过 device、单算子、单样本三层 smoke；
- [ ] 上述全部完成后才释放旧实例。
