# PPU 资源释放前快照与恢复手册

更新时间：2026-08-27

## 1. 已完成的三层备份

### Git 可公开层

- 分支：`5070ti`；
- residual-RMSNorm 正向里程碑本地提交：`a71108fe53675bcec4123703061274722e9367d7`；
- GitHub 当前 HTTPS 被本机网络重置，提交尚待网络恢复后 push；
- 源码、脚本、小型结果与实验说明进入 Git；密钥、模型、数据和原始巨型 trace 不进入。

### 本地 ignored artifact 层

目录：`artifacts/ppu-snapshot-20260827/`。它被 `.gitignore` 排除，但已在本机保存：

| 文件 | 内容 | SHA-256 |
|---|---|---|
| `ppu-progress-snapshot-20260827-final.tar.gz` | 远端最终实验源码、编译产物、全部小型结果（含 SwiGLU 负实验）、pip freeze、设备/编译器信息；排除巨型 trace | `fcd3e21b4d474bf9061cc926ba58fdc890150e36d7c8edf2978d15b9c282b9b2` |
| `ppu-profile-traces-20260827.tar.gz` | 全部原始 PyTorch profiler trace | `6e8f3f317a7721e5cfc47b17b7df434950b84c33d1cdce6bde421cd8b8d8eceb` |
| `mmbench-local-copy-20260827.tar.gz` | 远端 `datasets/mmbench` 完整副本 | `7e27032f5b5cccc75371bd1c7d7115cad59ccd5be449f3ea011088c3bd19af01` |

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
```

释放计算实例前不要删除 CPFS 文件系统；创建下一台 DSW 时把同一 CPFS 根目录重新
挂载到 `/mnt/workspace`。镜像负责系统环境，CPFS 负责代码、venv、模型、数据和结果，
两者缺一不可。

## 2. 在 PAI DSW 控制台制作镜像

阿里云官方步骤见：[制作 DSW 实例镜像](https://help.aliyun.com/zh/pai/create-a-dsw-instance-image)
和 [访问与管理 DSW 实例](https://help.aliyun.com/zh/pai/access-dsw-instance)。

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
test -f /mnt/workspace/seu/Qwen3.5-2B/config.json
test -f /mnt/workspace/seu/acblas-extension-work-20260827/gdn_recurrent_ppu.hg
source /mnt/workspace/seu/envs/seu-vlm-ppu-20260826/bin/activate

export PPU_SDK=/usr/local/PPU_SDK
export PPU_HOME=/usr/local/PPU_SDK
export LD_LIBRARY_PATH="$PPU_SDK/lib:$PPU_SDK/lib64:${LD_LIBRARY_PATH:-}"

python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY

sha256sum /mnt/workspace/seu/Qwen3.5-2B/model.safetensors-00001-of-00001.safetensors
sha256sum /mnt/workspace/seu/datasets/mmbench/mmbench_dev_cn.tsv
```

然后先运行单算子 smoke，再运行正式 wrapper 单样本；禁止直接以完整公开集开局：

```bash
cd /mnt/workspace/seu/acblas-extension-work-20260827
python smoke_residual_rmsnorm_integration.py \
  --library build/residual-rmsnorm/libseu_ppu_gdn.so \
  --threads 512 --warmup 5 --iters 20
```

若自定义 `.so` 因新镜像 ABI 变化无法加载，使用保存的 `gdn_recurrent_ppu.hg` 和
`build_gdn_shared.sh` 在新实例重新编译，不要复制其他机器的系统库。

## 4. 释放前人工确认清单

- [ ] ACR 镜像状态成功，记录完整镜像地址和 tag；
- [ ] CPFS 文件系统仍存在，且快照归档 SHA-256 可读取；
- [ ] 本地三个 tar.gz SHA-256 全部匹配；
- [ ] 本地 Qwen 权重哈希匹配；
- [ ] `5070ti` 本地分支提交存在；网络恢复后 push 并核对远端哈希；
- [ ] 用新镜像 + 同一 CPFS 创建的新实例通过 device、单算子、单样本三层 smoke；
- [ ] 上述全部完成后才释放旧实例。
