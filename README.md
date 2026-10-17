# sl-core

StarLight 的运行时装配与无状态单步执行核心：接收已编译的 `CommandBuffer`、权重绑定与设备，构建 `Engine` 并推进一次 `step`。

## 当前能力

- `CompiledProgram`：把 IR 与语义绑定放在一起；
  - `ir`、`weights`（`ValueId → BufferId`）
  - `token_ids`、`positions`、`seq_len`、`past_len`、`logits`
- `EngineConfig`：prefill / decode 两个程序 + `Box<dyn Device>` + 上下文与批大小；
- `Engine::build`：校验 IR、分配 scratchpad/KV/输入/输出、预解析每条命令的 `BufferRef`；
- `Engine::step`：上传 token/positions、按序 `dispatch`、同步、下载 logits；
- `EngineInfo`：IR 版本、最大上下文与批大小。

`step` 的签名里没有 `Vec`/`Box`/`String`；命令绑定在 build 期预计算，运行期不分配、不加锁。

## 边界

- **不编译图**：只接收 `CommandBuffer`，因此不依赖 `sl-graph`；
- **不实现算子**：算子由 `sl-device::Device` 的后端实现执行；
- 不做 HTTP、队列、tokenizer、采样、流式或会话管理——这些属于 `sl-daemon` / `sl-tokenizer`；
- 不解析模型文件；模型加载与 lowering 属于 `sl-model` / `sl-model-adapters`。

契约依据：[ARCH-0007](<../../docs/ARCH-0007: Graph Core Runtime 三线施工方案.md>)。

## 测试

```sh
cargo test -p sl-core
cargo clippy -p sl-core --all-targets -- -D warnings
```

当前 9 个测试，使用记录型 `MockDevice` 覆盖 build/step、绑定偏移、输出下载与错误路径。
