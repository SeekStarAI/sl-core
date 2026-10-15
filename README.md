# sl-core

StarLight 的模型装配与无状态单步推理核心，负责组合静态计算图、权重和虚拟设备接口，并推进一次 `step`。

## 职责边界

- 连接 `sl-graph`、`sl-precision` 与 `sl-device`。
- 管理模型执行所需的页表绑定和预分配缓冲区。
- 保持热路径无网络、请求队列和运行期堆分配。

接口与实现仍在基线阶段；具体契约以 RFC-0001 为准。

## 开发

在 StarLight workspace 根目录运行 `cargo check -p sl-core` 或 `cargo test -p sl-core`。
