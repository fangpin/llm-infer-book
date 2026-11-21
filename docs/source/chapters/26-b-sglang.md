# 附录 B：SGLang 插件接入

```{contents} 本页目录
---
depth: 2
local: true
---
```

`entry_points` 是 Python 包提供的一套安装时注册、运行时发现机制。插件包把可调用对象写进安装元数据，主程序按约定的名字查找并加载。它适合独立发布的命令行工具和插件，也常用于硬件适配、框架扩展与后端接入。

这里的 entry point 指包元数据，不是 `if __name__ == "__main__"` 这种模块执行入口。

文中的 SGLang 示例按提交 `7c99ace127526c2a7b21e5474a2981e4789b367c` 核对。后续版本可能调整 group 或接口，应以对应版本的 `pyproject.toml` 和加载代码为准。

## entry_points 的声明、安装元数据与命令行区分

一条 `entry_point` 由 `group`（主程序约定的扩展分类名）、`name`（插件名）与 `value`（形如 `module:object` 的导入路径）三部分组成。执行 `pip install`（或开发期 `pip install -e .`）后，打包工具把 `pyproject.toml` 中的声明写入环境内的 `*.dist-info/entry_points.txt`，运行时主程序再通过 `importlib.metadata.entry_points()` 枚举并按需调用 `ep.load()`；仅把源码放进 `PYTHONPATH` 而未安装元数据，插件不会被发现。

```toml
# 1. SGLang 主仓库独享的 CLI 命令入口（扩展包切勿重复注册同名 sglang 脚本）
[project.scripts]
sglang = "sglang.cli.main:main"
killall_sglang = "sglang.cli.killall:main"

# 2. 独立硬件或后端扩展包声明的插件入口
[project.entry-points."sglang.srt.platforms"]
my_xpu = "sglang_my_xpu.platform:activate"
```

## 插件 group 定义主程序与扩展包的契约

SGLang 使用不同的 group 区分扩展目的。group 是双方约定的协议名，不会自动检查返回类型，主程序仍要在加载后验证。

| group | entry point 指向 | 加载后的约定 |
|-|-|-|
| `sglang.srt.platforms` | 无参数激活函数 | 硬件可用时返回平台类的完整路径，不可用时返回 `None`；`SGLANG_PLATFORM` 可以显式选择插件 |
| `sglang.srt.plugins` | 无参数注册函数 | 注册 Hook 等运行时扩展；`SGLANG_PLUGINS` 可以限制加载名单 |
| `sglang.serve_backends` | 无参数工厂函数 | 返回 `ServeBackend`；entry point 的 `name` 会成为 `--model-type` 的可选值 |

平台插件负责设备能力和必须成立的运行时契约。通用插件适合注册 Hook。serve backend 则把独立推理运行时接到 SGLang 维护的命令行入口上。

## 按需延迟加载与真实 SGLang 插件骨架

SGLang 枚举插件时先比对 `ep.name`（或环境变量 `SGLANG_PLATFORM` / `SGLANG_PLUGINS` 白名单），命中目标后才执行 `ep.load()` 导入模块，防止未启用的异构硬件驱动污染当前进程。因此插件包的顶层 `__init__.py` 必须保持轻量，绝不能在 import 阶段直接初始化设备、分配显存或拉起子进程。

下面给出一个同时对接 `sglang.srt.platforms` 与 `sglang.srt.plugins` 的最小生产级扩展包骨架：

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "sglang-acme-plugin"
version = "0.1.0"
dependencies = ["sglang"]

[project.entry-points."sglang.srt.platforms"]
acme_npu = "sglang_acme.platform:activate"

[project.entry-points."sglang.srt.plugins"]
acme_ops = "sglang_acme.plugin:register"
```

```python
# sglang_acme/platform.py
import importlib.util

def activate() -> str | None:
    # 仅做轻量探测：当前环境未安装厂商驱动时返回 None，让主程序跳过
    if importlib.util.find_spec("torch_acme") is None:
        return None
    return "sglang_acme.platform_impl.AcmePlatform"

# sglang_acme/plugin.py
def register() -> None:
    # 在此处向 SGLang 的算子、量化或自定义模型注册表挂载扩展实现
    from sglang.srt.models.registry import ModelRegistry
    # ModelRegistry.models["AcmeForCausalLM"] = ...
    return None
```

```bash
pip install -e .
python3 -c "from importlib.metadata import entry_points; print([(e.name, e.value) for e in entry_points(group='sglang.srt.platforms')])"
```

## 调试从当前 Python 环境和安装元数据开始

- 确认 `python` 与 `pip` 指向同一个虚拟环境。entry points 只能发现当前环境已经安装的 distribution。
- 修改 `pyproject.toml` 后重新安装。单纯修改 Python 文件通常不需要重装，修改安装元数据需要。
- 检查 group 名是否完全一致。group 只是字符串，拼写不一致不会得到额外提示，只会发现不到插件。
- 处理同名冲突。多个 distribution 可以在同一 group 注册相同的 `name`，主程序应检测冲突并给出提供方信息。
- 把 `ep.load()` 视为代码导入边界。它会执行模块顶层代码，只应加载可信的已安装插件。
- 为长期维护的插件协议增加类型检查或 API 版本。entry points 负责发现对象，不负责兼容性管理。

固定在同一项目内部的实现，用普通 import 和显式注册表通常更清楚。扩展来自独立安装包，而且主程序事先不知道会安装哪些实现时，entry points 才最合适。

本附录只说明安装元数据、插件发现与加载契约。插件能被发现，不表示其模型、设备或运行时接口已经兼容；实际接入仍须按对应 SGLang 版本验证。原始论文、仓库和其他阅读资料集中在〈参考〉。
