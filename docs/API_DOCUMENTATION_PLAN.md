# DyNN API 文档编写计划

本文档概述了为 DyNN 框架创建 API 文档的计划和结构。

## 目标

为 DyNN 框架的每个主要模块和组件提供清晰、全面的 API 参考文档，方便开发者理解和使用。

## 文档存放位置

所有 API 文档将存放在 `docs/api/` 目录下。每个主要模块或子模块将对应一个 Markdown 文件。

## 计划涵盖的模块

我们将按以下顺序为各模块创建 API 文档：

1.  **`dynn.core` (核心SNN组件)**
    *   `docs/api/core_neurons.md`: 描述 `IzhikevichNeuron` 类和 `NeuronPopulation` 类。 (进行中)
    *   `docs/api/core_synapses.md`: 描述突触连接管理，如 `SynapseCollection`。
    *   `docs/api/core_learning_rules.md`: 描述学习规则，如 `STDP` 和 `RewardModulatedSTDP`。
    *   `docs/api/core_network.md`: 描述 `NeuralNetwork` 类。
    *   `docs/api/core_simulator.md`: 描述 `Simulator` 类。

2.  **`dynn.io` (输入/输出与环境交互)**
    *   `docs/api/io_input_encoders.md`: 描述输入编码器，如 `GaussianEncoder`, `CurrentInjector`。
    *   `docs/api/io_output_decoders.md`: 描述输出解码器，如 `WinnerTakesAllDecoder`, `SpikeRateDecoder`。
    *   `docs/api/io_reward_processors.md`: 描述奖励处理器，如 `SlidingWindowSmoother`。

3.  **`dynn.utils` (通用工具)**
    *   `docs/api/utils_probes.md`: 描述数据探针，如 `PopulationProbe`, `SynapseProbe`, `CustomDataProbe`。

4.  **`dynn.config` (系统配置)**
    *   `docs/api/config.md`: 描述配置加载和管理机制。

## 单个模块 API 文档结构

每个模块的 API 文档 (例如 `core_neurons.md`) 将遵循以下结构：

1.  **模块概览**:
    *   简要介绍该模块的功能和用途。
    *   列出该模块中主要的类和函数。

2.  **类详细说明**:
    *   对于模块中的每个公开类：
        *   **类名及其构造函数签名**: `ClassName(param1, param2, ...)`
        *   **类描述**: 详细说明类的作用和设计目的。
        *   **构造函数参数**:
            *   `param1` (`type`): 描述。默认值（如有）。
            *   `param2` (`type`): 描述。默认值（如有）。
            *   ...
        *   **主要属性/特性**:
            *   `attribute1` (`type`): 描述。
            *   ...
        *   **主要方法**:
            *   `method1(self, arg1, arg2, ...)`:
                *   方法描述。
                *   参数:
                    *   `arg1` (`type`): 描述。
                    *   ...
                *   返回值 (`type`): 描述。
            *   `method2(self, ...)`:
                *   ...
        *   **(可选) 使用示例**: 简短的代码片段展示如何使用该类。

3.  **函数详细说明 (如果模块包含独立函数)**:
    *   对于模块中的每个公开函数：
        *   **函数签名**: `function_name(arg1, arg2, ...)`
        *   **函数描述**: 详细说明函数的作用。
        *   **参数**:
            *   `arg1` (`type`): 描述。
            *   ...
        *   **返回值 (`type`): 描述。
        *   **(可选) 使用示例**。
