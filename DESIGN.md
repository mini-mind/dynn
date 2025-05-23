# DyNN 项目模块设计说明

本文档旨在阐述 DyNN 脉冲神经网络仿真框架的模块化设计，以及各个组件的职责与相互关系。

## 1. 项目总体结构

项目根目录包含以下主要部分：

*   `dynn/`: 存放 DyNN 框架的核心库代码。
*   `experiments/`: 包含用于运行具体仿真实验的脚本。
*   `tests/`: 包含项目的单元测试和集成测试。
*   `README.md`: 项目的高级别概述和快速开始指南 (取代了旧的需求规格)。
*   `requirements.txt`: 项目的 Python 依赖列表。
*   `DESIGN.md`: 本模块设计说明文档。

## 2. DyNN 核心库 (`dynn/`)

此目录是 DyNN 框架的核心，包含了实现 SNN 仿真所需的全部逻辑。

### 2.1. `__init__.py`

*   **用途**: 将 `dynn/` 目录标记为一个 Python 包，使其内部模块可以被导入。

### 2.2. `core/` - SNN 核心组件

此子模块封装了脉冲神经网络的基本构成要素和运行机制。

*   `core/__init__.py`:
    *   **用途**: 将 `core/` 目录标记为一个 Python 子包。
*   `core/neurons.py`:
    *   **用途**: 定义神经元模型（如类Izhikevich模型）及其参数化。实现 `NeuronPopulation` 类来管理一组神经元，处理其状态初始化、更新和信息获取。
    *   **对应旧需求**: 2.1. 神经元模型。
    *   **详细设计要求**:
        *   **基础模型:** 需实现一个可参数化的类Izhikevich尖峰神经元模型。允许对标准Izhikevich模型进行简化或修改，以平衡生物学合理性和计算效率。
        *   **参数化:** 必须支持为网络中每个神经元实例独立配置其模型参数 (如时间尺度、敏感度、重置参数等，对应于类Izhikevich模型中的 `a, b, c, d` 或等效参数)。
        *   **状态初始化:** 
            *   应提供默认的神经元初始状态 (如膜电位 `v` 和恢复变量 `u` 的固定值)。
            *   必须提供接口，允许用户通过配置指定初始状态的生成方式，例如从特定的统计分布 (如均匀分布、正态分布，参数可配置) 中采样。
        *   **状态变量访问:** 神经元对象必须能够追踪并暴露其关键内部状态变量 (例如，膜电位、恢复变量、自上次脉冲以来的时间、当前是否处于不应期等)，以便于外部模块进行数据记录、监控和调试。
        *   **群体抽象 (`NeuronPopulation`):** 设计一个抽象层来管理一组神经元。该抽象层应能高效地执行针对整个群体的操作，例如参数设置、状态更新、脉冲收集等。
*   `core/synapses.py`:
    *   **用途**: 定义和管理神经元之间的突触连接。包括权重的表示（如矩阵形式）、初始化策略、兴奋性/抑制性突触的区分，以及连接拓扑的灵活配置（如邻域连接、稀疏连接）。
    *   **对应旧需求**: 2.2. 突触与连接。
    *   **详细设计要求**:
        *   **突触连接 (`SynapseCollection` 或 `ConnectionManager`):** 设计一个组件来定义和管理神经元之间（或群体之间）的突触连接。
        *   **权重表示:** 突触权重应以高效的矩阵形式存储和更新。
        *   **权重初始化:** 必须支持根据用户定义的统计分布 (例如，具有可配置均值和标准差的正态分布) 来初始化突触权重。
        *   **兴奋性/抑制性突触:** 系统应能区分并管理兴奋性和抑制性突触。一个设计目标是允许配置网络中这两类突触的大致比例 (例如，初步考虑4:1的兴奋性:抑制性比例)。这可以通过权重符号的约束、分离的权重矩阵或与特定神经元类型关联来实现。
        *   **连接拓扑配置:** 
            *   **内部连接:** 对于单个神经元群体内部的连接，必须支持灵活的拓扑结构配置。具体应包括：
                *   **邻域连接:** 允许定义神经元的邻域范围 (基于其在一维或多维排列中的索引/位置)，并在该邻域内实现全连接。
                *   **稀疏连接:** 在非邻域或整个群体范围内，支持按指定概率或数量生成稀疏的随机连接。
                *   连接参数 (如邻域大小、连接概率、最大连接数等) 均需可配置。
        *   **学习规则接口:** 连接组件应与学习规则模块解耦，允许动态地为不同的连接组应用不同的学习规则。
*   `core/learning_rules.py`:
    *   **用途**: 实现学习规则，特别是基于脉冲迹的 STDP（尖峰时间依赖可塑性）。包含权重更新逻辑，并提供接口以接收外部奖励信号，用于调制学习率（LTP 和 LTD）。
    *   **对应旧需求**: 2.3. 学习规则。
    *   **详细设计要求**:
        *   **STDP核心:** 
            *   **基于脉冲迹的STDP:** 实现一种基于突触前和突触后神经元脉冲迹的STDP机制。
                *   **迹动态:** 神经元发放脉冲时，其关联的突触前/后迹变量应增加一个固定值，随后该迹变量随时间以可配置的速率指数衰减。
                *   **权重更新:** 权重的调整量应取决于相关的突触前/后迹值以及当前权重。初步要求实现**权重依赖的乘性STDP (multiplicative weight-dependent STDP)**。具体的数学形式和参数需通过实验来优化和最终确定。
        *   **奖励调制:** 
            *   **接口:** 学习规则模块必须提供接口，以接收一个外部计算的标量奖励信号。
            *   **学习率调制:** 该奖励信号用于动态调整STDP规则的有效学习率。调制机制应能以相同的方式 (例如，同向缩放或反向缩放) 影响权重的长期增强 (LTP) 和长期削弱 (LTD)。
*   `core/network.py`:
    *   **用途**: 提供一个顶层的 `NeuralNetwork` 对象，用于组织和管理整个 SNN，包括注册神经元群体、突触连接和学习规则。定义网络的输入和输出点。
    *   **对应旧需求**: 2.4. 网络构建与管理。
    *   **详细设计要求**:
        *   **网络对象 (`NeuralNetwork`):** 提供一个顶层对象来表示和管理整个SNN，包括所有的神经元群体、突触连接及其学习规则。
        *   **组件注册:** 支持将创建的神经元群体和突触连接注册到网络对象中。
        *   **输入/输出指定:** 允许用户通过配置明确指定网络中的哪些神经元或神经元子集作为外部输入的接收端，以及哪些作为产生行为输出的信号源。这通常通过神经元索引或标签来实现。
        *   **信号流完整性:** 确保从网络输入到输出的信号处理完全在SNN内部通过已定义的神经元和突触动态进行，避免外部模块对核心SNN运算的直接干预。
*   `core/simulator.py`:
    *   **用途**: 实现离散时间的仿真引擎。控制仿真循环（如按步长 `dt` 运行），协调神经元状态更新、突触信号传播和学习规则的应用。确保核心仿真逻辑与应用场景分离。
    *   **对应旧需求**: 2.5. 仿真引擎。
    *   **详细设计要求**:
        *   **离散时间仿真:** 仿真过程基于离散的时间步 (`dt`) 进行迭代。
        *   **高时间分辨率:** 仿真时间步长 `dt` 的选择必须足够小，以确保神经元模型动态的准确性和脉冲时序的精确性。同时，`dt` 也需要与目标应用 (如实时环境交互) 的时间尺度相协调。
        *   **仿真循环控制:** 仿真引擎应提供清晰的控制接口，例如 `run_step()` (执行单个 `dt` 步)、`run_for_duration(T)` (执行一段时间) 或 `run_n_steps(N)` (执行N步)。
        *   **核心逻辑分离:** SNN状态更新的核心逻辑 (神经元动态、突触传播、学习规则应用) 应与具体的应用场景 (如与特定Gym环境的交互逻辑) 分离，以保证框架的通用性和可重用性。

### 2.3. `io/` - 输入/输出与环境交互接口

此子模块负责处理 SNN 与外部世界（如 OpenAI Gym 环境）之间的数据交换。

*   `io/__init__.py`:
    *   **用途**: 将 `io/` 目录标记为一个 Python 子包。
*   `io/input_encoders.py`:
    *   **用途**: 提供将外部环境的观察值或高级控制信号转换为 SNN 输入神经元活动（如注入电流或目标脉冲）的机制。针对特定环境（如 MountainCar-v0 的位置信息）实现具体的编码器。
    *   **对应旧需求**: 3.1. 输入接口与编码器。
    *   **详细设计要求**:
        *   **通用输入接口:** 框架应提供一个通用的机制，用于将外部世界或高级控制模块的信号映射到SNN输入神经元的活动 (例如，直接注入电流或生成目标脉冲)。
        *   **特定编码器实现:** 
            *   针对 "MountainCar-v0" 环境，需实现一个输入编码器，该编码器仅使用小车的位置信息 (忽略速度信息)。
            *   编码方式和参数 (如高斯感受野的数量、宽度、中心点，或电流的转换尺度) 必须是可配置的。
        *   **配置灵活性:** 系统必须支持用户通过配置文件或编程接口灵活定义和选择输入编码方案及其参数。
*   `io/output_decoders.py`:
    *   **用途**: 提供从 SNN 输出神经元的活动中提取信息并转换为外部可理解的信号或动作（如离散动作）的机制。针对特定环境实现具体的解码器（如基于瞬时脉冲发放的决策）。
    *   **对应旧需求**: 3.2. 输出接口与解码器。
    *   **详细设计要求**:
        *   **通用输出接口:** 框架应提供一个通用的机制，用于从SNN输出神经元的活动中提取信息，并将其转换为外部世界或高级控制模块可理解的信号/动作。
        *   **特定解码器实现:** 
            *   针对 "MountainCar-v0" 环境，需实现一个输出解码器，该解码器基于SNN输出神经元的**瞬时脉冲发放**情况来决定离散动作 (左、右、无)。避免使用需要在一个时间窗口内累积活动的方法。
            *   解码逻辑 (例如，基于"赢者通吃"的竞争机制、特定神经元组合的激活模式、或简单的投票机制) 及其参数应可配置。
        *   **配置灵活性:** 系统必须支持用户通过配置文件或编程接口灵活定义和选择输出解码方案及其参数。
*   `io/reward_processors.py`:
    *   **用途**: 处理来自环境的原始奖励信号。例如，可以实现奖励的平滑处理（如滑动平均），生成用于调制学习规则的最终标量奖励信号。
    *   **对应旧需求**: 3.3. 奖励接口与处理。
    *   **详细设计要求**:
        *   **通用奖励接口:** 学习规则模块应能接收一个经过处理的标量奖励信号。
        *   **特定奖励函数:** 
            *   针对 "MountainCar-v0" 环境，用于生成原始奖励的函数的形式需要实验确定。一个可供初步探索的方案是基于小车位置和加速度 (或速度变化) 的组合。
        *   **奖励信号平滑:** 提供一个奖励处理器模块，该模块接收来自环境的原始（或自定义计算的）奖励值，并基于一个可配置的时间窗口内的**滑动平均值**来计算最终用于调制学习率的平滑奖励信号。

### 2.4. `utils/` - 通用工具模块

此子模块包含框架所需的通用工具和辅助功能。

*   `utils/__init__.py`:
    *   **用途**: 将 `utils/` 目录标记为一个 Python 子包。
*   `utils/probes.py`:
    *   **用途**: 实现数据记录和监控功能。提供灵活的"探针"机制，用于在仿真过程中追踪和记录各种关键数据。
        *   定义了 `BaseProbe` 基类，提供通用功能如记录间隔控制、数据存储、数据获取 (`get_data`)、重置 (`reset`) 和CSV导出 (`export_to_csv`)。子类需实现具体的 `_collect_data` 逻辑。
        *   实现了具体的探针类：
            *   `PopulationProbe`: 用于记录神经元群体中特定状态变量（如膜电位 `v`、脉冲 `fired`）的时间序列数据。
            *   `SynapseProbe`: 用于记录突触集合的状态，例如突触权重矩阵随时间的变化。
            *   `CustomDataProbe`: 允许用户提供自定义函数来收集和记录仿真过程中的任意数据。
        *   探针实例会被注册到 `NeuralNetwork` 对象中，并在每个仿真步骤由网络协调其 `attempt_record` 方法的调用。
    *   **对应旧需求**: 3.4. 数据记录与监控。
    *   **详细设计要求**:
        *   **可监测变量:** 系统必须能够追踪和记录仿真过程中的各种关键数据，包括但不限于：
            *   **神经元级别:** 膜电位、恢复变量、脉冲发放的精确时间或在每个 `dt` 的发放状态。
            *   **突触级别:** 单个或一组突触的权重随时间的变化。
            *   **学习规则相关:** STDP迹变量的值，学习率调制因子的变化。
            *   **网络输入/输出:** 施加到输入神经元的等效电流/脉冲模式，解码器输出的动作信号。
            *   **环境与奖励:** 环境的原始观察值、智能体执行的动作、环境返回的原始奖励、处理后的平滑奖励信号。
        *   **记录机制:** 提供灵活的数据记录机制 (类似"探针"的概念)，允许用户选择要记录哪些变量、记录频率以及存储格式。
        *   **数据访问:** 记录的数据应易于通过编程接口访问，并能方便地导出为常用格式 (如CSV, NumPy数组) 以供后续分析和可视化。

### 2.5. `config.py` - 系统配置管理

*   **用途**: 集中管理整个 DyNN 框架的所有可配置参数。这包括 SNN 模型参数、连接拓扑参数、学习规则参数、输入/输出编码/解码方案参数、仿真控制参数等。支持从文件加载默认配置，并允许通过编程接口覆盖。
*   **对应旧需求**: 4. 系统配置与管理。
*   **详细设计要求**:
    *   **分层与模块化配置:** 所有的系统参数应通过一个结构化、易于理解和修改的配置系统进行管理。建议采用分层或模块化的配置文件 (例如，针对SNN核心、特定环境、特定实验的独立配置部分或文件)。
    *   **参数覆盖与加载:** 配置系统应支持从文件加载默认参数，并允许用户通过编程接口或额外的配置文件方便地覆盖特定参数以进行实验。
    *   **可配置项列表 (示例):**
        *   SNN模型参数 (神经元总数、神经元类型及其参数的分布定义等)。
        *   连接拓扑参数 (邻域定义、稀疏连接的概率/密度、兴奋性/抑制性突触的比例等)。
        *   权重初始化策略 (分布类型、均值、标准差等)。
        *   STDP及其他学习规则的参数 (基准学习率、迹的时间常数、权重依赖性的具体形式、奖励调制函数的参数、滑动平均窗口大小等)。
        *   输入编码方案及其参数。
        *   输出解码方案及其参数。
        *   仿真控制参数 (总仿真时长、时间步长 `dt`)。
        *   特定于所选Gym环境的参数。

## 3. 实验脚本 (`experiments/`)

此目录用于存放具体的实验运行脚本。

*   `experiments/mountain_car_v0_experiment.py`:
    *   **用途**: 示例脚本，演示如何配置和运行 DyNN 框架来解决 "MountainCar-v0" 环境。

## 4. 测试 (`tests/`)

此目录用于存放单元测试和集成测试代码，以保证框架的质量和稳定性。

*   `tests/__init__.py`:
    *   **用途**: 将 `tests/` 目录标记为一个 Python 包，方便测试框架（如 `unittest`）发现和运行测试。
*   `tests/test_core_neurons.py`:
    *   **用途**: 示例测试文件，用于测试 `dynn.core.neurons` 模块的功能。

## 5. 根目录文件

*   `README.md`: 项目的高级别概述和快速开始指南 (取代了旧的需求规格)。
*   `requirements.txt`: 列出项目运行所需的 Python 库及其版本，如 `numpy` 和 `gymnasium`。
*   `DESIGN.md` (本文档): 对项目模块设计的详细说明。 