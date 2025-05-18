'''
## MountainCar-v0 强化学习实验脚本 (基于 DyNN)

本脚本使用 DyNN (Dynamic Neural Networks) 脉冲神经网络 (SNN) 框架，
实现了一个强化学习智能体来解决 OpenAI Gym 中的 "MountainCar-v0" 环境。

核心设计思路:
- **输入编码 (Input Encoding)**:
    - 小车的水平位置 (X) 被映射到一个线性神经元阵列 (输入群体)。
    - 每个输入神经元对应一个特定的X坐标区间，形成类似感受野的机制。
    - 当小车处于某个位置时，对应位置及其附近的数个输入神经元被激活 (例如，激活强度可由高斯函数确定)。
    - 小车速度信息不直接编码为输入层神经元的活动。
- **SNN 结构与动力学 (SNN Architecture & Dynamics)**:
    - **神经元模型**: 采用 Izhikevich 模型，其参数可灵活配置。
    - **网络连接**: 输入神经元通过全连接方式连接到输出神经元 (注：连接细节可配置)。
    - **突触权重**: 初始突触权重从正态分布中采样生成，但允许配置为其他分布或固定值。
    - **神经元初始状态**: 神经元的初始膜电位和恢复变量使用固定值，但支持灵活配置为从某种分布中采样。
    - **仿真保真度**: 神经元模型的内部变量（如膜电位、恢复变量等）会尽可能保留，以便于后续的数据记录、监测与分析。
- **输出解码 (Output Decoding)**:
    - **神经元分组**: 输出神经元群体被划分为两个数量相等的部分，分别代表"向左运动"和"向右运动"的决策意图。
    - **决策机制**: 采用瞬时脉冲计数。比较在一个极短时间窗口内，"向左"组神经元和"向右"组神经元的脉冲发放数量。
    - **动作选择**: 当两个方向组的脉冲计数的差值绝对量超过一个预设阈值时，智能体将采取对应方向的运动（向左或向右）。如果差值未达到阈值，或者没有神经元发放脉冲，则可能不采取动作或执行预设的默认动作。
- **学习与奖励 (Learning & Reward)**:
    - **奖励函数设计**: 目标奖励函数旨在结合小车的高度 (Y) 与其向右的速度 (V) 来综合评估当前状态的好坏。例如，一个可能的奖励函数形式为 `Reward = height(Y) + velocity_right(V)`。
        (注: 实际脚本执行时，奖励函数的确切形式可能需要根据配置或代码实现确认，此处描述的是核心设计目标。)
    - **学习规则**: 采用奖励调制的脉冲时间依赖可塑性 (Reward-Modulated STDP)。
        - STDP 迹 (trace) 的更新规则为：神经元发放脉冲后，其对应的迹增加一个固定值，随后迹值随时间快速指数衰减。
        - STDP 更新类型选择为权重依赖的乘性更新规则。
    - **奖励调制机制**: 奖励信号（可能经过滑动时间窗口平均平滑处理）用于动态调整 STDP 学习规则中的学习率，从而调制突触权重的增强（LTP）和抑制（LTD）的幅度。这种调制对增强和抑制过程产生相同方式的影响。
- **仿真与环境交互 (Simulation & Environment Interaction)**:
    - **时间同步与分辨率**: SNN 仿真器内部使用较高的时间分辨率，并与 Gym 环境的步进保持同步。
    - **模块化与可配置性**: 输入编码、输出解码以及SNN到环境的映射关系设计为可灵活配置，实现仿真核心与具体环境的解耦。
    - **实验参数化**: 大部分实验参数，如网络结构、神经元参数、学习规则参数、仿真时长等，均可通过外部 YAML 配置文件进行设置和管理。
- **数据记录与分析**: 
    - 实验过程中会记录关键数据，例如每轮的总奖励、SNN内部的脉冲活动、突触权重的动态变化等，以供后续的性能分析和算法调试。

实验目标:
- 通过上述设计的SNN智能体和学习机制，训练智能体在 "MountainCar-v0" 环境中学习有效的攀爬策略，最终目标是能够稳定地将小车驱动到山顶的旗帜位置。
'''
import argparse
import os
import sys
import yaml # For loading config

# Ensure project root is in Python path
# Assuming run_main_experiment.py is in the project root.
# If dynn is installed or project root is in PYTHONPATH, this might not be strictly necessary.
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir)) # Assumes script is in root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"DEBUG: project_root = {project_root}")
print(f"DEBUG: sys.path = {sys.path}")

from dynn.session import ExperimentSession
# Update imports to use the new mountain_car_setup.py
from mountain_car_setup import (
    create_mountain_car_env,
    build_mountain_car_network,
    create_mountain_car_input_encoder,
    create_mountain_car_output_decoder,
    create_mountain_car_reward_processor,
    get_env_properties,
    plot_experiment_results # Added for plotting
)

def load_config(config_path):
    """Loads YAML configuration file."""
    abs_config_path = config_path
    if not os.path.isabs(abs_config_path):
        # current_script_dir is the directory of this script (run_main_experiment.py)
        # Configs are now in 'configs' subdirectory relative to this script's location
        abs_config_path = os.path.join(current_script_dir, abs_config_path) 
    
    if not os.path.exists(abs_config_path):
        print(f"错误: 配置文件 '{abs_config_path}' 未找到。")
        # Update the hint for the default config path
        common_default_location = os.path.join(current_script_dir, "configs", "mountain_car_v0_config.yaml")
        # Check if the originally sought path (after joining with current_script_dir) matches the new common_default_location
        # This comparison might be tricky if config_path was already absolute.
        # A simpler approach for the hint:
        print(f"提示: 默认配置文件应该位于: {os.path.join(current_script_dir, 'configs', 'mountain_car_v0_config.yaml')}")
        print(f"请检查路径: {config_path} (解析为: {abs_config_path})")
        return None
        
    with open(abs_config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"错误: 解析 YAML 配置文件 '{abs_config_path}' 失败: {e}")
            return None
    print(f"配置文件 '{abs_config_path}' 加载成功。")
    return config

def main():
    parser = argparse.ArgumentParser(description="使用 DyNN ExperimentSession 运行实验。")
    
    # Update default config path
    default_config_path = os.path.join("configs", "mountain_car_v0_config.yaml") 
    
    parser.add_argument(
        '--config',
        type=str,
        default=default_config_path,
        help=f'实验的配置文件路径 (YAML)。默认为: {default_config_path}'
    )
    args = parser.parse_args()

    config_data = load_config(args.config)
    if config_data is None:
        return

    print("初始化 ExperimentSession...")
    session = ExperimentSession(config=config_data)

    # Create env first to get its properties for other creators
    env_config = config_data.get('environment', {})
    temp_env = create_mountain_car_env(env_config) # Create once to get props
    env_props = get_env_properties(temp_env)
    temp_env.close() # Close temp env, session will create its own managed one

    # snn_config, env_config etc. are now read by ExperimentSession from the main config_data
    # print(f"DEBUG: snn_config in run_main_experiment.py before session.setup: {snn_config}") 

    print("设置实验组件...")
    # ExperimentSession.setup will now get its sub-configs from self.config
    session.setup(
        env_creator_fn=lambda: create_mountain_car_env(config_data.get('environment', {})),
        network_builder_fn=lambda net_conf: build_mountain_car_network(net_conf, env_props),
        input_encoder_creator_fn=lambda enc_config, target_pop_instance: create_mountain_car_input_encoder(enc_config, target_pop_instance, env_props),
        output_decoder_creator_fn=lambda dec_config, source_pop_instance: create_mountain_car_output_decoder(dec_config, source_pop_instance, env_props),
        reward_processor_creator_fn=lambda proc_config: create_mountain_car_reward_processor(proc_config)
    )

    if not session.is_setup_complete():
        print("实验设置未完成。")
        return

    print("开始实验运行...")
    try:
        results = session.run()
        print("实验运行结束。")
        # Results are saved by session._finalize_and_save_results()
        # Additional plotting or analysis can be done here if needed,
        # for example, using data from `results` (list of episode_log dicts)
        # or by loading saved probe data.
        
        # Activate plotting
        if results:
            # Ensure each ep_data is a dict and has 'total_reward'
            total_rewards_list = [ep_data['total_reward'] for ep_data in results if isinstance(ep_data, dict) and 'total_reward' in ep_data]
            if total_rewards_list: # Only plot if there are rewards to plot
                plot_experiment_results(config_data, total_rewards_list)
            else:
                print("没有有效的总奖励数据可供绘图。")

    except ImportError as e:
        print(f"运行实验时发生导入错误: {e}")
        print("请确保 DyNN 包及其依赖已正确安装并且在 Python 路径中。")
        print(f"当前 sys.path: {sys.path}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"实验运行期间发生未捕获的异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 