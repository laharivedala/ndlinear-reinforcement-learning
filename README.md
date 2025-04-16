
# NdLinear vs Baseline: Reinforcement Learning Benchmark

This project evaluates the performance of Ensemble’s open-source module **NdLinear** against standard `nn.Linear` layers in a reinforcement learning (RL) setting. The goal is to assess whether NdLinear can deliver comparable accuracy with modular design benefits, while analyzing system-level metrics relevant to edge deployment.

## Project Overview

This notebook-based study benchmarks two Deep Q-Network (DQN) agents trained on **CartPole-v1**:

- **Baseline**: Uses standard PyTorch `nn.Linear` layers.
- **NdLinear**: Replaces all linear layers with `NdLinear` modules from [Ensemble's repo](https://github.com/ensemble-core/NdLinear).

The models are compared on:
- Total reward per episode
- Inference latency
- FLOPs per forward pass
- CPU usage
- Training time
- Model size and parameter count

## Structure

1. **Model Training**  
   - Both models are trained separately for 300 episodes each.  
   - We store and reuse trained models to avoid re-training.

2. **Performance Metrics**  
   - Reward curves (smoothed and raw)
   - Final 50-episode boxplot comparisons
   - Average training time per episode

3. **System Metrics**  
   - Inference latency (per forward pass)
   - FLOPs/MACs using `ptflops`
   - CPU usage tracking with `psutil`

4. **Evaluation Summary**  
   - Final comparison table with all metrics
   - Discussion on tradeoffs between NdLinear and Baseline

## How to Run

1. Clone this repository or open the notebook in Colab or Jupyter.
2. Install dependencies:
   ```
   pip install torch numpy matplotlib ptflops psutil
   ```
3. Run the notebook from top to bottom:
   - Models will train (unless saved `.pth` files are available).
   - All plots and metrics will be generated automatically.

## Results Summary

| Metric                    | Baseline (nn.Linear) | NdLinear              |
|---------------------------|----------------------|------------------------|
| Max Reward                | 500                  | 500                    |
| Avg Final 50 Episodes     | High                 | High                   |
| Inference Latency         | 0.0479 ms            | 0.0825 ms              |
| FLOPs (per forward pass)  | 1.15 KMac            | 1.15 KMac              |
| CPU Usage (Avg per Episode) | 60–100%             | 60–100%                |
| Model Parameters          | 898                  | 898                    |

NdLinear maintains performance parity with baseline linear layers while enabling modular extensibility for compression and deployment use cases.

## Key Takeaways

- NdLinear works as a drop-in replacement in real-time RL tasks without loss in reward.
- Inference time is slightly higher but within acceptable bounds for many edge applications.
