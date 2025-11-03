## Objectives

### 1. Benchmarking Phase
- Evaluate the performance of multiple LLM families (Qwen, Gemma, DeepSeek, LLaMA, SmolLM) on the Raspberry Pi 5.  
- Measure efficiency through metrics such as latency, token throughput, accuracy, memory footprint, and CPU/GPU utilization.  
- Identify the most balanced model considering both computational cost and real-world responsiveness.  

### 2. Simulation Phase
- Integrate the top-performing LLM into a simulated autonomous environment (e.g., robot, drone, or vehicle simulation).  
- Test real-time reasoning, adaptability, and control under constrained conditions.  
- Measure performance in decision-making latency, context comprehension, and task success rate.  

### 3. Unified Goal
Demonstrate that lightweight, optimized LLMs can act as intelligent agents for embedded or autonomous systems without requiring cloud dependency.

---

## Expected Outcomes
- A benchmark dataset and toolset for evaluating edge-compatible LLMs.  
- A quantitative comparison of major LLM families on embedded hardware.  
- A simulated demonstration showing how an LLM can guide autonomous systems.  
- A research foundation for future work in deploying AI models in robotics, IoT, and real-time edge inference.

---

## Roadmap

| Stage | Focus | Key Tasks | Deliverables |
|--------|--------|------------|---------------|
| **Phase 1: Setup & Benchmarking** | Model Evaluation | Benchmark 5 leading LLM families on Raspberry Pi 5 with unified prompts and profiling tools. | Performance CSVs, graphs, and radar plots. |
| **Phase 2: Comparative Analysis** | Normalization & Scoring | Develop normalization pipeline, composite scoring, and radar visualization for model comparison. | Ranked model results and reports. |
| **Phase 3: Selection of Top Model** | Model Selection | Identify top model based on normalized multi-metric evaluation. | “Top-5 Showdown” benchmark harness. |
| **Phase 4: Simulation Integration** | Real-world Application | Integrate best model into a simulated agent for testing decision-making and control performance. | Simulation results and performance report. |

---

## Why Benchmarking and Simulation Together

The benchmarking and the autonomous system simulation are two halves of the same pipeline.

### 1. Benchmarking: Selecting the Optimal Intelligence Core
We are testing several lightweight LLM families (such as Qwen, Gemma, and DeepSeek) on the Raspberry Pi 5 to:
- Identify which model offers the best trade-off between reasoning power, speed, and resource efficiency.  
- Build a quantitative performance map (CPU, memory, temperature, latency, tokens per second, accuracy).  
- Determine which model can run autonomously on limited hardware without cloud dependence, a key requirement for embedded or edge-based robots.

### 2. Simulation: Testing That Intelligence in a Realistic Environment
Once the best model is selected, it will be integrated into a virtual autonomous agent (robot or vehicle) inside a physics-based simulator (such as Gazebo, Webots, or PyBullet).  
This approach allows:
- Observation of how the LLM performs as a controller or reasoning agent in dynamic, uncertain conditions.  
- Measurement of decision latency, task success rate, and energy efficiency in simulation before physical deployment.  
- Demonstration of end-to-end autonomy powered entirely by a locally optimized model.

**In summary:**
- Benchmarking determines which model is technically capable on the Pi 5.  
- Simulation evaluates how that capability translates into real autonomous performance.  

Together, they form a closed validation loop:  
**Compute metrics → Choose model → Simulate tasks → Feed results back into model selection.**

---

## Hardware Setup
- Platform: Raspberry Pi 5  
- Operating System: Raspberry Pi OS 64-bit  
- Storage: Minimum 64 GB microSD  

---

## Reproduction & Setup

### 1. Prerequisites

<!--  ```bash
sudo apt update && sudo apt install -y python3 python3-pip git cmake g++ libopenblas-dev
pip install torch transformers datasets matplotlib psutil pandas -->

