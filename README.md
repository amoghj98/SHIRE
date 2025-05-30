<h2>SHIRE: Enhancing Sample Efficiency using Human Intuition in Reinforcement Learning</h2>

This repository contains code associated with SHIRE: Enhancing Sample Efficiency using Human Intuition in Reinforcement Learning. This code has most recently been tested with Python 3.10 and PyTorch 2.5.1

[//]: # (SHIRE is a Probabilistic Intuition-based Reinforcement Learning Framework.)

<h3> Introduction </h3>
The ability of neural networks to perform robotic perception and control tasks such as depth and optical flow estimation, simultaneous localization and mapping (SLAM), and automatic control has led to their widespread adoption in recent years. Deep Reinforcement Learning has been used extensively in these settings, as it does not have the unsustainable training costs associated with supervised learning. However, DeepRL suffers from poor sample efficiency, i.e., it requires a large number of environmental interactions to converge to an acceptable solution. Modern RL algorithms such as Deep Q Learning and Soft Actor-Critic attempt to remedy this shortcoming but can not provide the explainability required in applications such as autonomous robotics. Humans intuitively understand the long-time-horizon sequential tasks common in robotics. Properly using such intuition can make RL policies more explainable while enhancing their sample efficiency. In this work, we propose SHIRE, a novel framework for encoding human intuition using Probabilistic Graphical Models (PGMs) and using it in the Deep RL training pipeline to enhance sample efficiency. Our framework achieves 25-78% sample efficiency gains across the environments we evaluate at negligible overhead cost. Additionally, by teaching RL agents the encoded elementary behavior, SHIRE enhances policy explainability.

<h3>Installation</h3>
Clone this repository using:

```
git clone git@github.com:amoghj98/SHIRE.git
```

Create a conda environment using the provided yaml file as follows:

```
conda create -n intuitiveRL -f intuitiveRL.yml
```

Activate the environment using:

```
conda activate $ENV_NAME
```

<h3>Training RL Agents</h3>
RL agents can be trained using the run.bash file as follows, providing a convenient encapsulation of the otherwise cumbersome task of using the large number of available cmd-line args.:

```
bash run.bash
```

This script internally calls the script intuitiveTrain.py, which supports a large number of command-line arguments. These args are documented in the file "argConfig.py"

<h3> Citations </h3>
If you find this work useful in your research, pleas consider citing: <a href="https://arxiv.org/abs/2409.09990">Amogh Joshi, Adarsh Kosta and Kaushik Roy, "SHIRE: Enhancing Sample Efficiency using Human Intuition in Reinforcement Learning", arXiv preprint, 2024</a>

```
@misc{joshi2025shireenhancingsampleefficiency,
      title={SHIRE: Enhancing Sample Efficiency using Human Intuition in REinforcement Learning}, 
      author={Amogh Joshi and Adarsh Kumar Kosta and Kaushik Roy},
      year={2025},
      eprint={2409.09990},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.09990}, 
}
```

<h3> Authors </h3>
Amogh Joshi, Adarsh Kosta and Kaushik Roy
