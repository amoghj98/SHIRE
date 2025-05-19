<h2>SHIRE: Enhancing Sample Efficiency using Human Intuition in Reinforcement Learning</h2>
SHIRE is a Probabilistic Intuition-based Reinforcement Learning Framework.<br>

<h3>Conda Environment Installation</h3>
Install the required Conda environment using this:

```
conda create -n intuitiveRL -f intuitiveRL.yml
```

<h3>Training RL Agents</h3>
RL agents can be trained using the run.bash file as follows, providing a convenient encapsulation of the otherwise cumbersome task of using the large number of available cmd-line args.:

```
bash run.bash
```

This script internally calls the script intuitiveTrain.py, which supports a large number of command-line arguments. These args are documented in the file "argConfig.py"