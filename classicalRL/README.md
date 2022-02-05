# Implemention of classical RL on recommendation environment

This part takes advantage of the repository [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL), which implements the following model-free deep reinforcement learning (DRL) algorithms on Pytorch: 
+ **DDPG, TD3, SAC, PPO, PPO (GAE),REDQ** for continuous actions
+ **DQN, DoubleDQN, D3QN, SAC** for discrete actions
+ **QMIX, VDN; MADDPG, MAPPO, MATD3** for multi-agent environment

## Environment
+ Recommendation environment employs [VirtualTaobao](https://github.com/eyounx/VirtualTaobao.git). 
The detailed algorithm (via Pytorch) can be [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bqSnJOcAOgOtfQdZsnxT6ANj7nvyfUW0?usp=sharing).

![details of implementation](./classicalRL/VTB_stru.png)
<br/>

+ Another environment: [Recsim](https://github.com/google-research/recsim)
The detailed algorithm (via tf1) can be [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oWTYwgrDMZGAgRpoyb3_ie_SzmTAz5sQ?usp=sharing)
