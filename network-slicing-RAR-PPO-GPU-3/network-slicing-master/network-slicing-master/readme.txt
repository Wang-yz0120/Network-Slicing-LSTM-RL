首次将工作一的实验代码加入本地仓库 2026.4.28

代码修改工作总结如下：
1.总体采用RL+LSTM的方案对包含eMBB以及mMTC切片的5G网络场景进行动态prb分配
2.引入prb的占用机制，每个切片对应两个动作，包含仅在该step生效的prb以及跨多个step占用的prb。状态空间与动作空间改为：状态空间为各个eMBB切片与mMTC切片的原始状态(eMBB十个维度，mMTC三个维度) + 各个切片的跨step prb占用量 + LSTM预测预测结果(eMBB的cbr_traffic、vbr_traffic，mMTC的new_devices) + 系统未分配prb量；动作空间为各个切片的跨step的prb占用量 + 各个切片的只在当前step的prb量 + prb预留量
3.在参考论文的基础上引入流量的波动机制：流量在normal平常态以及突发态之间进行有一定随机性与规律性的切换
4.为了突出LSTM的预测结果，引入动作延迟机制，将step t的动作放在step t+1执行
5.解决了部分已知bug，如prb大于200时出现报错，计算动作时采用的remaining_prb为上一step的值的问题