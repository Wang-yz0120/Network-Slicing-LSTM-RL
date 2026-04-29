# from gym.envs.registration import register

# from .ran_slice import RanSlice

# register(
#     id='RanSlice-v1',
#     entry_point='gym_ran_slice:RanSlice'
# )
from gymnasium.envs.registration import register

from .ran_slice import RanSlice

# 推荐：带命名空间
register(
    id="gym_ran_slice/RanSlice-v1",             # 注意这里用斜杠
    entry_point="gym_ran_slice:RanSlice",
)
