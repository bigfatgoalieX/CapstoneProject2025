# Notes

## 1

1.5重力，1.5摩擦力

```
dr_reward: 4552 -> 2552
baseline_reward:6833 -> 4418
```

## 2

```
--obs_noise
```

这个观测噪音会导致baseline训练的模型表现大幅下滑，dr训练的模型表现稳定得多

## 3

```
--mass
```

当调整质量为1.5或者2.0时，baseline和dr的效果都大幅下降，或许这是一个试一试DANN方法的契机

## 4
