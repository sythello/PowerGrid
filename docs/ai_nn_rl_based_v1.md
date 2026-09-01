# `ai_nn_rl_based_v1`

`ai_nn_rl_based_v1` 是 Power Grid 的第一版离线 RL / 搜索蒸馏 AI。它保留
`ai_nn_rank_value_v1` 的公开 513 维状态特征、42 维动作特征和候选动作生成器，
但用一个 listwise Policy head 和一个多玩家 vector-Q head 替代 behavior-only
rank-value 打分。

每个原始特征的类型与中文含义见
[`ai_nn_rank_value_v1_feature_dictionary.md`](ai_nn_rank_value_v1_feature_dictionary.md)；
本版本不改变这些特征或归一化，只隔离验证 Policy/Q/搜索目标的收益。

## 支持范围

- 正式支持德国地图、3 名玩家。
- controller 名称：`ai_nn_rl_based_v1`。
- 默认 checkpoint：`src/powergrid/data/ai_models/ai_nn_rl_based_v1.npz`。
- 可用 `POWERGRID_NN_RL_BASED_CHECKPOINT` 指向其他 checkpoint。
- 在线决策只使用 Policy，不进行 session rollout，也不混合 Q 分数。
- v1 不包含在线探索、replay buffer、自动 target-network 更新或在线自博弈。

## 模型

```text
state[513] -> ReLU(128) -> ReLU(64)
                                 + action[42]
                                      |
                                 ReLU(64)
                                  /      \
                         policy_logit   q_values[6]
```

同一 decision 的全部合法候选被展平成一个批次，用 `decision_offsets` 划分候选组。
Policy 在每个组内单独做 softmax。Q 使用 `tanh` 输出最多六个玩家的最终排名价值。

状态特征中的 player slot 顺序是：当前 request actor 位于 slot 0，其余玩家按
`turn_order_position, player_id` 排序。Q 使用完全相同的顺序；搜索切换 actor 后，
先把 child Q 转回 `player_id -> value`，再按 parent slot 排列，不能直接复用下标。

最终排名标签为：

```text
rank_value = (player_count + 1 - 2 * final_place) / (player_count - 1)
```

三人局对应第一名 `1`、第二名 `0`、第三名 `-1`。不存在中间 reward，折扣率为 1。

训练损失：

```text
loss = policy_loss + q_mc_loss + q_search_loss
```

- `policy_loss`：每个 decision 等权的 listwise cross entropy。
- `q_mc_loss`：实际 behavior action 对所有玩家的终局 rank vector Huber loss。
- `q_search_loss`：搜索状态全部候选、全部玩家的 frozen-Q 搜索 label Huber loss。
- 非搜索状态 Policy target 为 deterministic one-hot。
- `legacy_soft_mix` 搜索 target 为
  `0.5 * teacher + 0.5 * softmax(search_actor_q / 0.25)`，仅用于复现早期实验。
- Stage-1 推荐 `advantage_gate`：没有足够优势时使用 teacher one-hot；存在可表示且
  `search_actor_q(best) - search_actor_q(teacher) > delta` 的新动作时，target 为
  `0.25 * teacher + 0.75 * best-action-one-hot`。

## `E_pi(Q)` 与语义搜索

本模型不训练独立 V：

```text
V_i(s) = sum_a policy_actor(a | s) * Q_i(s, a)
```

每个被抽样的 root state 都展开全部 runtime 候选，不使用 first-N 截断。搜索边先
应用一个候选，再让固定 continuation controller 推进到自然语义边界：

- auction：当前电厂拍卖结束；购买后的 pending discard chain 一并完成。
- buy resources：当前玩家结束资源购买回合。
- build houses：当前玩家结束建造回合。
- bureaucracy：本轮完整结算。
- pending decision：pending chain 完成。
- terminal：直接返回真实 rank vector。

depth 1 始终完整生成。启用 adaptive depth 2 时，最多使用 512 条语义动作边；
超过预算便丢弃本次所有 depth-2 中间结果，完整回退 depth 1。每条语义边最多执行
128 个 continuation actions，超限视为边界错误。

第一轮搜索的 leaf policy 为 `ai_deterministic` one-hot，因此 `E_pi(Q)` 只使用
teacher action 的 Q。后续迭代可改为 checkpoint Policy。对手节点按该节点行动者
自己的 Q 分量构造策略，不使用 minimax。

未来电厂牌堆是隐藏信息。所有 sibling fork 共享当前局相同牌序，作为一次 common
determinization；模型 observation 仍不包含牌序或 seed。它是一个有方差的环境样本，
不是在线可见信息。

## 数据格式

格式名：`powergrid.nn_rl_based.parquet`，version 1。每个 decision 一行：

```text
game_id / seed / decision_index / phase / decision_type
behavior_controller / continuation_controller / selected_regions
player_ids_in_slot_order / player_mask[6]
state_features[513]
candidate_jsons[]
candidate_action_features[][42]
teacher_action_index
terminal_rank_values[6]
has_search_targets
search_q_values[][6]
search_depth_used / search_nodes_evaluated / depth_2_completed
```

非搜索行的 `search_q_values` 为空；搜索行必须与候选数完全一致。数据按 game_id
确定性切分 train/validation/test，按完整游戏流式写入 zstd Parquet，每局一个 row
group；manifest 记录 schema、shard 行数/游戏数/校验和、target checkpoint SHA-256、
搜索配置和区域集合。仅保留三局完整 JSONL 示例。

德国 3 人局共有 13 组合法连续区域组合。未传区域参数时，默认按 seed 在全部合法
组合中确定性轮换；`--regions` 固定一组，或重复传 `--region-set` 限定轮换集合。

默认 shard 目标为 512 MiB。`--workers > 1` 优先使用进程池；若受限运行时禁止
semaphore 探测，则自动降级为同样有界的线程池，不改变 seed、行顺序或数据内容。

现有 `ai_nn_rank_value_v1` 数据不含完整候选组和全玩家 vector-Q，不能直接用于这版
Policy 训练。

## 命令

Bootstrap 数据与训练：

```bash
PYTHONPATH=src .venv/bin/python -m powergrid.tools.generate_nn_rl_based_dataset \
  --output artifacts/datasets/nn_rl_bootstrap \
  --games 5000 --seed-start 20001 \
  --behavior-controller ai_deterministic \
  --search-fraction 0 --workers 4

PYTHONPATH=src .venv/bin/python -m powergrid.tools.train_nn_rl_based \
  --dataset artifacts/datasets/nn_rl_bootstrap \
  --output artifacts/models/ai_nn_rl_based_bootstrap.npz \
  --epochs 20 --batch-decisions 128 --learning-rate 0.001 \
  --policy-weight 1 --q-mc-weight 1 --q-search-weight 0
```

第一轮搜索蒸馏：

```bash
PYTHONPATH=src .venv/bin/python -m powergrid.tools.generate_nn_rl_based_dataset \
  --output artifacts/datasets/nn_rl_search_1 \
  --games 5000 --seed-start 30001 \
  --behavior-controller ai_deterministic \
  --continuation-controller ai_deterministic \
  --target-checkpoint artifacts/models/ai_nn_rl_based_bootstrap.npz \
  --search-fraction 0.10 --search-depth 1 \
  --adaptive-depth-2 --max-search-nodes 512 \
  --leaf-policy deterministic --workers 4

PYTHONPATH=src .venv/bin/python -m powergrid.tools.train_nn_rl_based \
  --dataset artifacts/datasets/nn_rl_search_1 \
  --init-checkpoint artifacts/models/ai_nn_rl_based_bootstrap.npz \
  --output src/powergrid/data/ai_models/ai_nn_rl_based_v1.npz \
  --epochs 20 --batch-decisions 128 --learning-rate 0.001 \
  --policy-weight 1 --q-mc-weight 1 --q-search-weight 1 \
  --search-policy-mix 0.5 --search-temperature 0.25
```

保守 Stage-1 Policy improvement 复用同一 search 数据和 Stage-0 checkpoint，不改变
Parquet schema。每个 epoch 保留全部 searched rows，并确定性抽取等量 non-search
BC anchor；validation/test 仍按自然分布评估。分别用 `--min-search-advantage` 的
`0.05`、`0.10`、`0.20` 训练三份候选，其余命令参数如下：

```bash
PYTHONPATH=src .venv/bin/python -m powergrid.tools.train_nn_rl_based \
  --dataset artifacts/datasets/nn_rl_search_1 \
  --init-checkpoint artifacts/models/ai_nn_rl_based_bootstrap.npz \
  --output artifacts/models/ai_nn_rl_based_v1_gate_d010.npz \
  --epochs 20 --batch-decisions 128 --learning-rate 0.001 \
  --policy-weight 1 --q-mc-weight 1 --q-search-weight 1 \
  --policy-target-mode advantage_gate \
  --improved-action-weight 0.75 \
  --min-search-advantage 0.10 \
  --training-sampling balanced_search
```

在 `advantage_gate` 下，搜索 label 的 actor slot 0 决定候选优势。与 teacher 的 42 维
action feature 完全相同的候选不可由当前网络区分，不允许触发 target 切换。相同最高
Q 按候选顺序稳定选择。checkpoint 记录 target/sampling 配置、源 searched/non-search
行数，以及每个 epoch 实际训练和 accepted improvement 数量。

新增验证指标：`accepted_improvement_rate`、`accepted_policy_top1_accuracy`、
`searched_fallback_teacher_accuracy`、`non_search_teacher_accuracy` 和
`gated_policy_cross_entropy`。旧的 teacher `policy_accuracy` 继续保留，但它会按设计
把 accepted improvement 计为不一致，不能单独用来判断新 Policy 是否学对。

验证和对战：

```bash
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_nn_rank_value tests.test_nn_rl_based -v

PYTHONPATH=src .venv/bin/python -m powergrid.tools.validate_nn_rl_based \
  --section all --games 100 \
  --output artifacts/validation/ai_nn_rl_based_v1.json

# 对正式两阶段数据/checkpoint执行发布模型门槛；失败时返回非零状态。
PYTHONPATH=src .venv/bin/python -m powergrid.tools.validate_nn_rl_based \
  --section training \
  --dataset artifacts/datasets/nn_rl_bootstrap \
  --search-dataset artifacts/datasets/nn_rl_search_1 \
  --bootstrap-checkpoint artifacts/models/ai_nn_rl_based_bootstrap.npz \
  --checkpoint src/powergrid/data/ai_models/ai_nn_rl_based_v1.npz \
  --calibration-roots 50 --enforce-acceptance \
  --output artifacts/validation/ai_nn_rl_based_v1_model_acceptance.json

PYTHONPATH=src POWERGRID_NN_RL_BASED_CHECKPOINT=src/powergrid/data/ai_models/ai_nn_rl_based_v1.npz \
  .venv/bin/python -m powergrid.tools.evaluate_ai_ratings \
  --controllers ai_nn_rl_based_v1 ai_deterministic \
  --players 3 --games-per-lineup 200 --seed-start 50001
```

通用 Elo 工具默认根据 game seed 从德国 3 人局的全部合法连续区域组合中可复现地
随机采样；相同 seed 的两种换座 lineup 使用同一区域组合。用 `--regions` 可显式
固定一组区域，或用 `--region-sampling-seed` 改变区域调度。

## 发布验收门槛

`validate_nn_rl_based --section training` 会在 game-exclusive validation/test 上流式
计算以下指标；`--enforce-acceptance` 使任一失败返回非零状态：

- bootstrap Policy 对 teacher 的整体 top-1 至少 95%，每个高频 decision type 至少
  90%；
- Q-MC MSE 比训练集各 player slot 的均值 baseline 至少降低 10%；
- final checkpoint 对相同 search label 的 Q MAE 比 bootstrap 至少降低 20%，且
  search-policy target 的交叉熵下降；
- 50 个独立 seed 的校准 root 对所有候选继续 deterministic rollout 到终局，Q 的
  pairwise ordering accuracy 至少 55%。

强度 section 按跨 controller seat pair 统计 `wins + 0.5 * draws`，并按整局 bootstrap
2000 次给出 95% CI。少于 400 局明确标记 `SMOKE_ONLY`；正式发布还要求 score 至少
0.50、平均名次不差于 deterministic、CI 下界至少 0.48。

## 当前验证结果

### 2026-08-30 正式 5000 + 5000 局实验

正式实验已经完成数据生成、训练、模型验收和 400 局强度验收。结论是：数据与训练
流水线可用，Q 指标达到目标，但 Policy-only 控制器没有达到发布强度，因此没有把
checkpoint 复制到 `src/powergrid/data/ai_models/ai_nn_rl_based_v1.npz`。可复现实验产物
保留在 `artifacts/`，并由 `.gitignore` 排除。

Bootstrap 数据：

- 5000 局、2,560,925 decisions，德国 3 人全部 13 组合法连续区域组合；
- train/validation/test 分别为 3982/535/483 局和
  2,040,164/273,787/246,974 行；
- 3 个 zstd Parquet shard，共 200.9 MiB，约 82.3 bytes/decision；
- 4 进程生成耗时 616.0 秒；manifest 行数、游戏数、schema 与 SHA-256 全部通过。

Search 数据：

- 5000 局、2,563,665 decisions，其中 255,732 个搜索 decision，实际抽样率 9.98%；
- 共展开 7,701,659 条语义动作边；全部搜索点都完整覆盖全部候选；
- 255,732/255,732 个搜索点完成 depth 2，512 边预算下完成率 100%，无 partial-depth
  label；
- train/validation/test 分别为 3994/485/521 局和
  2,046,373/250,027/267,265 行；
- 3 个 zstd Parquet shard，共 216.9 MiB；4 进程生成耗时 6989.8 秒；manifest 和
  checksum 全部通过。

Bootstrap 20 epoch 训练耗时 7762.0 秒。validation 的 deterministic top-1 为
98.32%，Q-MC MAE 为 0.4317；Q-MC MSE 相对 player-slot 均值 baseline 降低
42.57%。第一版 search 模型 20 epoch 训练耗时 6464.5 秒，validation 指标为：

- Policy accuracy 98.60%；
- Q-MC MAE 0.4376；
- search-Q MAE 0.1508，相对 bootstrap 的 0.1887 改善 19.18%，略低于 20% 门槛；
- 50 个终局 rollout 校准根上的动作 pairwise ordering accuracy 为 74.14%。

按失败后的预设调参顺序，从该模型继续微调 5 epoch，将 `search_policy_mix` 从 0.5
降到 0.25，耗时 1804.8 秒。最终候选位于
`artifacts/models/ai_nn_rl_based_v1_candidate_mix025.npz`，validation 指标为：

- Policy accuracy 98.56%，search-policy target CE 0.1381；
- Q-MC MAE 0.4281；
- search-Q MAE 0.1470，相对 bootstrap 改善 22.11%，达到 20% 门槛；
- 50 个校准根全部完成，pairwise ordering accuracy 65.52%，达到 55% 门槛。

精确动作分类验收发现一个 schema-v1 已知限制：bureaucracy validation 的 top-1
只有 83.88%。这不是增加 epoch 可以修复的欠拟合。18,351 个 bureaucracy 状态中，
7782 个（42.40%）包含两个或更多完全相同的 42 维 action vector；3151 个 teacher
动作落在碰撞组中，2823 个 teacher 不是稳定 tie-break 的第一个候选。因此精确 JSON
动作一致率的理论上限只有 84.62%。碰撞动作只是在“由哪台同产能电厂执行”上不同，
而 `resolve_bureaucracy` 最终只使用总资源消耗和总发电城市数；这些碰撞动作产生相同
后继状态。后续应选择以下一种最小修复，再重新定义该项验收：

- 在候选生成时按后继状态语义 canonicalize 等价 bureaucracy 动作；或
- 保留候选，但把 Policy agreement 改成语义等价一致率；若未来规则让电厂身份影响
  后继状态，再升级 action feature schema 编码逐电厂计划。

400 局强度验收均正常完成，但两个候选都未达发布线：

| checkpoint | RL pairwise score | RL 平均名次 | deterministic 平均名次 | game-bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: |
| mix 0.50，20 epoch | 0.48125 | 2.0250 | 1.9750 | [0.4425, 0.5225] |
| mix 0.25，再微调 5 epoch | 0.48625 | 2.0183 | 1.9817 | [0.44625, 0.5275] |

降低 mix 有小幅改善，但仍未满足 score 至少 0.50、平均名次不差于 deterministic、
CI 下界至少 0.48 三个强度条件。当前目标本身也限制了可实现的 Policy improvement：
当 `search_policy_mix <= 0.5` 时，teacher one-hot 的 target 概率始终不小于任何 Q-soft
动作，所以 target 的 top-1 不会主动从 teacher 切换到 Q 更优动作。下一次实验应先
修正这个目标（例如对置信 Q 差值使用 Q-greedy/advantage-gated target，或让 mix 可超过
0.5 并单独约束 KL/behavior cloning），再复用现有数据训练；继续在 0.5 以下扫描 mix
或只增加 epoch 不太可能带来稳定胜率提升。

机器可读结果：

- `artifacts/validation/ai_nn_rl_based_v1_model_acceptance.json`
- `artifacts/validation/ai_nn_rl_based_v1_strength.json`
- `artifacts/validation/ai_nn_rl_based_v1_model_acceptance_mix025.json`
- `artifacts/validation/ai_nn_rl_based_v1_strength_mix025.json`

### 2026-08-31 Stage-1 保守策略改进实验

本轮从原始 Stage-0 checkpoint `artifacts/models/ai_nn_rl_based_bootstrap.npz`
分别重新训练三档 advantage gate，不继承上一版 soft-mix checkpoint。数据仍为现有
`artifacts/datasets/nn_rl_search_1`，没有重新生成或修改 schema。

实现行为：

- `--policy-target-mode advantage_gate` 直接使用搜索 label 的 actor slot 0；
- 排除 teacher 本身和与 teacher 42 维 action feature 完全相同的不可区分候选；
- 稳定选择剩余候选中 Q 最高者，仅在 `Q(best) - Q(teacher) > delta` 时接受；
- 接受后的单一 Policy target 为 `0.25 * teacher + 0.75 * best`，否则为 teacher
  one-hot；Q-MC/Q-search loss 和网络结构不变；
- `--training-sampling balanced_search` 每个 epoch 恰好使用全部 204,210 个 searched
  row，并确定性抽取 204,210 个 non-search anchor；20 个 epoch 均验证为严格 1:1；
- checkpoint 记录 target/sampling 配置、源数据计数，以及逐 epoch 的
  searched/anchor/accepted 数量；validation/test 保持自然分布。

三档均训练 20 epoch，batch 128，学习率 0.001，Policy/Q-MC/Q-search 权重均为 1。
总训练及全分割评估耗时分别为 1655.3/1851.8/1904.8 秒。validation 指标如下；
Q-search MAE 已按全部有效候选/player 元素精确加权：

| delta | accepted rows/rate | accepted best top-1 | fallback teacher | non-search teacher | 相对 Stage 0 non-search | Q-search MAE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 6215 / 25.13% | 5.37% | 97.30% | 96.15% | -2.09 pp，淘汰 | 0.1508 |
| 0.10 | 4330 / 17.51% | 4.53% | 97.43% | 96.88% | -1.36 pp，淘汰 | 0.1495 |
| 0.20 | 2409 / 9.74% | 2.08% | 98.00% | 97.95% | -0.29 pp，通过 | 0.1531 |

Stage-0 在同一 search validation split 的 non-search teacher accuracy 为 98.24%。按
预先固定的“下降不超过 1 个百分点”规则，只有 `delta=0.20` 能进入模型合格集合。
它的 0.1531 也低于 `0.1470 * 1.05 = 0.15435` 的 Q-search 上限。正式验证进一步确认：

- Stage-0 对 `delta=0.20` accepted target 的 top-1 为 0%，新模型为 2.08%；
- fallback/non-search teacher 回归均小于 1 个百分点；
- 50/50 个 terminal-rollout calibration roots 完成，Q pairwise ordering accuracy
  为 74.14%；
- bootstrap Q-MC MSE 相对 player-slot 均值基线降低 42.57%。

集中测试覆盖 gate fallback/阈值/feature collision/stable tie-break、legacy target
逐元素回归、accepted/fallback 合成过拟合、1:1 流式采样和 checkpoint round-trip。
加入 paired rollout 和 deterministic-suite 聚合测试后，RL 测试 17/17 通过；与
rank-value 合并回归 25/25 通过。

三档分别使用独立 seed 做 400 局、两个 seat-balanced lineup 的强度筛选。下表的
score 是全部 800 个跨 controller seat-pair 的 `(wins + 0.5 * draws) / comparisons`；
CI 按完整游戏 bootstrap 2000 次：

| delta | RL pairwise score | game-bootstrap 95% CI | RL 平均名次 | deterministic 平均名次 | 模型门槛 | 结论 |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0.05 | 0.6481 | [0.6131, 0.6838] | 1.8017 | 2.1967 | non-search 失败 | 淘汰 |
| 0.10 | 0.6031 | [0.5650, 0.6413] | 1.8600 | 2.1333 | non-search 失败 | 淘汰 |
| 0.20 | 0.5000 | [0.4612, 0.5413] | 2.0000 | 2.0000 | 通过 | 未 beat baseline |

这个结果说明保守阈值存在清晰的稳定性/改进强度权衡：较小阈值在 400 局上显著变强，
但共享网络也在大量 non-search 状态发生超过允许范围的行为漂移；唯一保持 Stage-0
行为的 0.20 档没有表现出净强度优势。三档没有任何一个同时通过模型门槛和强度门槛，
所以严格按照预注册流程没有运行 2000 局最终测试，也没有复制默认 checkpoint 到
`src/powergrid/data/ai_models/ai_nn_rl_based_v1.npz`。下一阶段应改造 paired terminal
rollout label，而不是继续复用同一测试 seed 调 gate/mix/epoch。

机器可读产物（均位于 gitignored `artifacts/`）：

- `models/ai_nn_rl_based_v1_gate_d005.npz`
- `models/ai_nn_rl_based_v1_gate_d010.npz`
- `models/ai_nn_rl_based_v1_gate_d020.npz`
- `ai_ratings/ai_nn_rl_based_v1_gate_d005_vs_deterministic_400.json`
- `ai_ratings/ai_nn_rl_based_v1_gate_d010_vs_deterministic_400.json`
- `ai_ratings/ai_nn_rl_based_v1_gate_d020_vs_deterministic_400.json`
- `validation/ai_nn_rl_based_v1_gate_d020_model_acceptance.json`

正式 validator 的总体 `all_checks_pass` 仍会被已记录的 schema-v1 bureaucracy
精确动作碰撞置为 false：Stage-0 在该类型的精确 top-1 为 83.88%。本轮新增的所有
advantage-gate 检查均通过，这一旧限制不影响上述“本轮无可发布候选”的结论。

#### 基于 baseline continuation 的 paired terminal rollout 复验

为避免继续用 deterministic action agreement 代替实际收益，新增了独立评估工具：

```bash
PYTHONPATH=src .venv/bin/python -m powergrid.tools.evaluate_nn_rl_paired_rollouts \
  --checkpoint d005=artifacts/models/ai_nn_rl_based_v1_gate_d005.npz \
  --checkpoint d010=artifacts/models/ai_nn_rl_based_v1_gate_d010.npz \
  --checkpoint d020=artifacts/models/ai_nn_rl_based_v1_gate_d020.npz \
  --games 100 --seed-start 70001 --bootstrap-samples 2000 \
  --output artifacts/validation/ai_nn_rl_based_v1_paired_rollouts_100.json
```

评价口径保持窄而直接：在全新 seed 的纯 deterministic 轨迹上检查每个 decision；当
checkpoint 与 baseline 的实际 intent 不同时，从同一 `GameSession`、同一隐藏牌序
fork 两个分支，分别执行 RL/baseline 首动作，之后全部玩家都使用
`ai_deterministic` 到终局。优势为根行动者的
`RL normalized terminal rank - baseline normalized terminal rank`。CI 以源游戏为
cluster 做 2000 次 bootstrap，避免把同一局的多个差异状态当作完全独立样本。

100 局全部完成，共 51,608 个 baseline decisions，按局循环覆盖全部 13 个合法区域
组合；5333 个实际 rollout 分支耗时 180.2 秒。结果如下：

| delta | 差异状态/比例 | 改善/持平/变差 | harmful rate | mean advantage | mean 95% CI | paired score | score 95% CI | 结论 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.05 | 1901 / 3.68% | 289/1411/201 | 10.57% | 0.0684 | [0.0232, 0.1104] | 0.5231 | [0.5077, 0.5375] | 显著正向 |
| 0.10 | 1500 / 2.91% | 209/1161/130 | 8.67% | 0.0713 | [0.0275, 0.1159] | 0.5263 | [0.5120, 0.5407] | 显著正向 |
| 0.20 | 984 / 1.91% | 89/833/62 | 6.30% | 0.0386 | [-0.0105, 0.0913] | 0.5137 | [0.4976, 0.5309] | 点估计正向，证据不足 |

将未改变的 baseline decisions 视为零 advantage 后，每个 baseline decision 的平均
收益及 95% CI 分别为：`delta=0.05` 0.00252 `[0.00087, 0.00405]`、
`delta=0.10` 0.00207 `[0.00081, 0.00333]`、`delta=0.20` 0.00074
`[-0.00019, 0.00177]`。因此 paired rollout 与先前 end-to-end 结果一致地确认：
0.05 和 0.10 的行为漂移整体是有益泛化，不能仅因 non-search exact agreement 下降
而淘汰；0.20 过于保守，目前无法确认其收益。

收益主要来自 `auction_bid` 和 `auction_start`。0.05 的 `buy_resources` 平均 advantage
为 -0.0342；0.10 的资源购买接近中性（0.0106），且 harmful rate 更低，因此若强调
收益与保守性的平衡，0.10 是三档中更合适的后续候选。此复验估计的是 baseline
状态分布上的单步策略改进 `Q^baseline(s,a_rl)-Q^baseline(s,a_det)`；它不替代已有的
端到端对局测试，也未自动发布 checkpoint。

#### δ=0.10 多 deterministic 策略验收与当前版本

后续版本固定以 `delta=0.10` 为候选，并将版本优劣的主要门槛改为：paired rollout
score 大于 0.50，且对每一种 deterministic 策略的 end-to-end pairwise score 均
严格大于 0.50。新增的 suite 工具不改通用 Elo 模块，而是隔离地调度两种
seat-balanced lineup，并按 seed offset 循环全部 13 个合法区域组合：

```bash
PYTHONPATH=src .venv/bin/python -m powergrid.tools.evaluate_nn_rl_deterministic_suite \
  --checkpoint artifacts/models/ai_nn_rl_based_v1_gate_d010.npz \
  --games-per-lineup 200 --seed-start 80001 --bootstrap-samples 2000 \
  --paired-rollout-report \
    artifacts/validation/ai_nn_rl_based_v1_paired_rollouts_100.json \
  --paired-checkpoint-label d010 \
  --output \
    artifacts/validation/ai_nn_rl_based_v1_d010_deterministic_suite_400_each.json
```

每个对手使用同一组 200 个全新 seed，并分别运行 `(RL, RL, opponent)` 与
`(RL, opponent, opponent)`，共 400 局/800 个跨 controller seat-pair。1600 局均
正常完成：

| opponent | RL wins/draws/losses | pairwise score | game-bootstrap 95% CI | RL/opponent 平均名次 |
| --- | ---: | ---: | ---: | ---: |
| `ai_deterministic` | 497/2/301 | 0.6225 | [0.5844, 0.6606] | 1.8350 / 2.1600 |
| `ai_deterministic_efficiency` | 556/0/244 | 0.6950 | [0.6575, 0.7338] | 1.7400 / 2.2583 |
| `ai_deterministic_expansion` | 444/2/354 | 0.5563 | [0.5162, 0.5925] | 1.9200 / 2.0717 |
| `ai_deterministic_reserve` | 416/0/384 | 0.5200 | [0.4838, 0.5587] | 1.9733 / 2.0267 |

四个 point score 均大于 0.50，且独立 paired rollout score 为 0.5263、95% CI
`[0.5120, 0.5407]`，因此满足预先固定的“保留为当前最优”标准。reserve 的 CI 仍
跨过 0.50，所以准确表述是“通过点估计门槛”，而不是“已统计显著战胜 reserve”。

δ=0.10 已发布为当前默认 checkpoint：

```text
src/powergrid/data/ai_models/ai_nn_rl_based_v1.npz
SHA-256 db45c1976e7da5762e19a4710e03ab51e29457a46578c1fd41fadf775e1f079e
```

发布 checkpoint 的 Policy/Q 预测与来源
`artifacts/models/ai_nn_rl_based_v1_gate_d010.npz` 逐元素一致；metadata 记录
`selected_delta=0.10`、`release_status=current_best`、来源与两份验证报告的
SHA-256，以及 reserve CI 限制。注册名仍为 `ai_nn_rl_based_v1`，未修改 `ai`
默认 alias。

### 2026-08-28 smoke/组件验证

2026-08-28 的实现 smoke/组件验证使用 5 局、2 个小模型 epoch；这不是最终 5000 局
模型或强度验收：

- 集中单元测试：`tests.test_nn_rl_based` 10/10 通过；相关非 GUI 回归 67/67 通过。
- 合成模型：Policy accuracy `4.17% -> 91.67%`；Q MC MAE
  `0.6881 -> 0.0927`。
- 搜索：opening 4 个候选全部覆盖；512 预算完成 depth 2（24 nodes）；低预算完整
  回退 depth 1。
- 数据：5 局、2567 decisions、5 组不同合法区域、3 个 JSONL 示例；game-exclusive
  row-group、shard/schema/checksum 验证通过。
- 在启用 `tracemalloc` 的验证环境中：286.8 decisions/s、峰值 54.2 MiB、约
  84.4 Parquet bytes/decision。
- 搜索数据 smoke：50 个搜索 decision、1101 个语义边、571.7 search-nodes/s；
  512 预算下 depth-2 完成率 100%。
- Policy-only opening 决策 p95 延迟约 0.315 ms；其中候选编码约 0.241 ms、模型
  推理约 0.047 ms。
- 小模型 2 epoch 的训练 Policy accuracy 为 68.41%；不用于发布。
- 另一次端到端 smoke 使用 30 局 bootstrap + 10 局 search：bootstrap 共 15,146
  decisions，search 共 5,147 decisions/510 搜索点/14,464 nodes，depth-2 完成率
  100%。搜索训练后 validation search-Q MAE `0.3901 -> 0.3289`（改善 15.7%，未达
  20% 发布线），search-policy CE `0.5126 -> 0.4884`。
- 该小模型的 20 局强度 smoke 为 RL pairwise score 0.40、game-bootstrap 95% CI
  `[0.225, 0.575]`，平均名次 2.133（deterministic 1.867），因此明确标为
  `SMOKE_ONLY` 且不发布。
- 当时尚未运行 400 局强度验收；正式结果见上一节，仍未发布 checkpoint。

固定 100 局、多区域、`workers=4` 基准的候选分布如下：

| decision type | mean | p50 | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| auction bid | 1.9935 | 2 | 2 | 2 | 2 |
| auction start | 4.4695 | 5 | 7 | 7 | 7 |
| build houses | 4.6287 | 2 | 12 | 15 | 22 |
| bureaucracy | 7.0056 | 8 | 8 | 10 | 24 |
| buy resources | 6.9478 | 6 | 12 | 14 | 25 |
| discard hybrid resources | 2.0000 | 2 | 2 | 2 | 2 |
| discard power plant | 4.0000 | 4 | 4 | 4 | 4 |

同一固定基准的 bootstrap 为 100 局 / 51,007 decisions：4.67 games/s、2383.7
decisions/s、82.4 bytes/decision，进程自身峰值 RSS 251.0 MiB；在途完整游戏最多
`2 * workers = 8`，低于 1 GiB 且不随累计局数增长。search 为 100 局 / 50,566
decisions / 4857 搜索点 / 150,274 nodes：0.386 games/s、195.1 decisions/s、579.9
search-nodes/s、89.1 bytes/decision；全部候选覆盖率和 512 预算下 depth-2 完成率均
为 100%。

smoke 的完整机器可读结果位于 `artifacts/validation/ai_nn_rl_based_v1.json`
（artifacts 默认不纳入 Git）。正式 5000 局两阶段训练和 400 局强度验收已在
2026-08-30 完成，结果未达到发布标准。
