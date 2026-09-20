`ai_nn_rl_based_v1` 训练审计，2026-09-19

审计对象是本地工作区的实际实现、默认 checkpoint、Parquet 数据与历史实验产物。审计前工作区已有未提交修改，本次未改动训练/推理代码或默认模型；新增内容仅在本审计目录。未重新训练任何模型。

**1. 总体判断**

当前已发布模型的第一次策略改进是有实测收益的；后续“从这个模型继续改进”的方法尚未可靠成立。主要瓶颈是：少量、带噪声、条件于旧策略的动作优势，被转成整个共享网络的监督微调，最终产生的 argmax 策略变化与已验证的改进之间缺少稳定联系。

这比笼统的“RL 不够强”“网络太小”“reward 太稀疏”更贴近已有证据。现阶段也不能把全部责任归给 Q loss、行为漂移或隐藏牌序中的任意一个。它们需要分开验证。

**2. 实际训练与推理链路**

| 环节 | 当前实现 | 判断 |
| --- | --- | --- |
| 范围 | Germany / 3 players；13 组合法区域组合 | 适用范围明确，不能外推到其他地图、人数或任意对手 |
| 输入 | 520 维状态、52 维动作；actor 固定 slot 0，其余按行动顺序排列 | observation 不输入隐藏牌序或 seed；搜索换 actor 后按 player ID 重排 Q |
| 候选 | 每个 root 完整枚举 runtime 候选；最低幅度加价、单城市建造、定序资源购买、发电组合 | “全部候选”不等于规则允许的所有整数报价等原始动作 |
| 网络 | state 128→64；拼接 action 后 64；Policy 标量与 tanh Q[6] 共享表示 | Policy 与 Q 的梯度共同更新 state/candidate trunk |
| 奖励 | 终局 normalized rank，三人通常 +1/0/-1；gamma=1 | 与平均名次/两两比较得分基本一致；不等于只优化第一名概率 |
| Bootstrap | deterministic 行为克隆 + 实际所选动作的终局 Q-MC | 先学会基线行为与长程价值 |
| 旧 Stage-1 | 冻结 Q 的语义搜索、depth 1 全覆盖、预算允许才接受完整 depth 2；advantage gate 蒸馏 | leaf Q 有自举误差；depth 2 中间节点使用 teacher/soft-Q 混合，不是严格固定旧策略的 Q |
| 新 Paired-MC | 冻结 incumbent 生成轨迹、同一隐藏牌序下枚举各首动作、同一冻结策略续局到终局 | 去掉 leaf-Q 自举，但单个隐藏实现的收益方差仍大 |
| K=8/16 确认 | 单次筛选；新的隐藏顺序配对续局；均值减 1.645×SE 大于零才可成为 Policy 改进 | 有效减少不稳健的标签，但只是近似置信筛选 |
| 训练 | 每 decision 等权 CE；所选动作 Q-MC；搜索全候选/有效 player 元素 Q-search；Adam | Q-search 按候选元素数平均，候选多的 root 对该项权重更高；不等于 Policy 的每 root 等权 |
| 采样 | balanced_search 每 epoch 全部 searched + 等量 non-search | 强调搜索标签；小数据时大部分原策略状态每轮不会作为 anchor 出现 |
| 上线 | 只选最大 Policy logit，稳定候选顺序打破平局 | Q 不参与上线选择，Policy softmax 的概率变化只有跨过 argmax 边界才改变动作 |
| 数据隔离 | 按完整 game_id 划分 train/validation/test，流式 Parquet，schema/checksum | 未发现明显逐行随机切分造成的同局泄漏 |
| 评估 | 旧 suite 两种人数构成；新 duel 六种座位排列、按 seed 聚类 bootstrap | 新 duel 更适合决定是否替换 incumbent |

代码依据：[controller](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/nn_rl_based/controller.py:99)、[共享梯度](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/nn_rl_based/model.py:245)、[训练采样](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/nn_rl_based/training.py:344)、[候选生成](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/nn_rank_value/candidates.py:43)。

训练数据中的 `teacher_action_index` 在新自博弈路径实际指 behavior incumbent 的动作，不再始终指 deterministic。新增代码与历史默认模型必须区分：默认模型仍是 9 月 6 日的 semantic-search / advantage_gate 模型，未升级成 Paired-MC 模型。

默认 checkpoint SHA-256：`466d8d507c406215014f222de8d5720ebd96bcbb8465b3b7e1d8061284a5bc0c`。

**3. 历史实验能说明什么**

| 阶段 | 主要证据 | 可支持的结论 |
| --- | --- | --- |
| 8/30 schema-v1 soft mix | mix=.50 对 deterministic 0.48125；mix=.25 为 0.48625 | 拟合 Q、提高 teacher agreement 不等于提高策略强度；mix≤.5 从目标构造上保留 teacher 为最高概率动作 |
| 8/31 gate | δ=.05/.10/.20：0.6481/0.6031/0.5000 | gate 突破了旧 target 的限制；non-search agreement 的下降不能直接当成有害漂移 |
| 9/1 schema-v1 δ=.10 | 四对手分数 .6225/.6950/.5563/.5200 | reserve 只有点估计通过，CI 跨 .5，不能称全部显著取胜 |
| 9/6 schema-v2 当前发布版 | 固定种子四对手 .704375/.7900/.59125/.8625；新种子 .7150/.81375/.5975/.8850 | 当前模型相对这四个对手有真实、可重复收益 |
| 9/18 K=0，500 局数据 | gate075=.4387；weighted075=.4667；weighted090=.4283，相对 incumbent | 未发现可晋级候选；weighted075 CI 包含 .5，不能说三者全部显著更差 |
| 9/19 同 100 局 K=0/8/16 | 相对 incumbent .4058/.4017/.4250，三个 CI 上界均小于 .5 | 这组训练候选显著退化 |
| K=16 对 K=0 | .5725，95% CI [.5383,.6067] | 确认标签改善了这次更新的效果，尚不足以超过 incumbent |

不同训练规模与 seed 的 score 不能直接相减当作某单项改动的因果收益。schema-v1/v2 涉及状态、动作和资源决策实现变更，更不能拿跨 schema 提升估算单独 RL 算法贡献。

同一 schema-v2、同一 suite，Stage-0 对四对手为 `.47625/.683125/.5525/.77625`，Stage-1 为 `.704375/.79/.59125/.8625`。因此，尤其对 canonical deterministic 的提升，并非仅仅因为模型模仿了一个原本强于其他变体的 teacher。独立 paired rollout：Stage-0 score `.49909`，Stage-1 `.55122`，也支持第一次策略改进有效。

历史依据：[主设计记录](/Users/mac/Desktop/syt/Projects/PowerGrid/docs/ai_nn_rl_based_v1.md)、[9/18 原始总结](/Users/mac/Desktop/syt/Projects/PowerGrid/artifacts/experiments/paired_mc_pi_20260918/RESULTS.md)、[9/19 原始总结](/Users/mac/Desktop/syt/Projects/PowerGrid/artifacts/experiments/paired_mc_confirm_20260919/RESULTS.md)。

**4. 本次重新计算的训练诊断**

K=16 的训练集有 40,322 个 decisions，但实际只有 837 个 searched roots、64 个 accepted roots；每 epoch 训练 837+837=1,674 个 decisions。64 个改进点占 epoch 的 3.82%，non-search anchor 每轮只覆盖 39,485 条中的约 2.12%。只有 9 个 accepted 训练点属于 auction_start。

当前 K=16 模型在完整 validation 的前向复核：

| 指标 | 实测 |
| --- | ---: |
| 验证 decisions | 4,549 |
| incumbent 动作与数据 teacher 不一致 | 0 |
| searched / accepted roots | 97 / 6 |
| 新模型改动动作 | 65（1.43%） |
| non-search 动作改动 | 62 / 4,452（1.39%） |
| 在 accepted root 上改动且选择 confirmed 动作 | 1 |
| 改动中与自己 Q 排序相反 | 39 / 65 |
| 平均 KL(incumbent || candidate) | 0.06477 nats |
| incumbent 自己对 hard teacher 的平均 CE | 0.13348 |

65 个变化中的大多数缺少同状态的确认标签。这不能推出它们全部有害，因为 non-search 本就没有反事实标签；它说明“accepted 目标训练得更准”与“实际上线改了哪些动作”是两回事。新的评估应直接追踪后者。

旧模型在自身 hard-one-hot 数据上的 CE 不为零，故 hard BC 并非严格保留原概率分布。但“不为零”和“棋力必然下降”也不同；下文新增直接对战专门检验这一点。

K=16 validation 只有 6 个 accepted roots，所谓 accepted top-1 16.7% 就是 1/6；K=8 的 25% 是 1/4。不能用这两个百分比宣称 K=8 的学习能力更强。

当前发布模型的 accepted top-1 也很低：训练 6.996%、验证 5.474%；验证 Q 排序校准 56.43% 只来自 50 roots 的 140 对非平局动作。它是有用诊断，不足以证明逐个改进标签可信。

机器可读复核：[diagnostics.json](/Users/mac/Desktop/syt/Projects/PowerGrid/artifacts/audits/ai_nn_rl_based_v1_20260919/diagnostics.json)。

**5. 需要区分的几个问题**

(a) 确认只进入 Policy 标签，Q-search 仍吃原始单次牌序的全动作结果。[search.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/nn_rl_based/search.py:339) 返回原始 `rows`，额外返回 confirmed advantages/mask；[model.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/nn_rl_based/model.py:216) 对全部搜索候选训练 Q，再通过共享 trunk 影响 Policy。因此“确认过的 Policy”不等于“所有更新梯度都基于确认过的证据”。但一个单次 MC 值也并非自动是错误或有偏的 Q 标签；问题是方差、分布变化和共享表示干扰，不能仅由 Policy/Q 排序不一致就证明因果。

(b) 旧策略下一步替换的优势，不保证多个替换组合后的策略更强。标签续局的所有玩家使用冻结 incumbent；上线时一到两名玩家切换成新模型，自己的后续动作和对手的应对都会改变。三人博弈里，同时更新多个玩家尤其没有简单的全局单调提升保证。保持 direct incumbent duel 是正确的。

(c) `advantage_weighted` 又出现了 argmax 目标稀释。[实现](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/nn_rl_based/model.py:697) 保留 teacher .25，把 .75 分给多个动作。若四个优势相同的动作分到 .1875，teacher 仍为 target top-1。实测 K=0 的 500 局训练数据中，775 个 accepted roots 有 108 个 teacher 仍至少并列最大（94 个稳定 argmax 仍选 teacher）；K=16 的 64 个中对应 7 个（稳定 argmax 仍 teacher 为 5 个）。这解释了 accepted 不一定表示一个可执行的 target switch，但其占比也不足以解释全部退化。

(d) 精确 best-action accuracy 对 weighted 目标过窄。当前指标只承认最高 advantage 的单个动作；选到另一个 confirmed-positive 动作也算错。应增加“选到任意 confirmed-positive”“实际选择动作的确认 advantage”“target 的 argmax 是否仍为 teacher”等诊断，保留旧指标但不单独据此判断学习失败。

(e) 当前策略表示确有能力上限。建城 action 编码包含成本、占用等，但没有城市身份与拓扑；state 中地图也主要被压成统计值。[编码](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/nn_rank_value/observation.py:413) 导致不同建城落点不可区分。K=16 全部 1,030 个 searched roots 中，有 166 个 build roots，55 个含同 feature 候选，21 个在同一隐藏实现下却具有不同终局 rank vector。这比“语义等价的发电动作撞 feature”更严重：网络被要求对相同输入拟合不同结果。相同数据中 62 个 bureaucracy roots 有 21 个 feature collision，未发现组内 rank vector 冲突。这个上限值得后续独立处理，但当前主要退化包含拍卖动作，不能把整轮失败归因于建城编码。

(f) 确认 sampler 不是完整、甚至不是处处符合发牌先验的 information-set sampler。除固定隐藏牌组成之外，开局第一张未公开补牌必须是 plug 类型：[发牌实现](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/model.py:1001)。[sampler](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/nn_rl_based/search.py:370) 对完整 draw stack 洗牌。本次构造 seed=61001 的合法开局，对重采样 seed=0…15 检查，12/16 次首张成了 socket。该复现是 sampler 正确性反例，并不是对整套数据受影响比例的估计。继续生成正式确认数据前应单独修复并验证这项约束。

(g) K=8/16 下的 `mean - 1.645 SE > 0` 是近似筛选，不能视作经过校准的“每个接受动作 95% 正确”。小样本、离散收益、多候选筛选、SE=0 的情况都限制了覆盖率；筛选样本排除是正确做法，但用确认均值同时过门槛及赋权仍有选优偏差。增加 K 也不修复错误的采样分布。

(h) 终局 rank 很粗，但它与当前评估目标一致。败局报告中“同为第三名、发电从 15 城降到 7 城”确实没有 rank 惩罚；这提示方差/分辨率问题，并不自动证明应该加入城市数 reward。加入辅助收益可能偏离获胜目标，而且是新的变量，不适合作为下一轮第一步。

**6. 评估和工程可追溯性**

旧 deterministic suite 对同一 seed 的两种 lineup 分别运行，却按 game 独立 bootstrap；应按 seed 保留相关性。两种 lineup 也不等于对每个 seed 穷举全部座位排列。新 checkpoint duel 已完整实现这两点。

本次直接用历史 game records 按 seed 重算 10,000 次 bootstrap，当前 schema-v2 模型四对手的固定种子 CI 仍分别为 `[.664375,.744375]`、`[.755,.8225]`、`[.55125,.6325]`、`[.835,.88875]`；新种子最弱的 expansion 仍为 `[.5425,.655]`。因此修正统计口径不会推翻当前模型的主要历史结论。[重算结果](/Users/mac/Desktop/syt/Projects/PowerGrid/artifacts/audits/ai_nn_rl_based_v1_20260919/additional_checks.json)。

离线 acceptance 中 `all_checks_pass=false` 与外层 training `status=PASS` 含义不同：前者是模型门槛，后者是流程运行完成。旧门槛还保留 schema-v1 的固定 Q-MAE 参考 `.1470`；换成 MC target 后不应直接拿这个值跨任务比较。

`save()` 默认继承初始模型所有 metadata；本次读取的失败候选仍有 `release_status=current_best` 等旧发布字段。[继承位置](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/nn_rl_based/model.py:409)。此外 `training_iteration` 只要有 init 就固定为 1，旧诊断 checkpoint 没记录 init checkpoint hash。它们不会直接改变动作，但会误导实验追溯，应与算法实验分开修正。新代码已写入 init hash，不能据此假定所有旧产物都有该信息。

训练只在全部 epoch 完成后统一评估，未记录逐 epoch 的真实棋力或策略漂移；加载 checkpoint 也不恢复 Adam 状态。因此“继续训练若干 epoch”实际包含优化器重启，比较实验时需要一致地重启。

**7. 复验范围**

本次运行 `PYTHONPATH=src .venv/bin/python -m unittest tests.test_nn_rank_value tests.test_nn_rl_based -q`，35/35 通过。另验证 K=16 manifest：3 个 shards、3 个 examples、100 局、50,316 行。检查包含玩家 slot、fork 隔离、全候选覆盖、语义边界、confirmation labels、采样与 checkpoint 读写；通过这些测试不代表统计假设或训练目标正确。

本次仅把验证/测试分割用于审计诊断，没有据其训练新模型。未来按本审计设计修改训练之后，晋级赛需要另留未用于选择方案的 seed 集。

**8. 本次新增的 1,800 局诊断对战**

为了避免仅复述旧败局报告，使用三个已经存在的 9/18 诊断/正式 checkpoint 分别对当前 incumbent 评估。同一批未出现在所检查历史实验中的 seeds `1909001–1909100`，每个 seed 六种 lineup，每模型 600 局、1,200 个跨 checkpoint seat-pair，95% CI 按 seed 聚类。

| 现成模型 | 改进标签 | Policy/Q-MC/Q-search 权重 | 对 incumbent score | 95% CI |
| --- | --- | --- | ---: | --- |
| diagnostic_teacher_only_balanced | 无，margin=3 大于 rank advantage 上限 | 1/0/0 | 0.50167 | [0.4650, 0.5375] |
| diagnostic_policy_only | 单次 MC，weighted .75 | 1/0/0 | 0.42250 | [0.38665, 0.45833] |
| weighted075 | 同一单次 MC，weighted .75 | 1/1/1 | 0.44125 | [0.40667, 0.47708] |

可核验的相同配置：同一 500 局 dataset manifest SHA-256 `7d31ab02025995167b5ed89c8eff9f0e039d7752860a34304cdba2a35dbe4c8a`，seed=1701，10 epochs，batch=128，lr=.001，balanced_search，temperature=.25，improvement weight=.75。旧诊断文件缺 init hash；不能把元数据相符说成完美记录了全部训练条件的预注册因果试验。本次确认的是这些实际 checkpoint 的表现。

按共同 seed 对各模型“相对 incumbent 的 score 差”再做 10,000 次配对 bootstrap：

- Policy-only − BC：−7.92 个百分点，95% CI [−12.33, −3.50] pp。
- 联合训练 − BC：−6.04 个百分点，95% CI [−10.58, −1.33] pp。
- 联合训练 − Policy-only：+1.875 个百分点，95% CI [−2.375, +6.292] pp。

这些是共同对手上的 paired performance difference，不是两个候选互相直接对战的胜率。

这组结果使诊断更明确：

- 纯 BC 有动作漂移，但没有检出整体退化。不能把 non-search agreement 下降直接当作失败原因；CI 包含 .5 也不是严格等价性证明。
- 不训练 Q 的改进候选仍显著退化。Q 干扰不是解释退化的必要条件，不能据自己 Q 与 Policy 不一致就把“关掉 Q”列为已证实的修复。
- 联合训练在本批 seed 上没有显著差于 Policy-only，点估计还稍高。并非证明 Q 没问题，而是反对“主要必然是 Q 拖累”的强归因。
- 加入单次 MC 改进目标后的策略更新是优先排查对象。结合 K=16 部分修复 K=0 的历史对战，标签噪声与更新幅度/泛化都值得重视，尚不能断言只修其中一个就能超越 incumbent。

[汇总与配对差值](/Users/mac/Desktop/syt/Projects/PowerGrid/artifacts/audits/ai_nn_rl_based_v1_20260919/new_duels_summary.json)、[BC 原始 600 局](/Users/mac/Desktop/syt/Projects/PowerGrid/artifacts/audits/ai_nn_rl_based_v1_20260919/teacher_only_vs_incumbent_100.json)、[Policy-only 原始 600 局](/Users/mac/Desktop/syt/Projects/PowerGrid/artifacts/audits/ai_nn_rl_based_v1_20260919/policy_only_vs_incumbent_100.json)、[联合训练原始 600 局](/Users/mac/Desktop/syt/Projects/PowerGrid/artifacts/audits/ai_nn_rl_based_v1_20260919/joint_k0_vs_incumbent_100.json)。

**9. 我建议的下一步：只改变学习率**

下一轮只回答一个问题：对完全相同的 K=16 标签，把参数更新缩小一个数量级，是否能减少有害泛化，同时保留一部分有效改进？这是一项区分“更新过大”与“现有标签/表示无法产生收益”的低成本实验，不是保证能晋级的修复。

| 项目 | 对照 A | 候选 B |
| --- | --- | --- |
| 初始 checkpoint | 当前 incumbent | 完全相同 |
| 数据 | 现有 data_smoke_k16 | 同一份数据与 split |
| 学习率 | 0.001 | **0.0001** |
| epochs / batch / seed | 10 / 128 / 1701 | 完全相同 |
| Policy target | advantage_weighted，weight .75，temperature .25，margin 0 | 完全相同 |
| loss weights | Policy/Q-MC/Q-search = 1/1/1 | 完全相同 |
| 采样、优化器重启、网络、特征、reward、候选、上线 argmax | 现有实现 | 完全相同 |

不同时减 epoch，不换成 gate，不加 replay/KL，不改 reward，不改 Q loss，也不同时修建城特征。这个实验不需要改任何训练源代码，也不需要重新做昂贵 rollout。

候选命令（本次审计未执行）：

```bash
PYTHONPATH=src .venv/bin/python -m powergrid.tools.train_nn_rl_based \
  --dataset artifacts/experiments/paired_mc_confirm_20260919/data_smoke_k16 \
  --init-checkpoint src/powergrid/data/ai_models/ai_nn_rl_based_v1.npz \
  --output artifacts/models/ai_nn_rl_based_v1_k16_lr0001_candidate.npz \
  --epochs 10 --batch-decisions 128 --seed 1701 \
  --learning-rate 0.0001 \
  --policy-weight 1 --q-mc-weight 1 --q-search-weight 1 \
  --policy-target-mode advantage_weighted \
  --search-temperature 0.25 --improved-action-weight 0.75 \
  --min-search-advantage 0 --training-sampling balanced_search
```

对照 A 最好同时按当前代码复现一次，并逐参数核对原有 K=16 checkpoint。不能用整个 NPZ 的 hash 比较复现是否一致，因为运行时长等 metadata 会改变文件 hash；应比较权重和 normalization 数组。如果对照无法复现，先解释差异，不继续堆候选。

评估预先固定三个层次：

1. 同一个 validation 上记录平均/高分位 KL、分 decision type 的实际 argmax switch、confirmed-positive 选择次数。全部报告绝对样本量。agreement/KL 只是诊断，不作为“有益变化一律淘汰”的硬门槛。
2. 独立 fresh 100 seeds × 6 lineups，对 A、B 使用完全相同赛程对 incumbent 评估，并计算 seed 配对分数差。比较 B 相对 A 是否改善，也看 B 相对 incumbent 是否仍退化。只比旧 K=16 的历史 score 不足以控制 seed 难度。
3. 只有 B 有正向信号，才冻结方案做新的 400 seeds × 6 lineups 晋级赛，并复核四 deterministic 对手。晋级继续要求 direct incumbent score 的 95% CI 下界 > .5、两种人数构成点估计均 ≥ .5；多次查看同一 holdout 后不得把它继续称为最终独立测试。

如果 B 只是恢复到 .50 或动作几乎完全不变，结论是“减轻了回退/基本没有有效改进”，不是“RL 已改善”。如果 B 仍明确退化，停止盲目扫描学习率；再设计下一项独立的标签或策略约束实验。现有 K=16 数据在这里用于隔离更新幅度，不代表 sampler 已经完全正确；在扩展正式确认数据之前，应单独修复并验证开局牌序约束。

不建议照旧败局报告一次实施“确认 Q 均值 + KL/replay + 降 lr/epoch + 加辅助奖励”。也不建议把 `improved_action_weight` 直接降成 .10–.25：当前代码强制它处于 (.5,1]；即使放宽，hard teacher 的 .75–.90 target 会令最大概率动作仍是 teacher，重演早期 soft mix 无法主动改变 argmax 的问题。缩小优化步长与降低改进动作的 target 概率是不同操作。
