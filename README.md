# lights-out-rl

ライツアウトの環境と、それを解く強化学習エージェント（表形式の Q 学習・SARSA・Profit Sharing・モンテカルロ法）の実験用プログラムです。

## セットアップ

pyenv の仮想環境 `lights-out-rl` (Python 3.13.12) を `pyenv local` で指定済みです。
このディレクトリに入るだけで有効になります。

```sh
pyenv virtualenv 3.13.12 lights-out-rl   # 初回のみ (作成済みなら不要)
pip install -e ".[dev]"
python -m pytest                          # テスト
```

## 使い方

### コマンドライン

```sh
python -m lights_out --episodes 10000 --seed 0 --demo                # Q 学習 (既定)
python -m lights_out --method sarsa --episodes 10000 --seed 0        # 手法は --method で選ぶ
python -m lights_out --method profit-sharing --size 4 --discount-factor 0.1
python -m lights_out --method monte-carlo --size 4 --demo
python -m lights_out --help               # 環境・エージェント・実験のオプション一覧
```

- `--method` は `q-learning`（既定）/ `sarsa` / `profit-sharing` / `monte-carlo`。
- `--learning-rate` `--discount-factor` `--epsilon-start` `--epsilon-min` `--epsilon-decay` は、省略すると選んだ手法ごとの既定値になります（`--help` に手法ごとの既定値が出ます）。
  その手法にない項目（例: `profit-sharing` への `--learning-rate`）を指定すると、使い方のエラー（終了コード 2）になり、何も記録しません。
- `--name` を省略すると手法の名前になります（`runs/<日時>_sarsa/` など）。

4×4・10000 エピソードの学習は、どの手法でも 1〜2 秒で終わります（後述の比較）。結果は `runs/<日時>_<name>/` に記録されます（後述）。

### Python から

```python
from lights_out.domain.environment import EnvironmentConfig, LightsOutEnvironment
from lights_out.domain.agent import QLearningAgent, QLearningConfig

env = LightsOutEnvironment(EnvironmentConfig(size=4, seed=0))   # size を変えれば 5x5 などにもなる
agent = QLearningAgent(env.n_actions, QLearningConfig(seed=1))

agent.train(env, n_episodes=10000)                              # 任意のエピソード数で学習
print(agent.evaluate(env, n_episodes=100))                      # 探索なし・学習なしで評価

env.reset()
env.render()                                                    # 標準出力に表示
env.get_state()                                                 # 盤面の一次元配列 (左上 → 右下, 1=点灯)
```

他の手法も同じ形で、`train` / `evaluate` はそのまま使えます。

```python
from lights_out.domain.agent import ProfitSharingAgent, ProfitSharingConfig   # SarsaAgent / MonteCarloAgent も同様

agent = ProfitSharingAgent(env.n_actions, ProfitSharingConfig(discount_factor=0.1, seed=1))
agent.train(env, n_episodes=10000)
print(agent.evaluate(env, n_episodes=100))
```

## 環境 (`LightsOutEnvironment`)

| メソッド | 内容 |
| --- | --- |
| `reset(initial_state=None)` | 新しいエピソードを開始して盤面を返す。省略時は「全消灯から乱数でマスを押した」解ける盤面（クリア済みにはならない） |
| `step(action)` | `action` のマス（左上 0 〜 右下 `size**2-1`）を押す。`StepResult(state, reward, terminated, truncated)` を返す |
| `get_state()` | 盤面を一次元の配列（左上 → 右下、行優先）で返す |
| `is_start()` | 開始状態か（`reset` 直後で 1 手も押していない） |
| `is_solved()` / `is_truncated()` / `is_done()` | クリア / 最大手数での打ち切り / どちらかで終了 |
| `render()` | 標準出力に表示する |

- 盤面の状態（`Board`）と手数は環境クラスが保持します。生成直後に `reset` 済みです。
- 報酬は 1 手ごとに `step_reward=-1`、クリアした手は `goal_reward=100`（`EnvironmentConfig` で変更可）。
- クリア（`terminated`）と打ち切り（`truncated`）は区別します。Q 学習・SARSA は打ち切りの先にも価値があるとして更新し、モンテカルロ法は打ち切りもエピソードの終わり（その先の収益は 0）として扱います（次節）。

## エージェント (`Agent` と 4 つの手法)

`Agent` は抽象クラスで、サブクラスは次を実装します。エピソードを回す処理（`play_episode` / `train` / `evaluate`）は共通です。

- `select_action(state, explore)` / `update(transition)` — 行動選択と学習
- `hyperparameters()` / `snapshot()` / `restore()` — 実験の記録と保存

`play_episode` は、学習するとき（`learn=True`）に 1 手ごとに `update` を呼び、最後に 1 回だけ `end_episode` を呼びます（探索率の減衰やエピソード単位の更新に使います）。
学習しないとき（`learn=False`）はどちらも呼びません。エージェントと環境の行動数が違うと `ValueError` になります。

4 つの手法は、どれも「盤面 → 各行動の値」の辞書を持つ表形式で、行動は ε-greedy で選びます（`TabularAgent`）。辞書に入るのは、Q 学習・SARSA・モンテカルロ法では訪れた盤面、Profit Sharing ではゴールしたエピソードの経路上の盤面だけです。
ε はエピソードごとに `epsilon_decay` を掛けて `epsilon_min` まで下げます。学習しない評価では、値が最大の行動（同値なら番号の小さい方）を選びます。

| 手法 (`--method`) | クラス / 設定 | 学習するもの・更新 | ブートストラップ | 方策 |
| --- | --- | --- | --- | --- |
| Q 学習 (`q-learning`) | `QLearningAgent` / `QLearningConfig` | Q(s,a) ← Q + α (r + γ maxQ(s',·) − Q) を 1 手ごとに更新 | する。打ち切りの先にも価値があるとして更新 | オフ方策 |
| SARSA (`sarsa`) | `SarsaAgent` / `SarsaConfig` | Q(s,a) ← Q + α (r + γ Q(s',a') − Q)。a' は実際に次に選んだ行動なので、1 手遅れて更新 | する。打ち切りで終わった手は、同じ ε-greedy で a' を選んで更新 | オン方策 |
| モンテカルロ法 (`monte-carlo`) | `MonteCarloAgent` / `MonteCarloConfig` | Q(s,a) ← Q + α (G − Q)。エピソードが終わってから、各 (盤面, 行動) の**最初の出現**だけを実際の割引収益 G で更新（初回訪問） | しない。打ち切りもエピソードの終わり（その先の収益は 0） | オン方策 |
| Profit Sharing (`profit-sharing`) | `ProfitSharingAgent` / `ProfitSharingConfig` | ルール (盤面, 行動) の重みを、**ゴールしたエピソードだけ**、経路上の全ルールについて強化する。ゴールの n 手前のルールには、ゴールした手の報酬（既定 +100）の γⁿ 倍を足す | しない | 実際にたどった経路を強化（オン方策） |

- Q 学習・SARSA・モンテカルロ法の設定項目は共通（`TabularConfig`）で、既定値は 学習率 α=0.1、割引率 γ=0.99、ε は 1.0 から 0.05 まで（エピソードごとに 0.999 倍）、Q 値の初期値 0.0 です。
- Profit Sharing の設定（`ProfitSharingConfig`）には学習率と初期値がありません。重みは 0 から始め、報酬をそのまま足します。
  `discount_factor` は Q 学習系の割引率ではなく、信用割当の減衰率 γ（既定 0.1、範囲は (0, 1]）です。
  1 手ごとの報酬（-1）と、ゴールできなかったエピソードは無視します（重みは変わりません）。
- 値の表は `snapshot()` / `restore()` で JSON にできます。キーは Q 学習・SARSA・モンテカルロ法が `q_table`、Profit Sharing が `weights` です。

### 新しいエージェントを足すには

`Agent` を継承して上の 5 つを実装すれば、`ExperimentRunner` の学習・評価・記録の仕組みに載せられます。
表形式（盤面 → 各行動の値）なら `TabularAgent` を継承すると、ε-greedy の行動選択・値の表・`snapshot` / `restore`・ε の減衰が付いてくるので、
`__init__`（ε の設定・表の初期値・シードを `super().__init__` に渡す）と `update`（と必要なら `end_episode`）、`hyperparameters` を書けば済みます。
`end_episode` を上書きするときは `super().end_episode()` も呼びます。表のキー名を変えたいときはクラス属性 `table_name` を上書きします（`ProfitSharingAgent` は `"weights"`）。

コマンドラインに出すには、`interfaces/cli.py` の `METHODS` に「名前: (エージェントのクラス, 設定のクラス)」を 1 行足します。
`--learning-rate` `--discount-factor` `--epsilon-start` `--epsilon-min` `--epsilon-decay` は、設定のクラスに同名の項目があれば渡され、なければ使い方のエラーになります。`--help` の既定値の表示も設定のクラスから作ります。

### 盤面サイズと学習の目安

既定の設定（報酬 -1 / +100, α=0.1, γ=0.99, ε 減衰 0.999, 最大 100 手）での **Q 学習**の実測です。他の手法の比較は後述します。

| 盤面 | 10000 エピソード | 備考 |
| --- | --- | --- |
| 3×3 | 約 0.6 秒、成功率 100% | |
| 4×4 | 約 1 秒、成功率 99% 前後（seed 0〜5） | 30000 エピソードならほぼ 100% |
| 5×5 | 約 10 秒、成功率 0% | 状態数が多すぎて表形式では学習が成立しない。メモリ約 1.5GB、`agent.json` は約 300MB |

5×5 以上を学習させるには、表形式でなく関数近似（DQN など）のエージェントを `Agent` のサブクラスとして追加する想定です。
環境（`size`）自体は 5×5 以上でも動きます。

### 手法の比較 (3×3・4×4)

4 つの手法を、それぞれ実際のコマンドライン（`python -m lights_out --method <手法> --size <3 か 4> --episodes 10000 --seed <0〜2> --eval-interval 0 --eval-episodes 1000`）で動かした実測です。
報酬・最大手数・ハイパーパラメータは手法ごとの既定値で、seed 0〜2 の 3 回ずつ走らせています。

- 成功率・平均手数: 学習後の最終評価（探索なし、1000 エピソード）。成功率は 3 回の平均（括弧内は最小〜最大）。平均手数は 3 回の平均で、失敗（100 手で打ち切り）も 100 手として含みます。
- 時間: コマンド全体の実時間（Python の起動と、学習前後の評価 1000 エピソードずつを含む）の 3 回の平均。手元の Mac での値で、環境によって変わります。

| 盤面 | 手法 | 成功率 | 平均手数 | 時間 |
| --- | --- | --- | --- | --- |
| 3×3 | `q-learning` | 100.0% (100.0〜100.0) | 4.6 | 0.8 秒 |
| 3×3 | `sarsa` | 100.0% (100.0〜100.0) | 4.6 | 0.9 秒 |
| 3×3 | `profit-sharing` | 100.0% (100.0〜100.0) | 4.7 | 0.8 秒 |
| 3×3 | `monte-carlo` | 99.5% (98.9〜100.0) | 6.7 | 1.1 秒 |
| 4×4 | `q-learning` | 99.6% (99.5〜99.8) | 5.5 | 1.4 秒 |
| 4×4 | `sarsa` | 99.4% (99.2〜99.5) | 5.8 | 1.5 秒 |
| 4×4 | `profit-sharing` | 100.0% (100.0〜100.0) | 7.0 | 1.5 秒 |
| 4×4 | `monte-carlo` | 94.6% (93.8〜95.2) | 15.5 | 1.9 秒 |

この条件では Q 学習・SARSA・Profit Sharing は 3×3・4×4 とも成功率 99% 以上でほぼ並び（seed 3 回では差を言えません）、モンテカルロ法だけが 4×4 で 94.6% と低く手数も長めでした（10000 エピソードでの結果で、原因までは調べていません）。

## 構成 (DDD)

依存は `interfaces → application → domain ← infrastructure` の向きです。

```
src/lights_out/
├── domain/                     # ルールと概念。他の層に依存しない
│   ├── environment/            #   Board (値オブジェクト), LightsOutEnvironment (エンティティ), EnvironmentConfig
│   ├── agent/                  #   エージェント
│   │   ├── agent.py            #     Agent (抽象), Transition
│   │   ├── tabular.py          #     TabularAgent (ε-greedy と盤面ごとの値の表の土台), TabularConfig
│   │   ├── q_learning.py       #     QLearningAgent
│   │   ├── sarsa.py            #     SarsaAgent
│   │   ├── profit_sharing.py   #     ProfitSharingAgent
│   │   └── monte_carlo.py      #     MonteCarloAgent
│   └── experiment/             #   EpisodeRecord / EvaluationSummary (値オブジェクト), ExperimentRepository (保存先の抽象)
├── application/                #   ExperimentRunner: 学習 + 途中評価 + 記録のユースケース
├── infrastructure/             #   FileExperimentRepository: runs/ への保存, 実行環境の情報 (git など)
└── interfaces/                 #   コマンドライン (--method の手法一覧 METHODS を持つ)
tests/                          # src と同じ構成
```

## 実験の追跡

`runs/<日時>_<name>/`（git 管理外）に、再現と比較に必要なものを残します。

| ファイル | 内容 |
| --- | --- |
| `config.json` | 実験・環境・エージェント（`agent.type` に手法のクラス名、ハイパーパラメータ）の設定（全シード込み）と、コミット・未コミット変更の有無・Python バージョンなど |
| `episodes.csv` | エピソードごとの報酬・手数・成否・ε・表の大きさ（表にある盤面の数。Q 学習・SARSA・モンテカルロ法は列名 `q_table_size` で訪れた盤面の数、Profit Sharing は列名 `weights_size` でゴールしたエピソードの経路上の盤面の数なので、初めてゴールするまでは 0）。ε と表の大きさは**そのエピソード終了後**の値で、エピソード N で実際に使った ε は N−1 行目の値 |
| `evaluations.csv` | 学習前（episode 0）と一定間隔ごとの評価（成功率・平均手数・平均報酬） |
| `agent.json` | 学習済みの値の表と ε（表のキーは Q 学習・SARSA・モンテカルロ法が `q_table`、Profit Sharing が `weights`。同じクラスのエージェントに `restore()` で復元可能） |
| `summary.json` | 終了時刻と最終評価。**これが無い run は学習が途中で終了したもの**（`config.json` だけが残る） |

- **再現**: 乱数は環境・エージェント・評価でそれぞれ独立した `random.Random` を持ち、`--seed` から決めます（環境=seed, エージェント=seed+1, 評価=seed+2）。同じ設定・同じシードなら同じ結果になります。
- **評価**: 評価は毎回同じシードの別の環境で行うので、開始盤面が揃います。探索も学習もせず、エージェントの乱数も使わないので、評価の頻度を変えても学習の結果は変わりません。
