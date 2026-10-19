# lights-out-rl

ライツアウトの環境と、それを解く強化学習エージェント（表形式 Q 学習）の実験用プログラムです。

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
python -m lights_out --episodes 10000 --seed 0 --demo
python -m lights_out --help               # 環境・Q学習・実験のオプション一覧
```

4×4・10000 エピソードの学習は 1〜2 秒で終わります。結果は `runs/<日時>_<name>/` に記録されます（後述）。

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
- クリア（`terminated`）と打ち切り（`truncated`）は区別します。Q 学習は打ち切りの先にも価値があるとして更新します。

## エージェント (`Agent` / `QLearningAgent`)

`Agent` は抽象クラスで、サブクラスは次を実装します。エピソードを回す処理（`play_episode` / `train` / `evaluate`）は共通です。

- `select_action(state, explore)` / `update(transition)` — 行動選択と学習
- `hyperparameters()` / `snapshot()` / `restore()` — 実験の記録と保存

新しいエージェントは `Agent` を継承してこの 5 つを実装するだけで、`ExperimentRunner` とコマンドラインの記録の仕組みに載せられます。

`QLearningAgent` は ε-greedy の表形式 Q 学習で、Q テーブルは訪れた盤面だけ持つ辞書です。
ε はエピソードごとに `epsilon_decay` を掛けて `epsilon_min` まで下げます。

## 構成 (DDD)

依存は `interfaces → application → domain ← infrastructure` の向きです。

```
src/lights_out/
├── domain/                     # ルールと概念。他の層に依存しない
│   ├── environment/            #   Board (値オブジェクト), LightsOutEnvironment (エンティティ), EnvironmentConfig
│   ├── agent/                  #   Agent (抽象), QLearningAgent
│   └── experiment/             #   EpisodeRecord / EvaluationSummary (値オブジェクト), ExperimentRepository (保存先の抽象)
├── application/                #   ExperimentRunner: 学習 + 途中評価 + 記録のユースケース
├── infrastructure/             #   FileExperimentRepository: runs/ への保存, 実行環境の情報 (git など)
└── interfaces/                 #   コマンドライン
tests/                          # src と同じ構成
```

## 実験の追跡

`runs/<日時>_<name>/`（git 管理外）に、再現と比較に必要なものを残します。

| ファイル | 内容 |
| --- | --- |
| `config.json` | 実験・環境・エージェントの設定（全シード込み）と、コミット・未コミット変更の有無・Python バージョンなど |
| `episodes.csv` | エピソードごとの報酬・手数・成否・ε・Q テーブルの大きさ |
| `evaluations.csv` | 学習前（episode 0）と一定間隔ごとの評価（成功率・平均手数・平均報酬） |
| `agent.json` | 学習済みの Q テーブル（`restore()` で復元可能） |
| `summary.json` | 終了時刻と最終評価 |

- **再現**: 乱数は環境・エージェント・評価でそれぞれ独立した `random.Random` を持ち、`--seed` から決めます（環境=seed, エージェント=seed+1, 評価=seed+2）。同じ設定・同じシードなら同じ結果になります。
- **評価**: 評価は毎回同じシードの別の環境で行うので、開始盤面が揃います。探索も学習もせず、エージェントの乱数も使わないので、評価の頻度を変えても学習の結果は変わりません。
