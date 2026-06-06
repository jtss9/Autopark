# 自主泊車模擬系統 — 專案總整理

## 專案定位

GPS-denied 環境下的自主泊車規劃系統。不做 SLAM 建圖，專注於**路徑規劃 + 閉環控制 + 量化評估**。
場景來源為已知幾何（ParkingLot），支援倒車入庫（垂直）與路邊停車（平行）兩種模式。

---

## 系統架構

```
場景幾何 (ParkingLot)
    → 佔用格柵 (OccupancyGrid)
    → 路徑規劃器 (Planner)
    → 路徑平滑化
    → 閉環追蹤 (Pure Pursuit Tracker)
    → 模擬視覺化 / CARLA 執行
```

### 車輛模型

- 自行車運動學，後軸中心為參考點
- `wheelbase = 車長 × 0.65`，最大轉向角 35°
- `min_turn_radius = wheelbase / tan(35°)`，預設約 3–5 m

### 可行域

L 形區域 = `lane_rect ∪ spot_rect`，碰撞檢測使用完整矩形四角頂點。

### 場景參數範圍

| 參數 | 範圍 | 預設 |
|---|---|---|
| 車道寬度 | 3.5 – 5.5 m | 4.4 m |
| 停車格長 | 5.0 – 6.0 m | 5.5 m |
| 停車格寬 | 2.0 – 3.0 m | 2.5 m |
| 車長 | 3.5 – 5.0 m | 4.2 m |
| 車寬 | 1.6 – 2.2 m | 1.8 m |

---

## 規劃器

### 1. Single-step MPC

幾何計算三段弧線參考路徑（前進接近 → 倒退弧線 → 直線入庫），MPC（SLSQP，horizon N=5, dt=0.05s）追蹤。
邊界懲罰項防止出界，但參考弧線為剛性預計算，車道較窄時失敗。

- 成功率：**10 %（1/10）**，平均規劃時間 0.19s
- 僅適用於倒車入庫 + 無障礙 + 寬車道

### 2. Multi-step MPC

多次嘗試（最多 5 次），每次從當前姿態重新計算弧線並加入修正機動。
關鍵 bug 修正：目標深度原本 cap 在 `y_end + 1.5m`，導致停在距目標 3.2m 處；修正後直接對準 `spot_top - 0.15m`。

- 修正後預設 6m 車道成功，最終誤差 0.086m
- 4.2m 窄車道（車長 3.8m）成功（Single-step 此處碰撞）
- 剩餘限制：約 5m 車道 + 大車（R ≈ 4.18m）時仍可能碰到格邊緣

### 3. Hybrid A\*

在連續 SE(2) 狀態空間 (x, y, θ) 上做 A\* 搜尋。

**格柵設定**
- 空間解析度：0.10 m/格；方向：36 bins（10°/bin）
- 運動步長：0.45m；積分步長：0.09m
- 最大展開次數：150,000

**運動基元**
- 前進 / 後退 × 5 種方向盤角（全左、半左、直、半右、全右）= 10 個基元
- 每個基元途中所有中間姿態均做碰撞檢測

**Reeds-Shepp Analytic Shot**
- 每隔 50 次展開，或距目標 < 4.5m 時每次展開，從當前姿態嘗試 RS 最短路徑直射終點
- 涵蓋完整 24-word CSC + CCC 家族（含時間翻轉 / 反射對稱）
- RS 路徑同樣做碰撞檢測，通過才接受；每次成功運行平均嘗試 ~1,580 次，接受 1 次
- 9 次成功全部透過 RS Shot 終止，路徑更短更平滑

**啟發式**：歐氏距離 + RS 非完整下界，有快取避免重複計算

- 成功率：**90 %（9/10）**，平均規劃時間 2.65s
- 唯一失敗：`tight_lane` 幾何上不可行

**各場景結果（Hybrid A\*，垂直 + 平行各一）：**

| 場景 | 成功 |
|---|---|
| Clear | 2/2 |
| Entry Blocker | 2/2 |
| Tight Lane | 1/2（另一個幾何不可行）|
| Pillar Near Entry | 2/2 |
| Parked Cars | 2/2 |

### 4. Q-learning（表格型 RL，對照基準）

- 狀態：(ix, iy, iθ)，0.45m × 12 heading bins；動作：10 種（前後 × 5 方向盤）
- Reverse curriculum：先從靠近目標的位置訓練，逐漸推遠起點
- 訓練上限 30 秒
- 成功率：**0 %（0/10）**，符合預期
- 結論：純表格 RL 無法在合理時間內收斂於連續 SE(2) 停車問題

---

## 三種方法整體比較

| 方法 | 成功率 | 平均規劃時間 | 支援平行停車 | 支援障礙物 |
|---|---|---|---|---|
| Single-step MPC | 10 %（1/10） | 0.19s | ✗ | ✗ |
| Hybrid A\* + RS | **90 %（9/10）** | 2.65s | ✓ | ✓ |
| Q-learning | 0 %（0/10） | 5.86s | ✓（訓練） | ✓ |

---

## Pure Pursuit 閉環追蹤

在規劃路徑上疊加 Pure Pursuit 閉環追蹤器，模擬真實控制誤差。
- 路徑按方向分段，倒退段翻轉轉向符號（否則車輛往反方向走）
- 追蹤前密化路徑至 ≤ 0.15m 間距

**9 次成功 Hybrid A\* 路徑的追蹤結果：**

| 指標 | 平均 | 最差 |
|---|---|---|
| 橫向誤差均值（CTE） | **0.019 m** | 0.040 m |
| 橫向誤差最大值 | 0.123 m | 0.486 m |
| 最終位置誤差 | 0.076 m | — |
| 最終航向誤差 | **1.57°** | — |
| 車體完全在格內 | **8 / 9** | — |

---

## 障礙物場景

`ParkingConfig.obstacle_scenario` 選項：

| 值 | 說明 |
|---|---|
| `none` | 無障礙物 |
| `entry_blocker` | 停車格入口前方有障礙 |
| `tight_lane` | 車道兩側縮窄 |
| `pillar_near_entry` | 入口旁有柱子 |
| `parked_cars` | 兩側有停放車輛 |

任何非 `none` 場景自動切換至 Hybrid A\*（MPC 不具障礙物感知能力）。

---

## CARLA 延伸整合

| 模組 | 功能 |
|---|---|
| `carla_bridge.py` | 座標系轉換、靜態障礙物提取、LiDAR 點雲格柵化 |
| `carla_controller.py` | Pure Pursuit → CARLA `VehicleControl`，換檔煞車緩衝 1.2s |
| `carla_demo.py` | `--dry-run` 無需 CARLA 安裝即可端到端驗證 |

**Dry-run 驗證結果：**

| 場景 | 規劃時間 | 最終位置誤差 | 執行成功 |
|---|---|---|---|
| 倒車入庫 clear | 1.00s | 0.012m | ✓ |
| 路邊停車 clear | 2.79s | 0.460m | ✓ |
| 倒車 + 旁邊有車 | 1.17s | 0.011m | ✓ |

---

## 評測平台

```bash
python evaluate.py --mode all --scenario all --planner all --track --output results/main.csv
python plot_results.py results/main.csv   # 產生 results/figures/ 下的圖表
```

CLI 參數：`--mode`（perpendicular/parallel/all）、`--scenario`、`--sweep`（lane_width/car_size）、`--planner`、`--track`、`--output`

---

## 模組一覽

| 檔案 | 職責 |
|---|---|
| `config.py` | `CarConfig`、`ParkingConfig` 資料類別 |
| `parking_lot.py` | 場景幾何（`lane_rect`、`spot_rect`、`car_corners`） |
| `geom.py` | 共用幾何工具（`angle_diff`、`wrap_pi`、`split_by_gear`） |
| `controller.py` | `CarDynamics`（運動學）、`MPCController`（SLSQP） |
| `trajectory.py` | `plan_trajectory()` 派送 + MPC 規劃器 + tracker 附掛 |
| `hybrid_astar.py` | Hybrid A\* + OccupancyGrid + Reeds-Shepp shot |
| `reeds_shepp.py` | Reeds-Shepp 最短路徑（24-word 家族） |
| `tracker.py` | Pure Pursuit 閉環追蹤器 |
| `rl_qlearn.py` | 表格型 Q-learning 基準 |
| `scenarios.py` | 各場景障礙物生成 |
| `simulation.py` | pygame 動畫主迴圈 + HUD |
| `settings_window.py` | tkinter 設定介面 |
| `evaluate.py` | 無頭批量評測 → CSV |
| `plot_results.py` | CSV → matplotlib 圖表 |
| `carla_bridge.py/controller.py/demo.py` | CARLA 延伸整合 |

---

## 模擬器操作

| 按鍵 | 功能 |
|---|---|
| `↑` / `↓` | 動畫速度 0.05x – 10x |
| `SPACE` | 暫停 / 繼續 |
| `R` | 重播 |
| `G` | 佔用格柵覆蓋層 |
| `T` | Pure Pursuit 執行路徑覆蓋層 |
| `S` | 返回設定 |
| `ESC` / `Q` | 離開 |

環境變數：`AUTOPARK_PLANNER=hybrid_astar`、`AUTOPARK_TRACK=1`

---

## 結論

- Hybrid A\* + Reeds-Shepp Shot 達到 **90% 成功率**，MPC baseline 僅 10%，Q-learning 0%
- 閉環追蹤均值橫向誤差 **< 2cm**，8/9 次執行後車體完全在格內
- 所有 7 個 Stage 均已完成，包含 CARLA dry-run 驗證

**未來方向：**
- DQN 取代表格 Q-learning / 用 Hybrid A\* 解暖啟動 RL
- LiDAR 點雲即時格柵化（已實作介面，未接入預設流程）
- CARLA 活體測試（需安裝 CARLA 伺服器）
