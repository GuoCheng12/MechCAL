# MechCAL 0.2.0 command 工作流架構審查

日期：2026-11-11。對象是目前 `main` 上 `mechcal ... run` 走的 command 工作流（0.2.0，profile `command_v1`）。
依據是程式碼本身和交接手冊記錄的 0003 運行數據；問題按「對論文的傷害」排序，不按程式碼位置。

## 一句話結論

系統把大部分複雜度花在「讓 LLM 管理自己的工作流程」上，而不是「讓證據推動機制比較」。
結果是：每次計算要 3–4 次以上的 LLM 調用，Planner 每一步要讀約 40 個區塊的流程狀態，
而工具目錄裡有三分之一是不產生計算證據的啟發式「工具」。

## 一、科學層面（直接影響論文主張）

### 1. 工具目錄裡混入大量「偽證據」工具

67 個能力中有 21 個 macro 工具，自述都是「structural prior proxy；不是直接機制證據」，其中一些：

- `macro.run_dimer_packing_proxy`：沒有建任何二聚體，輸出的是 `aromatic_fraction + min(logP/10,0.5) - min(rotatable/20,0.3)`
  這個自訂公式（在 TPE 上得 1.22），作為「dimer_contact_score」。
- `macro.run_water_fraction_aggregation_checklist`：輸出一份「需要哪些實驗」的清單。
- `macro.screen_rotor_rim_prior`：輸出 `rim_prior: true`。

這些都以 `status: success` 寫進 evidence ledger。後果有三個：

- **計數膨脹**：工具調用次數變多。
- **證據稀釋**：Planner 和 Reviewer 會把它們當證據討論。
- **審稿風險**：論文的「計算證據」裡混有沒有物理意義的分數。

真正有價值的結構資訊（供體/受體位置、可旋轉鍵、共軛路徑、H 鍵預組織）是確定性的 RDKit 事實，
應該在開始時一次性免費提供，而不是包裝成要花一輪去「調用」的工具。

### 2. 工具的參數陷阱暴露給 LLM

已知例子：baseline 預設 `td_tout=1`，之後的空穴-電子分析就會被拒絕，而且不會自動重跑（我在本機實測確認，
交接手冊第 10 節的實驗也踩到同一個坑）。類似的多步依賴還有：
- 二聚體搜索 → 選構型 → 接觸扭轉響應；
- 基態優化 → 激發態。

這些都要 Planner 自己記住順序和參數。它們是工程知識，不是科學判斷，應該做成固定的「配方」。

### 3. 「比較競爭機制」沒有成為核心數據結構

故事線是「證據幫助發現和比較機制」，但目前「我們知道什麼」分散在至少 6 個重疊的存儲裡：
- `evidence_ledger`、`claim_ledger`；
- protocol 裡的 `arguments`、`candidates`、`explanations`、`results`；
- hypothesis tracking 的 `entries`。

最終答案又是另外寫的一段文字，所以才需要一個多達 30 個模型 turn 的「交付與審查」階段，去核對文字和記錄是否一致。

## 二、編排層面（直接造成成本）

### 4. 每次計算固定觸發 3–4 次以上 LLM 調用

`command_loop.py` 的流程：`execute` 之後進入 `explain`（Worker 解釋，1 次 LLM），回到 `choose` 時若有未審查的結果，
先進入 `impact_review`（1 次 LLM），到了邊界條件再進入 `exploration_review`（1 次 LLM），然後才輪到 Planner（1 次以上，
讀取記錄還要額外的 turn）。0003 的記錄是 19 次計算、122 次請求，平均每次計算約 6.4 次請求。

### 5. Planner 被要求管理流程，而不是做科學判斷

`planner_payload()` 每一步組裝約 40 個頂層區塊，大部分是流程狀態：
- **流程控制：** `operation_policy`、`delivery_recovery`、`judgment_status`、`interim_review_status`、`query_state`、
  `review_workflow`、`correction_context`、`computation_allowance`、`investigation_policy`、`proposal_decision`、
  `computational_review`、`execution_inventory`、`untried_routes`……
- **記錄索引：** 六張（argument / evidence / result / artifact / action / explanation）。

每個區塊還附帶一段 `meaning` 或 `policy` 說明。

Planner 可用的命令約 18 種（各類 read、compute、compute_batch、delegate、hypothesis_step、update_hypotheses、
manage_experiments、amend/repair/discard_judgment、submit/amend/review_final……）。工具全部可見時，原生函數有 76 個，
schema 約 90 KB。交接手冊記錄的 Planner 請求體為 86–113 KB；0003 的平均每次請求輸入約 2.8 萬 token。

### 6. 狀態是一個無類型的大字典，每一步全量寫盤

- **狀態存放：** 流程狀態存在 `case.runtime` 裡一個無類型的 protocol dict。
- **寫盤方式：** 每個完整的計算結果被複製進這個 dict，每一步都整個 checkpoint 重寫一次。
- **連帶的補丁：** 這也是後來需要 `snapshot_storage`、內容定址 `_payloads`、指紋快取等一整套機制的原因。這些都是為了修補這個設計而加的。

## 三、工程層面（阻礙實驗和 ablation）

### 7. 52 個 profile 開關，新舊路徑交纏

- **開關數量：** `command_v1.json` 有 52 個開關。
- **依賴舊路徑：** command 協議本身要求 `decision_mode = question_loop_v1`，也就是依賴舊的 question loop 配置。
- **閒置的 agent：** `create_orchestrator()` 仍然建立並配置 `mechanism_agenda`、`photophysics_arbiter`、`mechanism_critic`、
  `conclusion_ledger` 四個 agent 的客戶端，但 command 層完全不引用它們。

在這種狀態下做 ablation，很難說清楚「去掉的到底是什麼」。

### 8. 程式碼量與功能不成比例

- **規模：** 主程式約 6.2 萬行，`runtime/` 有 110 個模組，command 層加上 Planner 工具約 7,200 行，prompt 69 份（275 KB）。
- **成本：** 這些程式碼服務的是一個「單分子、輸出 3 個標籤」的任務。每次修改都要讓 3,800 個測試跑 7 分鐘。
- **測試鎖住的是什麼：** 這些測試主要鎖住的是流程細節，不是科學行為。

## 四、0.3.0 重新設計的具體修正（在 [agent-architecture.md](agent-architecture.md) 之上補充）

| 問題 | 0.3.0 的處理 |
| --- | --- |
| 偽證據 macro 工具 | 不進入 0.3.0 的工具目錄。RDKit 結構事實在開始時一次性生成，作為可驗證的結構證據候選直接放進看板 |
| 參數陷阱 | 配方固定已驗證的參數（例如 `td_tout >= 10`）；單步工具只暴露經過篩選的少數幾個 |
| 證據分散在 6 處 | 只保留兩樣：產物記錄（閘道）和看板上的帶引用證據；最終答案是看板的投影，不需要交付審查階段 |
| 每次計算 3–4 次以上 LLM 調用 | 結果解讀併入 Planner 的同一次調用（直接寫出帶引用的證據）；Critic 每 3 次計算看一次，不是每個結果都看 |
| Planner 管理流程 | Planner 只看：看板、最新觀測值、剩餘預算、Critic 的建議。動作只有三類：配方/工具、讀產物、提交 |
| 無類型的大字典 | 看板是有類型的 pydantic 模型；產物由閘道落盤，看板只存引用，不複製結果 |
| 52 個開關 | 0.3.0 只有預算參數和 ablation 開關（Critic、配方、迭代、自檢），每個開關對應論文裡的一行實驗 |

## 五、舊系統裡值得保留的部分

- 科學工具 adapter，以及 Amesp/xTB/PySCF/PySOC 等的子進程管理、失敗分類、部分結果保留。這些質量不錯，0.3.0 直接透過閘道複用。
- 「計算失敗不是負面證據」「建模二聚體不是晶體」這類科學邊界。它們已寫進證據 rubric 的全域規則，0.3.0 的 prompt 也會沿用。
- 結構準備（RDKit 構象、原子映射）。
