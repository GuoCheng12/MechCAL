# 評估指標規格（7 個家族，重建數據集）

2026-11-22，作者確認。取代 2026-11-14 的草稿（ES-F1、按家族 MCC、Brier）和更早以「前三名」為核心的指標（R@3、nDCG@3）。適用於重建後的數據集 `rebuilt_302`。

## 1. 原則

- **只用單個分子就能算的指標。** 每個指標先對一個分子算出一個數，數據集上的分數是對分子的平均。不用需要同時考慮 N 個分子才能算的指標（如按家族 MCC）。
- **圍繞動機：證據要幫助發現和比較機制。** 主指標要求「說中而且有證據」才得分。
- **擋住投機策略。** 全選、隨機選、只答常見家族，都不能在主表上拿到好分數（見第 5 節）。

## 2. 輸出格式（所有方法相同：MechCAL、Direct、ReAct）

每個分子提交一個**有順序的聲明家族列表**，1 到 7 個，每個家族附證據陳述和理由：

- 系統自己決定聲明幾個，沒有固定門檻。未聲明的家族視為「沒有聲明」。
- 第一個家族就是系統預測的主要機制。
- 不再要求每個家族的機率（Brier 分數已取消）。

## 3. 參考答案

每個分子 × 家族在 `hidden_reference.mechanism_decisions` 裡有三種狀態：

| 記號 | 狀態 | 意思 |
|---|---|---|
| S | supported | 論文報導了這個機制（角色為 primary、co_primary 或 secondary） |
| X | contradicted | 論文明確排除了這個機制 |
| — | not_established | 論文沒有涉及，既不算對也不算錯 |

P（主要機制）= S 中角色為 primary 或 co_primary 的家族。

## 4. 指標

記號：C 是聲明的家族集合；C₁ 是第一個聲明的家族；s 是評審給聲明家族的證據分數（0、0.5、1）。

| 簡稱 | 全稱 | 對一個分子的算法 | 方向 | 位置 |
|---|---|---|---|---|
| **ES-Rec** | Evidence-Supported Recall | Σ（S 中被聲明的家族的 s）÷ \|S\| | 越高越好 | 主指標 |
| **Top-1** | Top-1 Primary Accuracy | C₁ 屬於 P 記 1，否則記 0 | 越高越好 | 主表 |
| **EVR** | Exclusion Violation Rate | \|C ∩ X\| ÷ \|X\| | 越低越好 | 主表 |
| **EP** | Evidence Precision | 評審判為「一致」的證據陳述數 ÷ 全部證據陳述數 | 越高越好 | 主表 |
| **Rec** | Label Recall | \|C ∩ S\| ÷ \|S\| | 越高越好 | 次要 |
| **BAL** | Balanced Score | (Rec + 1 − EVR) ÷ 2 | 越高越好 | 次要 |

### 證據分數 s

評審按已批准的評分規則（`mechcal/eval/rubrics/mechanism_evidence_rubric.json`）給每個聲明的家族打分：

- **1（強支持）**：滿足該家族的一條完整「強支持」路徑。例如 ICT：激發態的電子–空穴、軌道或電荷分析顯示電荷在指定區域之間轉移，並解釋它怎樣支持 CT。
- **0.5（部分支持）**：沒有強支持，但滿足一條「部分支持」路徑。例如 ICT：從結構上指出具體的給體和受體，並說明轉移方向，但沒有激發態證據。
- **0**：以上都不滿足。
- **程序核對**：評審依據的陳述中，只要有數值對不上工具記錄（編造），分數就是 0。

### 例子

參考答案：S = {ICT, RIM}，P = {ICT}，X = {ESIPT}。
系統提交 [ICT, AGG, ESIPT]，評審給分 ICT 1、AGG 0.5、ESIPT 0；6 條證據陳述中 5 條判為一致。

| 指標 | 計算 | 結果 |
|---|---|---|
| ES-Rec | (ICT 1 + RIM 0（未聲明）) ÷ 2；AGG 不在 S 中，不計 | 0.50 |
| Top-1 | ICT ∈ P | 1 |
| EVR | 聲明了 ESIPT → 1 ÷ 1 | 1.00 |
| EP | 5 ÷ 6 | 0.83 |
| Rec | 1 ÷ 2 | 0.50 |
| BAL | (0.50 + 1 − 1.00) ÷ 2 | 0.25 |

Rec 與 ES-Rec 的差，表示「說對了但證據不夠」的部分。EP 不看家族對錯，只看陳述是否和計算記錄、結構一致。

## 5. 下限

隨機策略每種抽 4000 次模擬。「最常見 3 個」按開發集的頻率選出，即 AGG、ICT、RAD。

| 策略 | 開發集 29 個：Rec / EVR / BAL | 全部 302 個：Rec / EVR / BAL | ES-Rec | EP |
|---|---|---|---|---|
| 隨機 1 個 | 0.14 / 0.14 / 0.50 | 0.13 / 0.14 / 0.50 | 0 | 不適用 |
| 隨機 3 個 | 0.43 / 0.43 / 0.50 | 0.42 / 0.42 / 0.50 | 0 | 不適用 |
| 隨機個數（1–7 個） | 0.57 / 0.56 / 0.50 | 0.57 / 0.56 / 0.50 | 0 | 不適用 |
| 全部 7 個 | 1.00 / 1.00 / 0.50 | 1.00 / 1.00 / 0.50 | 0 | 不適用 |
| 最常見 3 個（固定） | 0.46 / 0.02 / 0.75 | — | 0 | 不適用 |

隨機選 k 個家族時，Rec 和 EVR 都約等於 k/7，BAL 都約為 0.50。沒有證據時 ES-Rec 是 0。

| 策略 | Top-1：開發集 / 全部 302 個 |
|---|---|
| 隨機 1 個（期望值 \|P\| ÷ 7） | 0.17 / 0.18 |
| 固定答最常見的主要機制（開發集 PET；全集 ICT） | 0.21 / 0.19 |
| 結構 kNN-5（按論文單元留一，測試集，另一 thread 計算） | 0.33 |

結構 kNN-5 在開發集上（28 個可運行的案例，鄰居取自其他論文單元，取它的前三名作為答案）：Top-1 0.48、EVR 0.17、Rec 0.62、BAL 0.69，ES-Rec 0。

結果表（`examples/molecule_table.py`）會在方法實際跑過的分子上，同時列出隨機、全選、最常見 3 個和結構 kNN-5 的分數。

## 6. 計算範圍與統計

| 指標 | 計入的分子 | 開發集 | 全部 302 個 |
|---|---|---|---|
| ES-Rec、Rec | S 非空 | 28 | 288 |
| Top-1 | P 非空 | 28 | 281 |
| EVR | X 非空 | 22 | 222 |
| EP | 至少有一條證據陳述 | 依方法而定 | 依方法而定 |
| BAL | S 和 X 都非空 | 21 | 208 |

- 沒有支持標籤的 14 個分子只計 EVR。
- 同一分子跑多次時，先在分子內平均，再對分子平均。
- 置信區間：按論文單元分群做 bootstrap，報 95% 區間；方法之間做配對比較。
- 每個分子的 tokens（系統和評審分開）和費用與指標一起報告。

## 7. 依賴 LLM 評審的程度

6 個指標中，ES-Rec 和 EP 依賴評審；Top-1、EVR、Rec、BAL 只看標籤。因此評審的可信度要另外報告（放附錄）：

- 擾動實驗：改動陳述後被抓出的比例；
- 抽樣人工評分與評審的一致度；
- 換成另一個模型家族的評審後，主表的變化。

## 8. 先例

| 指標 | 先例 | 核實情況 |
|---|---|---|
| ES-Rec | FEVER score（標籤對且證據完整才計分） | 已核實 |
| EP | ALCE 的 citation precision；FActScore | ALCE 已核實；FActScore 憑記憶 |
| 聲明不定數量 | LAB-Bench 的 precision 與 coverage 分開報告 | 已核實 |
| 成本與準確率一起報告 | Kapoor 等，"AI Agents That Matter"（2024） | 憑記憶 |

「憑記憶」的條目寫論文前要逐一核對出處。

## 9. 實現（2026-11-22，作者批准後完成）

| 內容 | 位置 |
|---|---|
| 6 個指標、分子內平均、按論文單元 bootstrap | `mechcal/eval/molecule_metrics.py`（測試 `tests/test_molecule_metrics.py`：第 4 節的例子、第 5–6 節的下限和計數） |
| 讀 `rebuilt_302` 和新劃分 | `mechcal/eval/dataset.py`、`mechcal/eval/splits.py` |
| 提交格式（1–7 個有順序的家族） | `mechcal/eval/submission.py`；Direct、ReAct 的任務說明 `mechcal/eval/prompts/submission_instructions.md` |
| MechCAL 的答案 | 教授在假設板上給每個家族標記 `claimed`；被標記的家族按可信度排序就是答案（`mechcal/agent/board.py`）。一個都沒標記的運行按「沒有有效答案」計分 |
| 單次運行評分、結果表 | `examples/grade_agent_run.py`、`examples/molecule_table.py` |

真實模型冒煙（2026-11-22，案例 0047 和 EXPA_025，MechCAL、Direct、ReAct 各 1 次）：6 次運行都給出有效答案並完成評分，結果在 `outputs/agent/smoke_claimed`、`outputs/baselines/smoke_claimed_direct`、`outputs/baselines/smoke_claimed_react`。聲明的家族數：Direct 2–3 個，ReAct 3–4 個，MechCAL 4–5 個。

還沒做：`docs/evaluation-design.md` 仍描述舊的前三名指標。
