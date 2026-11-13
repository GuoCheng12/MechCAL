# 機制家族縮減：11 個改為 7 個（2026-11-13，待作者確認評分規則）

**狀態**：家族列表、定義、提示詞、舊標籤映射已改；合併後的評分規則是草稿（`mechanism_evidence_rubric.json` 的 status 為 `draft_pending_author`），需要作者逐條確認。

**原因**（作者決定）：只保留計算工具能給出區分性證據的家族。11 個家族中，HOST_GUEST、RACI、SOKR 在我們的工具下拿不到強支持（見已確認的 `rubric-review-zh.md` 第 9 點）；PACKING 只能用建模二聚體近似，而且 75/119 個 case 都有它，幾乎是 AIE 的前提，而不是區分點。

## 新的 7 個家族

| 家族 | 和原來的關係 |
| --- | --- |
| RIM_RIR_RIV | 擴大：限制可以來自分子本身，也可以來自堆積、主體或基質；併入 RACI（錐形交叉點通道受阻） |
| ICT_TICT_CT | 不變 |
| ESIPT_PT | 不變 |
| PET_ET | 不變 |
| AGGREGATE_EXCITON_EXCIMER | 擴大：包括堆積改變分子間電子耦合的情況 |
| RADIATIVE_RATE_STATE_BALANCE | 不變 |
| TRIPLET_ISC_RTP_TADF | 由 TRIPLET_METAL_ENERGY_TRANSFER 改名；去掉配體到金屬的能量轉移和金屬中心發光 |

刪除：HOST_GUEST_INTERACTION、SOKR_ANTI_KASHA。取消：PACKING_HOST_MATRIX_CONFINEMENT（環境改為每個 case 記錄的實驗條件）。

## 需要作者確認的評分規則改動

路線和條件原文盡量沿用已確認的 0.3.0 規則，只做合併。每條請回覆「同意 / 改成…」。

1. **RIM 強支持路線 1 明確包括環境造成的限制。**條件 1 原文加上「（分子內擁擠，或指定的接觸、堆積、主體或基質環境）」。這等於把原 PACKING 強支持路線 1（建模二聚體中的約束，作者已在第 1 點接受「計算」）併進 RIM，但仍然要求指出**被限制的是哪個分子內運動**。
2. **RACI 的兩條強支持路線成為 RIM 的路線**（`crossing_access_and_restriction`、`nonadiabatic_pathway`）。現有工具到不了，只是保留給以後。
3. **RIM 新增兩條部分支持路線**：
   - `environment_constraint_proxy`（原 PACKING 的 `specific_confinement_proxy`）：指出具體的接觸或空間約束，並說明它可能限制哪個分子內運動；
   - `coordinate_linked_crossing_proxy`（原 RACI 的部分支持路線，第 2 個條件接受「計算或解釋」，作者已在第 7 點接受）。
   兩條都只給 0.5，RIM 的分數上限不變（只靠結構 0.5，閉卷計算 1.0）。
4. **RIM 的混淆邊界改寫**：接觸如果改變的是分子間電子耦合而不是運動，算 AGGREGATE，不算 RIM；只有運動受限不等於錐形交叉點，談通道受阻要有座標到交叉點的論證。
5. **原 PACKING 的另外兩條路線不保留**：`environment_switching`（只接受實驗觀測，SMILES-only 下無法滿足）、`specific_structure_based_hypothesis`（純結構推測，RIM 和 AGGREGATE 已有對應的結構路線）。
6. **AGGREGATE 只改定義**，路線和條件不變；邊界說明改為「同一個聚集體可能同時涉及運動受限（RIM）和激子效應，分別評分」。
7. **TRIPLET 去掉 `metal_sensitization` 強支持路線**；部分支持路線 `triplet_proxy` 的條件 1 去掉「金屬發光」，改為「單重態-三重態能隙、自旋軌道耦合、軌道、氧氣或溫度響應、延遲信號」。

其他家族（ICT、ESIPT、PET、RADIATIVE）的路線、條件和上限都沒有改。

## 舊數據的標籤怎麼映射（119 個 case）

規則寫在 `mechcal/eval/family_reduction.py`，是機械規則，沒有逐個 case 判斷：

- RACI 併入 RIM：角色取較強的，gain 取較大的，證據合併。
- TRIPLET_METAL_ENERGY_TRANSFER 改名為 TRIPLET_ISC_RTP_TADF。
- PACKING 標籤移除，記入該 case 的 `environment_context`。**某個 PACKING 效應屬於 RIM 還是 AGGREGATE，是標註判斷，這裡不猜**；重建的數據集按新規則直接標註。
- HOST_GUEST、SOKR 標籤移除，記入 `out_of_pool_labels`。
- 原始標籤完整保留在每個 case 的 `hidden_reference.family_reduction` 裡。

映射後的統計見本次提交的 `raw_v2_119_corrected/manifest.json`（`family_reduction` 一節）。只報告總數，沒有逐個查看測試 case 的標籤。

### 映射結果（119 個 case 的總數）

| 家族 | 有這個標籤的 case | 其中是主要（primary / co_primary）的 |
| --- | ---: | ---: |
| RIM_RIR_RIV | 68 | 41 |
| ICT_TICT_CT | 59 | 15 |
| RADIATIVE_RATE_STATE_BALANCE | 61 | 17 |
| TRIPLET_ISC_RTP_TADF | 30 | 19 |
| AGGREGATE_EXCITON_EXCIMER | 26 | 10 |
| PET_ET | 21 | 15 |
| ESIPT_PT | 16 | 10 |

- 75 個 case 有 PACKING，現在記為環境條件；23 個 case 有 HOST_GUEST、16 個有 SOKR，記為池外標籤。
- 11 個 case 映射後**沒有主要標籤**（原來的主要標籤是 PACKING、HOST_GUEST 或 SOKR）：0008 0062 0063 0064 0066 0069 0071 0097 0140 0147 0148。其中只有 0140 在開發集。
- 0064 映射後**沒有任何標籤**，評分時應排除。
- 平均每個 case 2.38 個標籤、1.08 個主要標籤。

## 影響

- **固定答案的基線大幅升高。**用固定的「RIM、RADIATIVE、ICT」回答每個 case，R@3 就有 0.68、nDCG@3 0.60（118 個有標籤的 case）；11 個家族時最好的固定答案只有 R@3 0.48。原因是 7 個家族裡平均每個 case 就有 2.4 個標籤，前三名幾乎能把它們都覆蓋。固定答案取前兩名時 R@2 是 0.48，只取第一名時 R@1 是 0.27。所以**繼續用「前三名」作主要指標，區分度會很差**，需要決定：改成前兩名或只看第一名，或者以主要標籤和有證據支持的排名（ES-nDCG）為主。這要在重建數據集前定下來。
- 新的固定答案基線（`mechcal/eval/common_prior.py`）已改為 RIM、RADIATIVE、ICT。
- 舊的 11 家族評分規則已確認的版本留在 git 歷史裡；0.2.0 的規則文件不動。
- **0.2.0 工作流**（`mechcal/agents`、`mechcal/runtime`、`mechcal/schemas` 及其舊基線）保持凍結，仍用 11 家族（`ELEVEN_FAMILY_POOL`）。改成 7 家族的是 0.3.0 agent（`mechcal/agent`）、Direct/ReAct 基線（`mechcal/eval/grounded_subjects.py`）和評估（評分規則、統計、固定答案基線）。
- 開發集和測試集的劃分不變。劃分文件記錄的覆蓋統計是按原來 11 家族的標籤算的（`make_dev_test_split.py` 讀原始標籤），以便原樣重現。
- 之前所有開發運行（dev3、dev4 等）用的是 11 個家族，數字不能和之後直接比較。
