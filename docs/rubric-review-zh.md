# 證據 Rubric 審核表（化學內容，0.3.0）

由 `mechanism_evidence_rubric.json` 自動生成。路線與條件原文沿用 0.2.0 版 rubric（英文保留，避免翻譯改變含義）；
0.3.0 新增的是每個條件「接受哪類證據」、由此推導的分數上限，以及評估方內部的工具覆蓋註記（不給 judge 看）。

證據類型：**結構**＝輸入分子圖上可驗證的事實；**計算**＝有產物記錄、數值已被程式核對的計算結果；
**實驗觀測**＝評測協議提供的實驗數據（SMILES-only 設定下沒有，這類條件目前無法滿足）；
**解釋**＝把已引用證據連到機制的、針對該分子的論證，本身不算新證據。常識/文獻記憶只能用於「解釋」。

## 需要你判斷的點（按影響排序）

下列是我起草時做的化學判斷，每一條都會直接改變某些家族能拿到的分數。請逐條回覆「同意 / 改成…」。

1. **PACKING 強支持可否由「建模二聚體」達成？**（影響最大：PACKING 是最常見標籤，75 個 case）
   `environment_structure_and_emission` 條件 1 我設為接受「計算」：xTB 搜索出的二聚體接觸、接觸下的扭轉響應，算作「識別了外部約束及其約束對象」。
   全域規則另外要求聲明必須保持「建模構型」的範圍，不能說成真實晶體。若你認為建模二聚體最多只能算部分支持，就把它改成只接受「實驗觀測」，PACKING 的閉卷上限會降到 0.5。
2. **RIM 強支持**（`motion_restriction_and_consequence` 條件 1）接受「計算」：例如二聚體中的扭轉剛性變化、受限與自由的掃描對比，算作「在指定條件下顯示運動受限」。
3. **ICT 強支持**（`spectral_charge_transfer_assignment` 條件 1）接受「計算」：隱式溶劑下的極性依賴激發能變化，算作「極性依賴的發現」。
   注意 ICT 另一條強路線（空穴-電子/電荷轉移分析）本來就只接受計算，這一條只是多一條路。
4. **RADIATIVE 強支持**（`rate_decomposition` 條件 1）接受「計算」：例如由振子強度和激發能估算的輻射速率。
   若你認為計算估算的 kr 不能算「速率結果」，就改成只接受實驗觀測；但 `computed_state_balance` 仍可讓 RADIATIVE 在閉卷下拿到 1.0。
5. **ESIPT 強支持**（`proton_transfer_pathway` 條件 2）接受「計算」：包括我們的 CIS/6-31G 質子座標掃描。方法很粗糙，你是否接受它作為「激發態路徑/勢壘」證據？
6. **只靠結構的上限**：9 個家族是 0.5，RADIATIVE 和 SOKR 是 0（0.2.0 版 rubric 的路線本來就沒有純結構路線）。這意味著純文字的 Direct LLM 最多拿部分支持，這是我認為公平的設計。
7. **RACI 部分支持**（`coordinate_linked_crossing_proxy` 條件 2）接受「計算或解釋」：沿用 0.2.0 版原文「或一個明確的、針對該分子的解釋」，因此只靠結構也能拿 0.5。是否要收緊為必須有計算？
8. **只接受實驗觀測的條件**（HOST_GUEST `binding_and_emission`、PACKING `environment_switching`、ESIPT `proton_transfer_specific_observations`、PET `electron_transfer_species_or_control`、AGGREGATE `aggregate_species_observations`、SOKR `higher_state_emission_assignment`）：在 SMILES-only 設定下全部無法滿足，保留給未來提供實驗條件的子集。
9. **工具缺口**（只影響評估方的覆蓋說明，不影響打分）：HOST_GUEST 沒有主客體計算工具；RACI 沒有 CI 搜索；SOKR 沒有高激發態速率工具。這三個家族在我們的系統裡實際上拿不到強支持，論文裡應該如實寫出。

## 總覽：每個家族的分數上限

| 家族 | 只靠結構 | 結構+計算（閉卷） | 含實驗觀測 | 目前工具缺口 |
| --- | --- | --- | --- | --- |
| RIM_RIR_RIV | 0.5 | 1.0 | 1.0 | — |
| PACKING_HOST_MATRIX_CONFINEMENT | 0.5 | 1.0 | 1.0 | — |
| HOST_GUEST_INTERACTION | 0.5 | 1.0 | 1.0 | No host-guest or guest-binding calculation is available; only structural hypotheses are reachable. |
| ICT_TICT_CT | 0.5 | 1.0 | 1.0 | — |
| ESIPT_PT | 0.5 | 1.0 | 1.0 | — |
| PET_ET | 0.5 | 1.0 | 1.0 | — |
| AGGREGATE_EXCITON_EXCIMER | 0.5 | 1.0 | 1.0 | Dimer excited states require running state calculations on a modeled dimer geometry; modeled dimers are not observed aggregates. |
| RADIATIVE_RATE_STATE_BALANCE | 0.0 | 1.0 | 1.0 | — |
| TRIPLET_METAL_ENERGY_TRANSFER | 0.5 | 1.0 | 1.0 | — |
| RACI_CI_ACCESS | 0.5 | 1.0 | 1.0 | No conical-intersection search or nonadiabatic dynamics; strong routes are unreachable with the current tools. |
| SOKR_ANTI_KASHA | 0.0 | 1.0 | 1.0 | No pathway/rate calculation for higher-state emission; strong computed route needs a second computed finding the tools do not provide. |

## RIM_RIR_RIV

> Restriction of intramolecular motion, rotation, or vibration that suppresses nonradiative channels.

**強支持 (1.0) · `motion_restriction_and_consequence`**

1. [計算 / 實驗觀測] A case-specific structural, computational, or observational finding shows restriction of an identified intramolecular motion under specified conditions.
2. [解釋] Explain why restriction of that motion supports reduced nonradiative relaxation. Separate emission measurements or a full decay-rate analysis are not mandatory.

**強支持 (1.0) · `motion_pathway_calculation`**

1. [計算] A calculation identifies a specific motion-dependent nonradiative pathway for this molecule.
2. [解釋] Explain how restricting that coordinate could limit the identified pathway. A full unrestricted-versus-restricted dynamics comparison is not mandatory.

**部分支持 (0.5) · `specific_structural_motion`**

1. [結構 / 計算] Identify the location and type of a plausible intramolecular motion from the actual molecular structure.
2. [解釋] Explain how that particular motion could contribute to nonradiative relaxation; naming a rotatable bond without this link is insufficient.

**部分支持 (0.5) · `motion_sensitive_proxy`**

1. [計算 / 實驗觀測] Provide a case-specific conformation, torsion, vibration, or environmental-response finding.
2. [解釋] Connect the finding to a specific motion and its possible nonradiative role, while the full restriction-to-emission link remains unestablished.

可產生計算證據的工具（評估方註記）：`microscopic.run_torsion_snapshots`, `microscopic.run_torsion_brightness_coupling_scan`, `microscopic.run_relaxed_coordinate_scan`, `microscopic.analyze_ground_state_vibrations`, `microscopic.compare_contact_torsion_response`, `microscopic.evaluate_cis_s1_gradient`

## PACKING_HOST_MATRIX_CONFINEMENT

> External packing, host, matrix, framework, pressure, or morphology constrains structure or nonradiative channels.

**強支持 (1.0) · `environment_structure_and_emission`**

1. [計算 / 實驗觀測] A case-specific structural or computational finding identifies an external packing, host, matrix, framework, pressure, or morphology constraint and what it constrains.
2. [解釋] Explain how that constraint can affect a relevant molecular motion or emission channel. An additional measured emission change or matched environmental control is not mandatory.

**強支持 (1.0) · `environment_switching`**

1. [實驗觀測] A comparison of specified polymorphs, matrices, pressures, or other external conditions documents a structural or morphological change alongside an emission response.
2. [解釋] Explain the connection between the environmental change and the response. Full decomposition of the emission mechanism is not mandatory.

**部分支持 (0.5) · `specific_confinement_proxy`**

1. [結構 / 計算] Identify a concrete contact, steric constraint, free-volume feature, or environment-dependent structural finding for this molecule.
2. [解釋] Explain what the proposed external environment could constrain and why that constraint is relevant to emission, without claiming an unobserved packing arrangement.

**部分支持 (0.5) · `specific_structure_based_hypothesis`**

1. [結構] Identify particular structural sites or geometry that could enable an external contact or confinement effect.
2. [解釋] Explain the proposed contact and the molecular freedom or emission channel it could affect; aromaticity, planarity, or aggregate propensity alone is insufficient.

可產生計算證據的工具（評估方註記）：`microscopic.search_dimer_contacts`, `microscopic.scan_intermolecular_distance`, `microscopic.compare_contact_torsion_response`

## HOST_GUEST_INTERACTION

> Guest molecules, gases, solvents, ions, or analytes enter or bind a host, pore, or cavity and alter emission.

**強支持 (1.0) · `binding_and_emission`**

1. [實驗觀測] Binding, inclusion, occupancy, adsorption, or interaction evidence identifies the relevant host/receptor and guest pair.
2. [解釋] Explain how the specific interaction can affect the emissive unit. Direct measurement of the entire binding-to-emission chain is not mandatory.

**強支持 (1.0) · `computed_interaction_and_consequence`**

1. [計算] A calculation supports a relevant interaction for a specified host-guest pair through contact geometry together with an interaction or binding analysis, rather than just a proposed pose.
2. [解釋] Explain how that interaction can affect a chromophore, motion, or electronic property relevant to emission. An additional before/after emission calculation is not mandatory.

**部分支持 (0.5) · `specific_binding_hypothesis`**

1. [結構] Identify a concrete receptor, cavity, or binding site in the molecule and a specified or explicitly hypothetical guest interaction.
2. [解釋] Explain how that interaction could affect the emissive unit, without presenting hypothetical binding as an observed event.

**部分支持 (0.5) · `guest_response_proxy`**

1. [計算 / 實驗觀測] Provide a case-specific guest-dependent response or contact finding.
2. [解釋] Explain its connection to binding or inclusion while direct binding evidence or the emission consequence remains incomplete.

可產生計算證據的工具（評估方註記）：無

## ICT_TICT_CT

> Intramolecular charge transfer, twisted intramolecular charge transfer, or charge-transfer excited-state behavior.

**強支持 (1.0) · `excited_state_charge_transfer`**

1. [計算] A case-specific excited-state orbital, transition-density, electron-hole, charge-density, or dipole analysis shows charge redistribution between identified regions.
2. [解釋] Explain how the finding supports the proposed CT interpretation. Separate confirmation of an emission shift or quenching response is not mandatory.

**強支持 (1.0) · `spectral_charge_transfer_assignment`**

1. [計算 / 實驗觀測] Provide a case-specific spectral or polarity-dependent finding that supports CT, rather than an unexplained color or solvent response.
2. [解釋] Explain the CT assignment using the actual molecular or electronic features. Do not require ruling out every other photophysical mechanism.

**部分支持 (0.5) · `specific_structural_charge_transfer`**

1. [結構] Identify the actual donor and acceptor regions and their electronic connection in the molecule.
2. [解釋] Explain a plausible direction or distribution of charge transfer specific to those regions, while excited-state character or its photophysical consequence remains unestablished.

**部分支持 (0.5) · `charge_transfer_proxy`**

1. [計算 / 實驗觀測] Provide a case-specific orbital, charge, dipole, torsional, or polarity-dependent finding.
2. [解釋] Connect the finding to charge redistribution in the proposed mechanism; a twisted geometry or a solvent response alone is insufficient.

可產生計算證據的工具（評估方註記）：`microscopic.analyze_hole_electron_from_bundle`, `microscopic.extract_ct_descriptors_from_bundle`, `microscopic.run_frontier_orbital_partition`, `microscopic.run_targeted_charge_redistribution_analysis`, `microscopic.run_solvation_polarity_proxy`, `microscopic.run_torsion_brightness_coupling_scan`

## ESIPT_PT

> Excited-state intramolecular proton transfer or proton-transfer photophysics.

**強支持 (1.0) · `proton_transfer_pathway`**

1. [結構 / 計算] Identify a feasible proton donor-acceptor route in this molecule.
2. [計算] A relevant excited-state pathway, barrier, or tautomer-state calculation supports transfer along that route. A separate calculation or measurement of the resulting emission is not mandatory.

**強支持 (1.0) · `proton_transfer_specific_observations`**

1. [結構 / 計算] Identify a feasible donor-acceptor transfer route in this molecule.
2. [實驗觀測] Provide a transfer-specific tautomer assignment, kinetic, isotope, site-blocking/recovery, or comparable finding and explain its connection to that route. A large Stokes shift or dual emission alone is insufficient.

**部分支持 (0.5) · `specific_structural_transfer`**

1. [結構 / 計算] Identify both proton donor and acceptor, and explain their plausible intramolecular arrangement from the actual structure.
2. [解釋] Explain the proposed transfer or tautomeric change and its possible photophysical role, without asserting an uncomputed geometry or excited-state pathway.

**部分支持 (0.5) · `proton_transfer_proxy`**

1. [計算 / 實驗觀測] Provide a case-specific hydrogen-bond geometry, spectral, or tautomer-related finding.
2. [解釋] Connect it to an identified donor-acceptor transfer route, while excited-state transfer or the emission assignment remains incomplete.

可產生計算證據的工具（評估方註記）：`microscopic.scan_cis_excitation_coordinate`, `microscopic.extract_geometry_descriptors_from_bundle`

## PET_ET

> Photoinduced electron transfer or electron-transfer quenching/turn-on.

**強支持 (1.0) · `electron_transfer_energetics`**

1. [結構 / 計算] Identify electron donor, electron acceptor, and a plausible transfer direction relevant to the fluorophore.
2. [計算 / 實驗觀測] State-appropriate energy alignment, redox analysis, or a relevant calculation supports that transfer, with an explanation of its possible role in emission. Observing actual quenching or completing transfer dynamics is not mandatory.

**強支持 (1.0) · `electron_transfer_species_or_control`**

1. [實驗觀測] An assigned transient/radical-ion species or a controlled binding, protonation, or reaction result supports a particular electron-transfer channel.
2. [解釋] Connect the finding to identified transfer partners and the fluorophore. Full kinetic reconstruction and an additional emission control are not mandatory.

**部分支持 (0.5) · `specific_structural_transfer_channel`**

1. [結構] Identify the actual donor/receptor and acceptor/fluorophore regions and a plausible transfer direction.
2. [解釋] Explain why electron transfer could affect this fluorophore's emission, while energetics, transferred species, or channel switching remains unestablished.

**部分支持 (0.5) · `electron_transfer_proxy`**

1. [計算 / 實驗觀測] Provide a case-specific energy-level, binding-state, or quenching-response finding.
2. [解釋] Connect the finding to identified transfer partners and direction, while a complete transfer-to-emission link remains missing.

可產生計算證據的工具（評估方註記）：`microscopic.estimate_vertical_ip_ea`, `microscopic.run_frontier_orbital_partition`, `microscopic.analyze_hole_electron_from_bundle`

## AGGREGATE_EXCITON_EXCIMER

> Aggregate exciton, H/J aggregation, excimer, exciplex, or new intermolecular excited-state emission.

**強支持 (1.0) · `aggregate_species_observations`**

1. [實驗觀測] A case-specific aggregate- or concentration-dependent finding supports a new intermolecular excited or emissive species rather than only aggregate formation.
2. [解釋] Explain the species assignment using the relevant spectral, lifetime, time-resolved, or excitonic evidence. Do not require all these measurements together; size, close contacts, or a red shift alone is insufficient.

**強支持 (1.0) · `aggregate_excited_state_calculation`**

1. [計算] A calculation for a specified dimer or aggregate configuration identifies intermolecular excited-state character, coupling, or excimer/exciplex formation rather than only a ground-state contact.
2. [解釋] Explain how the result supports the proposed aggregate-state mechanism within the modeled configuration. Experimental confirmation of the aggregate or its emission is not mandatory.

**部分支持 (0.5) · `specific_contact_hypothesis`**

1. [結構 / 計算] Identify particular molecular surfaces, partners, or a contact/dimer geometry that could support intermolecular excited-state interaction.
2. [解釋] Explain the proposed interaction using those features, while aggregate-state electronic or emissive-species evidence remains absent.

**部分支持 (0.5) · `aggregate_species_proxy`**

1. [計算 / 實驗觀測] Provide a case-specific aggregate-dependent spectral or contact finding.
2. [解釋] Explain why it suggests intermolecular excited-state interaction, while species assignment or separation from confinement effects remains incomplete.

可產生計算證據的工具（評估方註記）：`microscopic.search_dimer_contacts (+ excited states on the selected dimer geometry)`

## RADIATIVE_RATE_STATE_BALANCE

> Emission changes arise from radiative rate, oscillator strength, bright/dark state balance, state population, or Kr/knr balance.

**強支持 (1.0) · `rate_decomposition`**

1. [計算 / 實驗觀測] Provide a case-specific radiative/nonradiative rate result, or compatible lifetime and quantum-yield findings that support a rate/state interpretation under specified conditions.
2. [解釋] Explain the particular radiative channel or state-balance contribution supported by the result. Full decomposition of all decay pathways or an additional matched control is not mandatory.

**強支持 (1.0) · `computed_state_balance`**

1. [計算] An analysis of relevant states or conformations shows a specific bright/dark-state relation or radiative-channel change using transition properties, state character, ordering, or populations.
2. [解釋] Explain the resulting rate/state-balance interpretation. Full population dynamics or measured emission validation is not mandatory; one isolated oscillator strength without a state/rate relation is insufficient.

**部分支持 (0.5) · `specific_state_or_rate_proxy`**

1. [計算 / 實驗觀測] Provide a case-specific oscillator-strength, lifetime, quantum-yield, transition, ordering, or population finding.
2. [解釋] Explain its relevance to a particular radiative channel or state-balance hypothesis, while a matched comparison or full rate/state analysis remains incomplete.

可產生計算證據的工具（評估方註記）：`microscopic.run_bright_dark_state_ordering`, `microscopic.run_baseline_bundle`, `microscopic.run_conformer_state_probe`, `microscopic.run_targeted_transition_dipole_analysis`

## TRIPLET_METAL_ENERGY_TRANSFER

> Triplet photophysics, RTP, TADF, heavy atom effect, lanthanide antenna effect, ligand-to-metal transfer, or metal-centered emission.

**強支持 (1.0) · `triplet_state_pathway`**

1. [計算 / 實驗觀測] A case-specific state, spectral, kinetic, coupling, or sensitization finding directly supports the proposed triplet-associated state or pathway.
2. [解釋] Explain the particular triplet interpretation supported by that finding. Multiple independent validation modalities or complete ISC/rISC kinetics are not mandatory; a heavy atom or long lifetime alone is insufficient.

**強支持 (1.0) · `metal_sensitization`**

1. [計算 / 實驗觀測] A case-specific state-energy, sensitization, excitation/emission, or transfer analysis supports a particular ligand-to-metal energy-transfer route.
2. [解釋] Identify the ligand and metal states and explain the transfer interpretation. Complete transfer-efficiency or kinetic measurements are not mandatory; metal presence alone is insufficient.

**部分支持 (0.5) · `specific_triplet_hypothesis`**

1. [結構 / 計算] Identify case-specific structural or electronic features relevant to a proposed triplet-associated state or pathway.
2. [解釋] Explain the particular state or transition those features could affect, while direct state, kinetic, or transfer evidence remains incomplete.

**部分支持 (0.5) · `triplet_or_transfer_proxy`**

1. [計算 / 實驗觀測] Provide a case-specific gap, orbital, oxygen/temperature response, delayed signal, or metal-emission finding.
2. [解釋] Connect it to an identified triplet or transfer hypothesis, while a decisive state assignment or pathway link remains absent.

可產生計算證據的工具（評估方註記）：`microscopic.compare_singlet_triplet_states`, `microscopic.compare_spin_states_pyscf`, `microscopic.evaluate_spin_orbit_couplings`

## RACI_CI_ACCESS

> Restricted access to conical intersections or conical-intersection-mediated nonradiative decay.

**強支持 (1.0) · `crossing_access_and_restriction`**

1. [計算] An appropriate calculation identifies a conical intersection or relevant excited/ground-state crossing connected to a specific molecular relaxation coordinate.
2. [解釋] Explain how that coordinate supports crossing-mediated relaxation. If reduced access is claimed, provide a relevant constraint or coordinate-access finding, without requiring a full solution-versus-aggregate comparison.

**強支持 (1.0) · `nonadiabatic_pathway`**

1. [計算] Nonadiabatic dynamics or comparable excited-state analysis supports a crossing-mediated relaxation route for the molecule.
2. [解釋] Explain the relevant motion or access mechanism. Evidence for a restriction claim must concern restricted access, but a complete emission-yield calculation or a second environmental simulation is not mandatory.

**部分支持 (0.5) · `coordinate_linked_crossing_proxy`**

1. [結構 / 計算] Identify a specific torsional, flapping, puckering, ring-opening, or other coordinate in this molecule.
2. [計算 / 解釋] Provide a coordinate-dependent state-gap/electronic finding, or an explicit case-specific explanation linking that coordinate to approach toward a crossing.

可產生計算證據的工具（評估方註記）：`microscopic.evaluate_cis_s1_gradient (proxy only)`, `microscopic.scan_cis_excitation_coordinate (proxy only)`

## SOKR_ANTI_KASHA

> Suppression of Kasha's rule, anti-Kasha emission, or higher-state emission.

**強支持 (1.0) · `higher_state_emission_assignment`**

1. [實驗觀測] Identify the relevant state ordering and provide state-resolved emission, kinetic, or comparable evidence assigning emission above the lowest excited state of the relevant multiplicity.
2. [解釋] Explain the higher-state-emission interpretation. Full population kinetics or a quantified competition with every lower-state relaxation channel is not mandatory.

**強支持 (1.0) · `computed_higher_state_emission`**

1. [計算] State ordering and transition properties identify a candidate higher emissive state for this molecule.
2. [計算] A relevant pathway, rate, dynamics, or comparable mechanism-specific result supports higher-state emission before relaxation to lower states. A complete population model or experimental confirmation is not mandatory; higher-state brightness alone remains partial support.

**部分支持 (0.5) · `ordered_state_proxy`**

1. [計算] Provide case-specific state ordering together with transition or oscillator-strength information identifying a candidate higher emissive state and its relation to the lower state.
2. [解釋] Explain the higher-state-emission hypothesis, while population dynamics or competition with internal conversion remains unestablished.

**部分支持 (0.5) · `state_linked_emission_proxy`**

1. [實驗觀測] Provide a case-specific excitation-dependent, time-resolved, or other state-related observation.
2. [解釋] Connect it to an identified higher-state assignment, while that assignment or its kinetic support remains incomplete.

可產生計算證據的工具（評估方註記）：`microscopic.run_bright_dark_state_ordering (ordering proxy only)`
