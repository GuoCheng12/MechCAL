# 在本地 Mac 上用 InstSci 下載論文（交給本地 Claude Code 執行）

**給本地 Claude Code 的說明**：這份文件描述一個任務。請按步驟執行；凡是需要登錄、雙重認證、驗證碼的地方，停下來讓用戶在瀏覽器裡自己完成。

## 任務

用 InstSci（https://github.com/Rimagination/instsci ）通過用戶自己的 CUHK 訂閱權限，下載 `dois_pilot.txt` 裡 6 篇論文的正文 PDF 和支持信息（SI），整理好後打包，由用戶上傳到 Google Drive。

這是 PhotoMechBench 數據集擴充的試點：先確認這條下載路線可行，之後每批 20–30 篇。

## 規則（必須遵守）

1. **不碰用戶的密碼。**不要讓用戶把 CUHK 帳號密碼告訴你，也不要寫進任何文件、命令或配置。登錄一律由用戶在 InstSci 彈出的瀏覽器窗口裡手動完成。
2. **小批量、不並行。**一次只跑一個下載任務，保留 InstSci 默認的延遲，不要調短，不要開多個瀏覽器同時下載。
3. **不要繞過驗證碼。**遇到「Are you a robot?」或類似驗證，讓用戶手動處理；處理不了就跳過這篇，記下來。
4. **不用任何 Sci-Hub 類的來源。**只用開放獲取和 CUHK 的合法訂閱。
5. **固定版本。**安裝固定的提交 `7a233a4df767c7fe64f03491ef1b6abe84d08c34`（Release v0.2.0，2026-09-29），不要裝最新的未審核代碼。
6. 安裝前快速看一下它會做什麼（README、`pyproject.toml` 的依賴），有異常就先問用戶。

## 步驟

### 1. 準備

- 確認 Python 版本在 3.10–3.13 之間（3.14 不支持瀏覽器流程）。
- 安裝 pipx 或 uv（任選一個）。
- **讓用戶連上 CUHK VPN**（或者在校園網裡）。CUHK 不在 InstSci 內置的學校清單裡，VPN 是最簡單的取得訂閱權限的方式。

### 2. 安裝

```bash
pipx install "git+https://github.com/Rimagination/instsci.git@7a233a4df767c7fe64f03491ef1b6abe84d08c34"
```

（或 `uv tool install` 同一個地址。）第一次使用瀏覽器時，InstSci 會自己下載它的 CloakBrowser。

### 3. 配置

```bash
instsci setup --institution-en "The Chinese University of Hong Kong" --institution-cn "香港中文大學" --email <用戶自己的郵箱>
instsci setup --check
```

- 郵箱只用於開放獲取的元數據服務（Unpaywall 等），由用戶自己填。
- 用 `instsci setup --help`、`instsci papers --help` 確認實際的參數名稱；如果和這裡寫的不一樣，以 `--help` 為準。

### 4. 先拆分出版社，再逐個出版社下載

```bash
instsci queue dois_pilot.txt --output ./runs/pilot-queue
```

這一步不登錄、不下載，只按出版社分組。6 篇預計分成：ACS 1、Wiley 3、Elsevier 1、MDPI 1。

然後每組分別跑，每次一組，加上支持信息：

```bash
instsci papers ./runs/pilot-queue/<組名>.csv --publisher <出版社> --institution "The Chinese University of Hong Kong" --include-supplementary --output ./runs/pilot-<出版社>
```

- 彈出瀏覽器、跳到 CUHK 登錄頁時，讓用戶自己登錄和完成雙重認證。
- 同一個出版社的後續論文會沿用這個登錄會話。
- **Elsevier**：InstSci 自己的測試顯示 ScienceDirect 網頁路線常被驗證碼擋住。如果失敗，就請用戶在瀏覽器裡手動下載這一篇（連著 VPN 打開 DOI 鏈接，點「Download PDF」，SI 在文章頁的「Supplementary data」裡）。

### 5. 檢查

對每一篇確認：
- 正文 PDF 能打開，第一頁的標題和 DOI 對得上；
- SI 有沒有下載到（不是每篇都有 SI）。

第一批試點（2026-11-15）遇到的問題，InstSci 都**沒有報錯**，只能靠上面的檢查發現：
- Chemistry – A European Journal 等 Chemistry Europe 期刊：加了 `--include-supplementary` 時，InstSci 可能把 SI 存成正文，還標為已核實。遇到時，不加 `--include-supplementary` 重新下載正文。
- Wiley 頁面的 `.txt`、`.css` 文件和期刊廣告 PDF 會被當成 SI 存下來，打包前刪掉。
- MDPI：InstSci 可能抓到期刊宣傳單（flyer）而不是論文。MDPI 是開放獲取，直接在瀏覽器裡下載更快。
- Elsevier：能打開 PDF 但存不下來時，在瀏覽器裡手動下載。
- 如果代理或 VPN 的默認節點連不上某個出版社（例如 `ERR_CONNECTION_CLOSED`），換一個能連上學校網絡的節點。

### 6. 整理並打包

把文件放進一個文件夾 `PhotoMechBench_pilot_batch1/`：

- 正文：`<DOI，把 / 換成 _>.pdf`，例如 `10.1002_adom.71933.pdf`
- SI：`<同上>_SI_1.pdf`、`<同上>_SI_2.zip`……（保留原擴展名）
- `report.csv`：每篇一行，欄位 `doi,publisher,main_pdf,si_files,route,notes`
  - `route`：`open_access`、`instsci_institutional`、`manual`、`failed`
  - `notes`：遇到的問題（驗證碼、需要重新登錄、找不到 SI……）

壓縮成 `PhotoMechBench_pilot_batch1.zip`，用戶上傳到 Google Drive，把分享鏈接發回服務器上的 Claude。

### 7. 收尾

- InstSci 的登錄會話會在本地保存一段時間；這是正常的，但不要把會話文件或 cookie 打包進壓縮包。
- 如果整個流程裡有任何一步不確定，問用戶，不要猜。
