# -*- coding: utf-8 -*-
# GitHub Actions 版：由 Colab 版修改而來
# 修改處僅三項：1) API key 改由環境變數 GEMINI_API_KEY 讀取
#              2) 輸出檔名改為 index.html（GitHub Pages 首頁）
#              3) 移除 Colab 專屬註解
# 其餘邏輯與 prompt 與 Colab 版完全相同。

import html as html_lib
import json
import os
import time
from dataclasses import dataclass
from datetime import timedelta, datetime
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from google import genai
from google.genai import errors, types

# ==========================================
# 0. 基本設定
# ==========================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    raise SystemExit("錯誤：找不到環境變數 GEMINI_API_KEY（請確認 GitHub Secrets 設定）")

# 主模型維持原設定；備援模型採固定白名單，避免自動選到 preview／實驗模型。
MODEL_NAME = "gemini-3.5-flash"
FALLBACK_MODELS = []
MODEL_CHAIN = [MODEL_NAME] + FALLBACK_MODELS

client = genai.Client(api_key=GEMINI_API_KEY)
print(f"🎯 主要模型：{MODEL_NAME}（已啟用 Google Search 檢索）")
print(f"🛟 備援模型鏈：{FALLBACK_MODELS if FALLBACK_MODELS else '（無，僅主模型＋重試）'}")

SEARCH_TOOL = types.Tool(google_search=types.GoogleSearch())
GEN_CONFIG = types.GenerateContentConfig(tools=[SEARCH_TOOL])

RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}
PRIMARY_RETRIES = 3    # 主模型：最多重試 3 次（等待 10/20/40 秒）
FALLBACK_RETRIES = 1   # 備援模型：各重試 1 次


class EmptyResponseError(RuntimeError):
    """模型回傳空白內容（可能因安全阻擋或回應截斷），視為可重試／可換援。"""


@dataclass
class ModelResult:
    text: str
    model: str
    sources: list
    queries: list
    error: str = ""


def _get_error_code(exc):
    """盡量從 google.genai.errors.APIError 取得 HTTP 狀態碼。"""
    code = getattr(exc, "code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def _is_retryable_error(exc):
    """判斷是否適合重試或切換到備援模型。"""
    if isinstance(exc, EmptyResponseError):
        return True
    if isinstance(exc, errors.APIError):
        return _get_error_code(exc) in RETRYABLE_HTTP_CODES

    exc_name = type(exc).__name__.lower()
    return "timeout" in exc_name or "connection" in exc_name


def extract_grounding_metadata(response):
    """擷取 Generate Content API 回傳的實際搜尋查詢與 grounding 來源。"""
    queries = []
    sources = []
    seen_uris = set()

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return queries, sources

    metadata = getattr(candidates[0], "grounding_metadata", None)
    if metadata is None:
        return queries, sources

    for query in (getattr(metadata, "web_search_queries", None) or []):
        query = str(query).strip()
        if query and query not in queries:
            queries.append(query)

    for chunk in (getattr(metadata, "grounding_chunks", None) or []):
        web = getattr(chunk, "web", None)
        if web is None:
            continue

        uri = str(getattr(web, "uri", "") or "").strip()
        title = str(getattr(web, "title", "") or "未命名來源").strip()
        if not uri or uri in seen_uris:
            continue

        seen_uris.add(uri)
        sources.append({"title": title, "uri": uri})

    return queries, sources


def _call_one_model(model_name, prompt, retries, label):
    """對單一模型呼叫，含指數退避重試；耗盡後拋出最後一個例外。"""
    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=GEN_CONFIG,
            )
            output_text = (getattr(response, "text", None) or "").strip()
            if not output_text:
                raise EmptyResponseError("模型回傳空白內容，可能因安全阻擋或回應不完整。")
            return response, output_text
        except Exception as exc:
            if attempt < retries and _is_retryable_error(exc):
                wait = min(10 * (2 ** attempt), 60)
                code = _get_error_code(exc)
                code_text = f"HTTP {code}" if code is not None else type(exc).__name__
                print(f"   ⏳ {label}［{model_name}］暫時性錯誤（{code_text}），"
                      f"{wait} 秒後重試 ({attempt + 1}/{retries})...")
                time.sleep(wait)
            else:
                raise


def call_model(prompt, label=""):
    """依模型鏈呼叫；每個模型先自行重試，耗盡後對暫時性錯誤切換下一個模型。"""
    last_error = None

    for idx, model_name in enumerate(MODEL_CHAIN):
        retries = PRIMARY_RETRIES if idx == 0 else FALLBACK_RETRIES
        try:
            response, output_text = _call_one_model(model_name, prompt, retries, label)
            queries, sources = extract_grounding_metadata(response)

            if model_name != MODEL_NAME:
                print(f"   🛟 {label} 改由備援模型完成：{model_name}")
            if not sources:
                print(f"   ⚠️ {label} 未取得 API grounding 來源，發布前需特別複核模型列出的網址。")

            return ModelResult(
                text=output_text,
                model=model_name,
                sources=sources,
                queries=queries,
            )

        except Exception as exc:
            last_error = exc
            can_fallback = idx < len(MODEL_CHAIN) - 1
            if can_fallback and _is_retryable_error(exc):
                code = _get_error_code(exc)
                code_text = f"HTTP {code}" if code is not None else type(exc).__name__
                print(f"   🔁 {label}［{model_name}］重試耗盡（{code_text}），改試下一個模型...")
                continue
            raise

    raise last_error or RuntimeError("所有模型均呼叫失敗。")
# ==========================================
# 1. 本期期間（上週五 ~ 本週四，供六幣別模板使用；
#    Spot/Swap/Forward 段的期間由該 prompt 自行認定並標示）
# ==========================================
TAIPEI_TZ = ZoneInfo("Asia/Taipei")
now_taipei = datetime.now(TAIPEI_TZ)
today = now_taipei.date()

this_monday = today - timedelta(days=today.weekday())
last_friday = this_monday - timedelta(days=3)
this_thursday = this_monday + timedelta(days=3)
period_end = this_thursday if today >= this_thursday else today

trade_end = this_thursday if today >= this_thursday else today
TRADE_PERIOD_STR = f"{this_monday.strftime('%Y/%m/%d')} – {trade_end.strftime('%Y/%m/%d')}"   # 通週交易區間：本週一~週四
NEWS_PERIOD_STR = f"{last_friday.strftime('%Y/%m/%d')} – {trade_end.strftime('%Y/%m/%d')}"    # 事件敘事範圍：上週五~本週四
PERIOD_STR = NEWS_PERIOD_STR
TODAY_STR = today.strftime("%Y/%m/%d")
NOW_STR = now_taipei.strftime("%Y-%m-%d %H:%M")
print(f"📅 通週交易區間（週一~週四）：{TRADE_PERIOD_STR}")
print(f"📰 事件敘事範圍（上週五~週四）：{NEWS_PERIOD_STR}（台北日期：{TODAY_STR}）")

# ==========================================
# 2. Prompt A：Spot / Swap / Forward（使用者提供之檢索版 prompt，原文置入）
# ==========================================
PROMPT_SSF = ("""今天日期：""" + TODAY_STR + """。

你是台灣銀行匯兌交易室的週報撰稿人。請先上網檢索本期資料，再依固定格式一次撰寫「FX Flow Weekly」的 Spot、Swap、Forward 三段（USDTWD），繁體中文、專業金融書面語。三段各自為一段連續文字，不分點。
【時間軸定義（三段共用，務必嚴格區分兩種期間）】
一、通週交易區間：僅計「本週一至本週四」之行情（本期為 __TRADE__），不含上週五。所有「通週交易區間」「通週高低點」等數字一律以此期間為準。
二、事件敘事範圍：涵蓋「上週五至本週四」（本期為 __NEWS__）。上週五發生之事件必須明寫「上週五」，不得併入「週初」；其行情數字亦不得計入通週交易區間。
三、週內定義：週初＝本週一；週中段＝本週二至週三；週後段＝本週四。
註：美國市場週四交易日於台北時間週五清晨收盤，其行情與盤中事件仍計入本期「週後段」；美國週五交易屬下一期，不得納入。
時間推進用語一律為「上週五」「週初」「進入週中段」「步入週後段」，除「上週五」外不得使用「週一/週二」等日別稱謂，亦不得使用「週前段」「週中」等混用寫法。
開始撰寫前，請先於輸出最上方以兩行分別標明你所認定的「通週交易區間期間」與「事件敘事範圍」實際日期，供使用者核對是否與其認知一致。
───────────────────────────── 【檢索指示】
■ 需檢索的項目 Spot 段： 新臺幣兌美元通週交易區間、週末收盤價（以台北收盤匯率為準；若僅取得離岸報價，須註明口徑） 外資買賣超台股金額（具體到億元），資料來源以臺灣證交所公布之三大法人買賣超為優先 影響台幣之國際面事件與數據 台灣本地事件（利率決議、台指期結算、連假等）
Swap 段： 美國兩年期公債殖利率通週交易區間與走勢（以官方或主要財經數據源為準） 影響殖利率之事件與數據（含通膨數據之實際值與市場預期值） 台幣隔夜拆款利率水位與趨勢 台灣資金面之公開可知因素（季底、月底、繳稅期等季節性因素）
Forward 段： 美元指數（DXY）通週交易區間與走勢形態 影響美元之事件與數據（含實際值與市場預期值） 其他主要央行動向 下週重大事件（選用）：僅在下週有足以主導行情之重大事件（如 FOMC、美國 CPI、非農就業等級別）時，擇「一項」最重要者檢索；無此等級事件則不檢索、不撰寫。
■ 用語統一規則 一律稱「通週交易區間」。不得使用「已完成交易區間」「本週迄今區間」「週通交易區間」等替代寫法。若截至檢索時點區間資料不完整，名稱仍用「通週交易區間」，並依下方「資料不完整標示規則」加註截止時點，不得另創名稱。
■ 檢索來源優先順序 官方或一手來源：美國聯準會 FRED、美國勞工統計局、臺灣證交所、中央銀行 主要財經媒體：Reuters、Bloomberg、CNBC、CNN、Financial Times、金十數據 次級整理型網站僅在前兩者無法取得時使用，且須於來源表中註明其為次級來源
■ 檢索結果的處理規則（此節為最重要規則，請嚴格遵守） 查不到就明寫「未查得」，不得省略、不得推測。 某項目檢索後無法取得可靠數字時，正文該處寫「（未查得）」或整句省略，並於來源表中該項標明「未查得」。嚴禁以推估、印象或相近數據替代。
資料不完整標示規則（適用三段全部數據，不得遺漏）： 凡數據因檢索時點關係尚未涵蓋完整期間者——例如美國週四（週後段）行情於檢索時點尚未收盤、官方日終值或統計數據尚未公布——正文與來源表「兩處」均必須明確標示，例如：「（週後段美國時段截至檢索時點尚未收盤，日終值未公布）」「（區間僅涵蓋至 X/X，完整通週區間未查得）」。嚴禁以部分期間之資料冒充通週數據，嚴禁靜默略過不標。若通週區間因此不完整，須寫明「通週交易區間未查得（截至 X/X）」或以截至時點之區間加註說明，二擇一，但不得不標。
來源互相矛盾時，不得自行擇一後靜默帶過。 同一數據在不同來源出現不一致數值時，正文採用較可靠來源（依上述優先順序）的數字，並必須於來源表該項後方加註：「（來源不一致：A源為X、B源為Y，已採A源，建議複核）」。
正文中出現的每一個數字，都必須能在來源表中找到對應項目與來源名稱。不得出現無來源支撐的數字。
嚴禁在任何輸出中提供網址或 URL（你無法取得真實網址，寫出的網址必為拼湊）。若僅取得搜尋結果摘要而未讀取原文，於該項註明「（僅摘要）」。 付費牆來源（Bloomberg、FT、Reuters 等）若僅能取得標題或摘要，敘事僅寫到摘要所支持的程度，不得補寫付費牆內的細節，並於來源表註明「（僅摘要）」。
不得檢索、不得生成的欄位：下列項目為銀行內部資料，網路上不存在，一律保留「【待填】」字樣，嚴禁自行編寫或以市場通則推測： 結售、結購交易量及其主因 B/S、S/B 交易量及其主因 預購、預售交易量及其主因 實需面動態（出口商拋匯、進口商觀望等行內觀察） NDF 報價 若使用者已於下方【使用者提供欄位】填入，則依其提供內容撰寫。
───────────────────────────── 【使用者提供欄位】 以下為行內資料，由使用者填寫；未填寫者於正文保留「【待填】」，不得生成： 結售交易量變化：較上週【增/減 X 成】，主因：____ 結購交易量變化：較上週【增/減 X 成】，主因：____ 實需面動態：____ 市場心態收尾（選填）：____ B/S 交易量變化：較上週【增/減 X 成】，主因：____ S/B 交易量變化：較上週【增/減 X 成】，主因：____ 預購交易量變化：較上週【增/減 X 成】，主因：____ 預售交易量變化：較上週【增/減 X 成】，主因：____ NDF 報價（選填）：____
───────────────────────────── 【撰寫規則】
■ 通則（三段皆適用） 每段各為一段連續文字，不分點、不使用條列。 時間推進用語一律為「上週五」「週初」「進入週中段」「步入週後段」。 嚴禁逐階段流水帳式羅列收盤價／日終值（禁例：「週初收於32.288及32.248，進入週中段轉貶至32.345及32.375，步入週後段回升至32.266」）。點位與數值只允許出現在因果句之中，作為事件驅動後的關鍵落點（如通週高低點、突破位、週末收盤），每段引用的具體點位以 2-4 個為度，其餘走勢以方向與幅度描述即可。此規則同樣適用於 Swap 段之殖利率與 Forward 段之美元指數。 不得虛構未檢索到或未提供的數字與事件；無資料處依前述規則標「【未查得】」「【待填】」或「【尚未公布】」，不得推測補寫。 每段各選 2-4 句「事件→匯價/殖利率」核心因果句，以 粗體 標示。 跨段數字一致性：三段若引用同一筆數據，其數值、預期值與描述須完全一致，不得互相矛盾。 三段之間不加入總結、前言或過場說明，依序輸出即可。
■ Spot 段規則 首段固定以【本週新台幣（定調全週走勢格局），通週交易區間落在 XX.XXX~XX.XXX。】破題。正文依時序（上週五若有重大事件明寫「上週五」，其後週初、進入週中段、步入週後段）推進，敘事主體緊扣台幣匯價，每一波段須嚴格遵循「外部事件（如國際事件/指標發布）→ 市場情緒與台股反應、籌碼動向（外資匯出入/進出口商/壽險業者/央行支撐）→ 台幣升貶結果」的因果傳導鏈。外資買賣超與匯出入為必要素材，金額具體到億元；匯價一律精確至小數後三位（或稱 X.XXX 元上方/附近）。文中不逐段報收盤價，僅描述升貶程度（如微升、量縮小貶、收斂貶幅），唯有「創…新高/新低」時才具體點出該點位（破題句之通週交易區間不在此限）。因果連接詞限用「受…影響、帶動、致使、加上、主因」；轉折一律用「惟」；若股匯走勢相反需使用「股匯不同調/脫鉤」。文末最後固定以【在交易量方面，結售量較上週【待填】，主因【待填】；結購交易量較上週【待填】，主因【待填】。】作結尾；若本週交易天數與上週不同，可在「在交易量方面」之後先以一句話簡述天數差異成因（如連假、颱風假），再接結售及結購量。正文約 500 字，題材充分涵蓋、由編輯自行取捨重要性。
■ Swap 段規則 你在本段扮演專業的金融市場交易員與分析師，負責撰寫「SWAP市場週報」；請以全球債市與台灣貨幣市場之檢索結果為素材，模仿專業法人語氣，撰寫結構嚴謹、邏輯連貫的週報文案。【寫作格式與字數限制（最高指導原則，嚴格遵守）】本段必須呈現為「單一一個完整段落」，嚴禁換行、嚴禁分段、絕不能使用任何列點標號（如 1. 2. 3.）或區塊標題，各分析面向透過指定關鍵字自然銜接。【內容結構要求】依序將以下三大部分流暢串接於同一段落中——第一部分（美債殖利率走勢）：首句必須點出「通週整體走勢（如：震盪走高、高檔盤整、走低）」與「通週交易區間」；根據檢索資訊過濾雜訊，依時間軸（上週五若有重大事件明寫「上週五」，其後週初、進入週中段、步入週後段）順序梳理兩年期公債殖利率走勢；必須精準連結驅動因素（地緣政治風險變化、能源價格等），若檢索到總經數據（實際值與預期值）或央行官員對降息/升息的態度，必須簡明扼要地寫出。第二部分（台幣利率與資金面因素）：必須直接以「台幣利率部分，」或「台幣利率方面，」起頭進行句意轉折；總結本週拆款與短票利率的整體趨勢（如：持平、微幅下滑、走升）；從檢索結果萃取並解釋資金面寬鬆或緊俏的因素，如台股波動與交割款需求、外資動向、季節性因素（月底/季底效應、繳稅期）、票券商與銀行間的資金供需狀況。第三部分（交易量結尾）：固定以「在交易量方面，B/S 交易量較上週【待填】，主因【待填】；S/B 交易量較上週【待填】，主因【待填】。」作結。正文約 500 字，題材充分涵蓋、由編輯自行取捨重要性。
■ Forward 段規則 敘事主體是美元指數；結構：美元走勢總述+通週交易區間 → 依時間軸推進 →（選用：NDF）→ 預購/預售交易量收尾。 每波敘事為「事件/數據→美元指數反應」因果句，數據必附具體數字及與市場預期比較（優於/符合/略低於預期）。 下週觀察點預設「省略不寫」；僅當下週有 FOMC、美國 CPI、非農就業等足以主導行情之重大事件時，方得以一句帶過「一項」最重要事件，不得羅列行事曆、不需寫多項。 因果連接詞：「受…影響」「帶動」「推動」「使」「削弱」「惟」「然因」「加上」「主因」「顯示」；轉折用「惟」或「然因」。 若有 NDF：「NDF市場方面，USDTWD【期別】NDF報價【方向】，顯示市場短線上對美元【偏強/偏弱】預期【抬升/降溫】。」 交易量固定拆預購/預售，各給成數變化+主因，預售歸因與美元匯價方向掛鉤。 正文約 500 字，題材充分涵蓋、由編輯自行取捨重要性。
───────────────────────────── 【輸出格式】 嚴格依下列格式輸出：
本期涵蓋期間：YYYY/MM/DD – YYYY/MM/DD
【Spot】 （正文一段）
【Swap】 （正文一段）
【Forward】 （正文一段）
【資料來源與數據對照表】 供使用者逐項複核。每一筆須含：數據項、正文中採用之數值、來源名稱（如：臺灣證交所、Reuters、鉅亨網）。注意：你在檢索模式下無法取得真實網址，因此「嚴禁」在輸出中提供任何網址或 URL；真實網址由系統另行附上（grounding 來源清單），使用者將以該清單複核。 排序依各該數據在該段正文中首次出現的先後，不依來源類型或重要性重排；同一來源支撐多處時以首次出現位置為準，僅列一次。
Spot: 數據項 採用數值 來源 備註
Swap: 數據項 採用數值 來源 備註
Forward: 數據項 採用數值 來源 備註
備註欄用途：標示「未查得」「僅摘要」「次級來源」「來源不一致：A源為X、B源為Y，已採A源，建議複核」「口徑：離岸報價」「截至檢索時點尚未公布/尚未收盤，僅涵蓋至X/X」等。
【本期未取得項目】 逐條列出：一、檢索後仍未取得的項目；二、因檢索時點關係尚未公布或期間不完整的項目（註明截至時點）；三、因屬行內資料而保留【待填】的項目。三類均不得遺漏，供使用者補齊或複核。""").replace("__TRADE__", TRADE_PERIOD_STR).replace("__NEWS__", NEWS_PERIOD_STR)

# ==========================================
# 3. Prompt B：六幣別「本週展望」模板（依使用者七週樣本歸納版）
# ==========================================
CCY_WRAPPER = """今天日期：{today}。
通週交易區間期間：{trade_period}（僅本週一至週四之行情，區間數字以此為準）。
事件敘事範圍：{news_period}（上週五至本週四；上週五之事件必須明寫「上週五」，不得併入「週初」，其行情不計入通週交易區間）。
週內定義：週初＝本週一；週中＝本週二至週三；週後段＝本週四。美國週四行情於台北時間週五清晨收盤，計入本期週後段；美國週五屬下一期。

你是台灣的銀行匯兌交易室週報撰稿人。請先上網檢索【{ccy_name}】本期的走勢、通週交易區間、關鍵事件與經濟數據，再依下方模板撰寫「本週展望」段落，繁體中文、專業金融書面語。

【檢索與撰寫規則（嚴格遵守）】
1. 模板中【】為填空處，（）內為選用段落，視當週檢索結果取捨；模板結構與慣用句式須保留。
2. 正文中出現的每一個數字都必須來自本次檢索結果；查不到的項目寫「（未查得）」或整句省略，嚴禁以推估、印象或相近數據替代。涵蓋期間內尚未收盤或尚未公布的數據必須明確標示，嚴禁以部分期間資料冒充通週數據。
3. 匯價精確到慣例位數：{decimals}。每個匯價動作都附具體價位或關口，不寫「大幅」「明顯」而不給數字。
4. 每句採「事件→市場解讀→匯價反應＋價位」三段式因果鏈，不單獨陳述事件。美國數據為各幣別共用素材，但解讀角度依本幣別調整（利差、避險、風險偏好、商品需求）。
5. 展望段正反情境並陳後收斂，不做單邊斷言。結尾慣例：{ending}。「預估區間」的具體數字只有在檢索到市場人士或機構之預估並能註明來源時才可填寫；否則該處寫【待填：由撰稿人判斷】，嚴禁自行編造預測區間。
6. 正文約 500 字，題材充分涵蓋、由編輯自行取捨重要性；一至數段連續文字，不分點。
7. 本幣別必查題材清單（寫作前逐項確認當週是否有題材，有題材者儘量寫入）：{checklist}
8. 慣用語句庫（優先取用）：格局定性：延續緩貶格局／震盪走貶／震盪走升／止升轉貶／區間窄幅震盪／高檔震盪／先升後貶／先抑後揚／震盪下行／築底反彈／偏弱震盪。轉折與因果：惟／然而／然隨／疊加／加上／帶動／推動／使／促使／隨後／時至本週後段／迫使／限制／壓抑／削弱／支撐／獲撐。匯價動作：一度觸及／一度下探／升抵／貶破／跌破／站穩／回吐部分漲幅／收斂跌幅／徘徊於X上下／背靠X一線短時獲撐。展望結構：「若【情境A】，則【方向】；反之若【情境B】，【方向】。整體而言，在【背景】下，預期下週【貨幣對】將於【區間】【定性】震盪。」
9. 正文結束後，另附「資料來源」清單：每筆含數據項、採用數值、來源名稱（如：臺灣證交所、Reuters、鉅亨網）。嚴禁提供任何網址或 URL（檢索模式下你無法取得真實網址，寫出的必為拼湊）。僅有摘要者註明「（僅摘要）」；查無資料的重要項目列為「未查得」。

【模板】
{template}"""

TPL_TWD = """本週台幣【延續貶勢／震盪走升／止升轉貶／呈區間窄幅震盪格局】，（並於週【X】【描述性高低點，如升抵逾三個月新高】，）主因【外資賣超台股並匯出／外資回補匯入／台股回檔／股利匯出／美元走強】，通週區間落在【　】–【　】（，週線【貶/升】幅【X%】）。
上週五【若有重大事件：事件＋台股/外資反應＋台幣價位】。
週初受【台股漲跌＋原因（AI題材／MSCI／重大財報／颱風休市補漲）】影響，外資【大舉匯入/匯出】，推動台幣【動作＋價位】；惟【實需反向力量：進口商、油款、投信、壽險、政府基金、軍款等美元買盤逢低承接／出口商於X上方拋匯】，（央行亦進場提供流動性，）使台幣【升勢成強弩之末／貶幅收斂】。
週中【美國數據（CPI等）／FOMC／台積電法說／台指期結算】，【市場反應】，外資【轉向／未轉向】，台幣【動作＋價位】。
週後段【台股走勢】，外資【連續賣超並維持匯出／重新匯入】，加上出口商【拋匯力道轉弱／逢高拋匯】，台幣【跌破/收復】【關口】，一度觸及【　】；（然央行持續提供流動性並引導出口商賣匯，／央行擴大調節力道，）終盤收於【　】（，單日成交量突破【X】億美元）。
（資金動向方面（選用）：本週USDTWD主要由【外資匯出/雙向操作】主導，外資連【X】個交易日賣超台股逾【X】億台幣；實需端方面，【進口商、油款與美元回補買盤】在【價位】上方承接美元，出口商則於台幣回貶時逢高拋匯，使台幣整體於【區間】狹幅盤整。）
經濟基本面方面，【週X】台灣公布【月份】【出口／外銷訂單／工業生產／製造業PMI／失業率】達【數值】，年增【X%】，【創歷年單月第X高／連續X個月擴張】，主要受惠【AI、高效能運算、提前拉貨】需求；【第二句延伸解讀】；惟【在外資匯出及股利需求主導下，對台幣支撐效果有限／短線匯率仍由資金流向主導】。
（壽險避險段（選用）：壽險業避險新制持續影響匯市，【X月】壽險避險比率降至【X%】，續創歷史新低；若壽險業持續降低避險部位，將減少換匯需求，削弱台幣升值動能。）
〔展望段（選用，僅題材明確時啟用）：展望後市，下週將公布台灣【失業率、外銷訂單、工業生產】，並聚焦美國【PMI】；若美國經濟數據轉弱，美元或將續弱並提供台幣反彈空間；然而目前仍值【股利發放旺季】，外資匯出需求短期難以消退，台幣預料仍將偏弱整理，預估下週區間【　】–【　】。〕"""

TPL_JPY = """本週USDJPY【高位震盪／先升後震盪／先升後貶／呈震盪上行】，通週區間落在【　】–【　】。
上週五【若有重大事件：事件＋USDJPY反應＋價位】。
週初受【地緣事件／美日利差／月末季末資金】影響，【推升美元避險需求及聯準會偏鷹預期】，帶動USDJPY自【　】【反彈至/站穩】【　】；惟【日本官方因素：財務大臣發言／GPIF配置／干預風險／財務省釋出干預訊號】，帶動日圓短時【獲撐】（，市場於【關口】下方追價情緒轉趨謹慎）。
週中美國【CPI及PPI／職位空缺／FOMC】【內容＋數值】，市場【降低/提高】FED升息預期，美元【承壓/走強】下推動USDJPY【回落至/上探】【　】；惟週後段【美國初請失業金／零售銷售／非農】【優劣於】預期，（部分FED官員表示【　】，）USDJPY【再度回升測試/快速回落至】【　】。
（整體資金流方面（選用）：週初【　】推升美元需求，週中因【　】引發獲利了結，週後段則因【　】重新吸引買盤，推動USDJPY維持【高檔】區間震盪。）
經濟基本面方面，日本【月份】【CGPI／東京核心CPI／短觀／PMI／貿易收支／機械訂單】【數值】，【高低於】預期（並創【　】以來最大【　】），顯示【企業成本壓力／通膨預期升溫／實體經濟動能改善】。（家庭通膨預期調查：預期一年後物價上漲的家庭比例升至【X%】，創【　】以來最高。）
政策面向上，日央【理事姓名／會議結果】表示【　】，反映日本央行【在物價穩定與經濟影響間評估／內部仍有鷹派聲音】；惟【日本政府因素：財政擴張計畫／要求配合成長導向政策／人事任命】，仍使BOJ後續政策路徑存在不確定性（，並削弱日圓基本面支撐）。
展望後市，市場將聚焦【USDJPY能否突破X關口、日本官方干預力道、美日利差與美國數據】。基本面上，【日本數據】仍支持日本央行【政策正常化路徑】；惟【政府偏鴿基調】可能牽制升息節奏。短線若【美元情境】，USDJPY可能【測試X】；若【日本官方出手干預／美元續弱】，料USDJPY將【向下跌破X】。預估下週USDJPY將維持【　】–【　】區間震盪。"""

TPL_CHF = """本週USD/CHF【交易區間介於【　】–【　】／呈高檔震盪格局／自高檔回落／先升後高檔震盪】。
上週五【若有重大事件：事件＋USDCHF反應＋價位】。
週初市場延續【聯準會偏鷹立場／避險需求】所帶來的支撐，在【地緣政治風險】及【美國貨幣政策預期】間拉鋸下，美元指數持穩於【　】上方（，市場預估聯準會【X月】升息機率一度升至約【X%】），帶動【美債殖利率走高】，USD/CHF於【週X】升抵【　】。
然而，隨著【美國CPI增幅低於預期／停火進展／Fed主席發言】，市場重新評估聯準會後續緊縮路徑（，FedWatch顯示【X月】升息機率由約【X%】下修至【X%】），美債殖利率一度回落，美元漲勢放緩，USD/CHF於週後段【回落至X整數關口上方整理】。（惟美國【零售銷售／初領失業金】優於預期，限制美元回檔幅度。）
經濟數據方面，瑞士基本面【仍展現一定韌性／偏弱】。【生產者及進口物價／CPI／零售銷售／PMI／經常帳／經濟信心指數】【數值＋與預期前值比較】，顯示【企業端價格壓力疲弱／內需具支撐／通膨維持在SNB目標區間內】。（SNB政策段：瑞士央行【維持政策利率0%不變／官員重申保留外匯干預彈性】，【理由】。）（結構因素：瑞士長期維持龐大的經常帳順差及避險貨幣地位，仍為瑞郎基本面提供支撐。）
展望後市，隨著【市場對聯準會升息預期降溫／地緣風險緩解】，美元短線缺乏持續走強的催化劑，預期USD/CHF將【維持震盪偏弱／由單邊升勢轉向震盪偏貶】；然而，【聯準會未明確轉向寬鬆、美瑞利差仍處高檔】，美元短期仍具支撐。結尾兩式擇一：
（單一區間式）綜合評估，預計下週USDCHF交易區間將落在【　】–【　】。
（條件式）短線觀察【X整數關卡】：若【站穩/跌破】【X】，上方可望挑戰【X–X】／下方可測試【X】附近支撐；反之若【　】，則【上方留意X及X壓力／下方留意X心理關卡支撐】。"""

TPL_AUD = """本週AUDUSD【震盪偏弱／震盪走弱／呈現築底反彈／先抑後揚／震盪下行】格局，通週交易區間【　】~【　】（，主因【中國經濟數據／聯準會政策預期／地緣風險／RBA預期】對澳幣形成【壓力/支撐】）。
上週五【若有重大事件：事件＋AUDUSD反應＋價位】。
【週間敘事，段落式（週初/週中/週後段）逐段推進】每段句式：【時間點】，【事件：中國PMI／貿易數據／鐵礦砂銅價／美國數據／RBA決議或會議紀要／地緣】，【解讀】，澳幣【動作＋價位】。中國因素為AUD特有必寫項：中國【數據】【優劣於】預期，【有助修正對中國需求的悲觀預期／反映內需動能疲弱】，（加上【鐵礦砂進口／大連鐵礦砂期貨／銅價】【回升至X關口／續處低檔】，）【激勵/壓抑】AUDUSD【　】。
經濟數據方面，澳洲【月份】【NAB商業景氣與信心／CPI（含trimmed mean）／就業與失業率／PMI／貿易帳／建築許可／家庭支出】【數值＋比較】，顯示【企業經營狀況／通膨黏性／勞動市場／房市】【解讀】。（若處數據空窗期：近期澳洲處於經濟數據空窗期，前月數據顯示【　】。）在【經濟成長具韌性及通膨僵固】下，市場定價RBA維持【相對高利率位階】，利差優勢持續【支撐/削弱】AUDUSD。
展望後市，【美國通膨與聯準會預期】＋【RBA態度】＋【中國需求與鐵礦砂、銅價】三線並陳：若【　】，則【　】；然而【　】。（技術面上（選用）：AUDUSD【站上月線X／月線下穿半年線】，【上方X轉為初步壓力／持續強勢】。）預估AUDUSD【延續X趨勢／維持弱勢震盪】，預估區間【　】~【　】。"""

TPL_CNH = """本週USDCNH呈【震盪下行／區間震盪偏升／狹幅震盪尾盤收高／衝高回吐】的格局，通週交易區間落在【　】~【　】。
上週五【若有重大事件：事件＋USDCNH反應＋價位】。
回顧一週走勢：週初受【中東局勢／美元強弱／季底結匯資金】影響，人民幣【小幅承壓/獲得支撐】（，USDCNH一度【上探/下探】【　】）。週中【美國CPI／PMI／FOMC點陣圖】【內容】，【顯著降低/推升】市場對聯準會升息的預期，成為人民幣本週【單日最大升幅的主要催化劑／承壓主因】，收【　】；週後段【事件：地緣反覆／PCE／非農】，人民幣【回吐漲幅／進一步反彈】，（尾盤）收在【　】附近。
經濟數據方面，中國【內外需呈現兩極化／經濟數據多空交織】。【GDP／社零／貸款與M2／稅收／PMI】【數值＋與預期比較】，顯示【內需疲弱／信用需求疲弱／築底】；相較之下，【出口／工業增加值】【數值】，【創X新高／優於預期】，主因【全球AI投資需求】，顯示【「生產強、內需弱」／「房地產弱、科技與股市資金強」】的結構性分化。
政策面方面，人民銀行【本週中間價設定：維持穩定偏強／連續偏弱／由X逐步降至X】，顯示官方【引導人民幣穩中趨升／轉趨中性／容忍隨美元彈性調整／仍願意維持偏強引導】的態度。（若有利率工具操作：人民銀行於【日期】將【工具】設定在【X%】，【低高於】預期，市場初步反應【　】，惟官方隨後澄清【　】。）
展望後市，外部因素方面，若美國【數據轉弱／官方釋出鴿派訊號】，將引導匯率朝區間【下緣】靠近；國內方面，【降息預期／利差收窄／政治局會議】恐形成【利空／變數】。（在【中間價背書／結匯需求】下，看多情緒依然穩固。）綜合評估，下週USDCNH主要交易區間預估將落在【　】–【　】。"""

TPL_EUR = """本週EURUSD【於【　】–【　】區間震盪／自低位回升／呈先升後貶／震盪偏弱】，通週交易區間落於【　】–【　】。
上週五【若有重大事件：事件＋EURUSD反應＋價位】。
週初受【中東局勢／美伊談判／油價】影響，市場對聯準會升息預期【升降溫】，美元指數【走強/回落】，EURUSD【一度偏弱整理/反彈走升】；（隨後【歐元區利多：德國預算／ZEW／ECB升息預期】，推升【德債殖利率】走揚，支撐EURUSD【站穩/重返】【關卡】。）
週中美國公布【CPI及PPI／非農】【內容】，利率期貨市場將Fed升息預期【延後至X月／下調】，美元【回落】，EURUSD一度觸及【X以來高低點】【　】；（然因歐元區【CPI年增X%低於預期X%】，仍限制歐元上行空間。）
（歐洲數據／ECB方面：歐元區【月份】【製造業PMI／CPI】為【　】，【與前值預期比較】，顯示【　】。ECB【行長/執委姓名】表示【　】，市場【維持/下修】對ECB【年內再升息】預期（，市場目前預估【X月】會議升息1碼的機率約【X%】）。）
展望後市，【下週ECB利率決議／Fed餘波／美伊局勢】將主導走勢。若【情境A】，歐元有望【重新挑戰/回測】【區間】；反之若【情境B】，歐元不排除再度測試【支撐位】。整體而言，在【美國利差優勢／避險需求／歐元區通膨韌性存疑】的背景下，預期下週歐元將於【　】–【　】區間【偏弱/偏貶】震盪。"""

# (標題, 檢索名稱, 小數位數說明, 必查清單, 結尾慣例, 模板)
CURRENCIES = [
    ("新台幣 TWD（本週展望）", "新臺幣兌美元 USDTWD",
     "USDTWD 寫到小數三位",
     "外資買賣超金額、台股與台積電、出口商/進口商/油款/壽險/投信實需、央行調節、出口/外銷訂單/PMI/失業率、股利匯出季、壽險避險比率；美國面：非農、CPI、FOMC",
     "多數週不寫預測區間，以經濟基本面或資金動向段作結；僅題材明確時啟用展望段",
     TPL_TWD),
    ("日圓 JPY（USDJPY）", "美元兌日圓 USDJPY",
     "USDJPY 寫到小數兩位",
     "BOJ決議與官員發言、政府（首相/財務大臣）財政與干預訊號、CGPI/東京CPI/短觀/PMI、投機倉位；美國面：CPI/PPI、非農、初請、FOMC、油價",
     "必寫展望段與預估區間",
     TPL_JPY),
    ("瑞郎 CHF（USDCHF）", "美元兌瑞郎 USDCHF",
     "USDCHF 寫到小數四位",
     "SNB利率（0%）與干預表態、生產者及進口物價/CPI/零售/PMI、經常帳、避險地位、瑞士公投等事件風險；美國面：CPI/PPI、FedWatch機率、零售、初領失業金",
     "單一區間式或條件式區間擇一",
     TPL_CHF),
    ("澳幣 AUD（AUDUSD）", "澳幣兌美元 AUDUSD",
     "AUDUSD 寫到小數四位",
     "RBA決議/紀要/總裁發言、CPI與trimmed mean、就業、NAB、中國數據、鐵礦砂/銅/金價、貿易帳、房市；美國面：CPI、PMI、非農、PCE、FOMC",
     "必寫展望段與預估區間（偶附技術面）",
     TPL_AUD),
    ("離岸人民幣 CNH（USDCNH）", "美元兌離岸人民幣 USDCNH",
     "USDCNH 寫到小數四位",
     "中間價設定與官方態度、GDP/社零/貸款M2/稅收、出口、PMI、人行利率工具、政治局會議；美國面：CPI、FOMC、非農、地緣",
     "必寫展望段與預估區間",
     TPL_CNH),
    ("歐元 EUR（EURUSD）", "歐元兌美元 EURUSD",
     "EURUSD 寫到小數四位",
     "ECB決議與Lagarde/執委發言、HICP/PMI/ZEW、德債殖利率、德國財政；美國面：CPI/PPI、非農、FOMC、地緣油價",
     "必寫展望段與預估區間",
     TPL_EUR),
]

# ==========================================
# 4. 依序生成：三段 + 六幣別（共 7 次呼叫）
# ==========================================
results = {}


def failed_result(exc):
    return ModelResult(
        text=f"（生成失敗：{exc}）",
        model="",
        sources=[],
        queries=[],
        error=str(exc),
    )


try:
    print("\n🤖 生成 Spot / Swap / Forward（含來源表，模型將自行上網檢索）...")
    try:
        results["SSF"] = call_model(PROMPT_SSF, label="Spot/Swap/Forward")
        print(f"   ✅ 完成（{len(results['SSF'].text)} 字）")
    except Exception as exc:
        results["SSF"] = failed_result(exc)
        print(f"   ❌ 失敗：{exc}")
    time.sleep(4)

    for title, ccy_name, decimals, checklist, ending, tpl in CURRENCIES:
        print(f"🤖 生成【{title}】...")
        prompt = CCY_WRAPPER.format(
            today=TODAY_STR,
            trade_period=TRADE_PERIOD_STR,
            news_period=NEWS_PERIOD_STR,
            ccy_name=ccy_name,
            decimals=decimals,
            checklist=checklist,
            ending=ending,
            template=tpl,
        )
        try:
            results[title] = call_model(prompt, label=title)
            print(f"   ✅ 完成（{len(results[title].text)} 字）")
        except Exception as exc:
            results[title] = failed_result(exc)
            print(f"   ❌ 失敗：{exc}")
        time.sleep(4)
finally:
    # 明確關閉同步 Client，釋放底層 HTTP 連線。
    try:
        client.close()
    except Exception:
        pass

# ==========================================
# 5. 輸出 HTML（七張卡片，共九個內容區塊）與 grounding 稽核 JSON
# ==========================================
def to_html(text):
    """先跳脫 HTML，再將成對的 **粗體** 轉為 <strong>。"""
    escaped = html_lib.escape(str(text), quote=True)
    parts = escaped.split("**")
    rendered = []

    for i, segment in enumerate(parts):
        if i % 2 == 1:
            rendered.append(f"<strong>{segment}</strong>")
        else:
            rendered.append(segment)

    return "<br>".join("".join(rendered).splitlines())


def safe_http_url(url):
    """HTML 連結僅允許 http／https，避免非預期協定。"""
    parsed = urlparse(str(url))
    return str(url) if parsed.scheme in {"http", "https"} else ""


def grounding_to_html(result):
    """將 API 實際回傳的搜尋查詢與 grounding 來源附在每張卡片下方。"""
    items = []

    for source in result.sources:
        title = html_lib.escape(source.get("title", "未命名來源"), quote=True)
        items.append(f"<li>{title}</li>")  # 依需求僅列標題，網址移除，供使用者自行搜尋

    if items:
        source_html = "<ol>" + "".join(items) + "</ol>"
    else:
        source_html = '<p class="warning">未取得 API grounding 來源，此區塊的數字複核需全部自行搜尋查證。</p>'

    if result.queries:
        query_text = "；".join(result.queries)
        query_html = (
            '<p class="queries"><strong>實際搜尋查詢：</strong>'
            + html_lib.escape(query_text, quote=True)
            + "</p>"
        )
    else:
        query_html = '<p class="queries"><strong>實際搜尋查詢：</strong>未取得</p>'

    return (
        '<details class="grounding"><summary>API 實際檢索來源（僅標題，請自行搜尋複核）</summary>'
        + query_html
        + source_html
        + "</details>"
    )


html_output = """<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FX Flow Weekly</title>
<style>
body{font-family:-apple-system,"Segoe UI",Roboto,"PingFang TC","Microsoft JhengHei",sans-serif;max-width:900px;margin:0 auto;padding:30px 20px;background:#f4f6f9;color:#1e293b;line-height:1.9}
.header{background:linear-gradient(135deg,#0f172a,#1e293b);color:#fff;padding:26px;border-radius:12px;margin-bottom:22px}
.header h1{margin:0;font-size:24px}.header .date{font-size:13px;color:#94a3b8;margin-top:6px}
.nav{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:22px;background:#fff;padding:14px;border-radius:10px}
.nav a{color:#2563eb;text-decoration:none;font-size:13px;padding:5px 12px;background:#eff6ff;border-radius:20px}
.card{background:#fff;border-radius:12px;padding:26px;margin-bottom:22px;border-top:4px solid #2563eb;box-shadow:0 2px 10px rgba(0,0,0,.04)}
.card h2{margin-top:0;font-size:20px;color:#0f172a;border-bottom:1px solid #f1f5f9;padding-bottom:8px}
.meta{font-size:12px;color:#64748b;margin:-2px 0 14px}
.body{font-size:15px;color:#334155;text-align:justify;overflow-wrap:anywhere}
.grounding{margin-top:22px;padding:12px 14px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;font-size:13px}
.grounding summary{cursor:pointer;font-weight:700;color:#334155}
.grounding ol{margin:10px 0 0;padding-left:22px}.grounding li{margin-bottom:5px;overflow-wrap:anywhere}
.grounding a{color:#2563eb}.queries{margin:10px 0}.warning{color:#b45309;margin:10px 0 0}
.note{background:#fff;border-radius:12px;padding:18px;border-top:4px solid #64748b;font-size:12px;color:#64748b}
</style></head><body>
<div class="header"><h1>📈 FX Flow Weekly</h1>
<div class="date">參考期間：""" + html_lib.escape(PERIOD_STR) + """｜產出時間：""" + html_lib.escape(NOW_STR) + """（Asia/Taipei）</div></div>
"""

titles = ["Spot / Swap / Forward（USDTWD，含資料來源表）"] + [t for t, *_ in CURRENCIES]
keys = ["SSF"] + [t for t, *_ in CURRENCIES]

html_output += '<div class="nav">'
for i, title in enumerate(titles):
    short_title = title.split("（")[0]
    html_output += (
        f'<a href="#s{i}">' + html_lib.escape(short_title, quote=True) + "</a>"
    )
html_output += "</div>"

for i, (title, key) in enumerate(zip(titles, keys)):
    result = results.get(key) or ModelResult("（無內容）", "", [], [], "無內容")
    model_text = result.model or "未完成"

    html_output += (
        f'<div class="card" id="s{i}">'
        + "<h2>📌 "
        + html_lib.escape(title, quote=True)
        + "</h2>"
        + '<div class="meta">實際使用模型：'
        + html_lib.escape(model_text, quote=True)
        + f"｜Grounding 來源數：{len(result.sources)}</div>"
        + '<div class="body">'
        + to_html(result.text)
        + "</div>"
        + grounding_to_html(result)
        + "</div>"
    )

html_output += (
    '<div class="note">本報告由 AI 檢索網路公開資料生成，發布前務必：'
    '1) 逐項核對各段「資料來源」表中的數值（正文來源表僅含來源名稱，不含網址）；'
    '2) 各卡片「API 實際檢索來源」僅列標題（網址依需求移除）；以標題自行搜尋原文複核，完整轉址連結仍保留於 grounding JSON 檔（約 30 天後失效）；'
    '3) 補寫【待填】之行內資料（交易量、實需面、NDF）；'
    '4) 點位以行內／Bloomberg 數據為準覆核。標示【未查得】【尚未公布】處請自行補齊或刪除。'
    "</div>"
)
html_output += "</body></html>"

with open("index.html", "w", encoding="utf-8") as file:
    file.write(html_output)

# 另存 API 實際回傳的 grounding 資料，方便逐項稽核與後續程式處理。
grounding_audit = {
    key: {
        "title": title,
        "model": results[key].model,
        "error": results[key].error,
        "web_search_queries": results[key].queries,
        "grounding_sources": results[key].sources,
    }
    for title, key in zip(titles, keys)
}

with open("fx_flow_weekly_grounding.json", "w", encoding="utf-8") as file:
    json.dump(grounding_audit, file, ensure_ascii=False, indent=2)

failed = [key for key in keys if results.get(key) and results[key].error]
print("\n========== 執行結果 ==========")
if failed:
    print(f"❌ 以下區塊生成失敗：{failed}")
    print("✅ index.html 與 fx_flow_weekly_grounding.json 已產生，請注意上述失敗區塊。")
    # 有區塊失敗時仍發布（頁面該卡片會顯示「生成失敗」）。
    # 若希望「任一失敗就不發布」，把下一行取消註解：
    # raise SystemExit(1)
else:
    print("🎉 完成！七張報告卡片、共九個內容區塊已寫入 index.html。")
    print("🔎 API grounding 稽核資料已寫入 fx_flow_weekly_grounding.json。")
    print("發布前請逐項複核來源表並補寫【待填】欄位。")
