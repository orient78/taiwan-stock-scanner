# stock_pool.py
#
# Taiwan Stock Scanner V3
# Focus universe:
#   1. Semiconductor
#   2. AI / AI Server
#   3. Technology
#   4. ETF
#
# 同一檔股票可以屬於多個分類。
# scanner.py 後續會自動去除重複代碼。


STOCK_GROUPS = {

    # ==========================================
    # Semiconductor
    # ==========================================

    "半導體": [

        # Foundry / IDM
        "2330",  # 台積電
        "2303",  # 聯電
        "6770",  # 力積電

        # IC Design
        "2454",  # 聯發科
        "3034",  # 聯詠
        "2379",  # 瑞昱
        "3661",  # 世芯-KY
        "3443",  # 創意
        "5274",  # 信驊
        "3529",  # 力旺
        "6531",  # 愛普
        "6415",  # 矽力-KY
        "4961",  # 天鈺
        "8016",  # 矽創
        "4966",  # 譜瑞-KY

        # Memory
        "2408",  # 南亞科
        "2344",  # 華邦電
        "2337",  # 旺宏
        "8299",  # 群聯

        # Packaging / Testing
        "3711",  # 日月光投控
        "6239",  # 力成
        "2449",  # 京元電子
        "8150",  # 南茂
        "6147",  # 頎邦

        # Semiconductor Equipment
        "3131",  # 弘塑
        "3583",  # 辛耘
        "6187",  # 萬潤
        "6223",  # 旺矽
        "6640",  # 均華
        "6515",  # 穎崴

        # Silicon / Materials
        "6488",  # 環球晶
        "3532",  # 台勝科
        "6182",  # 合晶

        # IP / ASIC
        "6643",  # M31
        "6533",  # 晶心科
    ],


    # ==========================================
    # AI / AI Server
    # ==========================================

    "AI": [

        # Server ODM
        "2382",  # 廣達
        "3231",  # 緯創
        "6669",  # 緯穎
        "2356",  # 英業達
        "2317",  # 鴻海
        "2376",  # 技嘉
        "2357",  # 華碩

        # AI ASIC / IC
        "2330",  # 台積電
        "2454",  # 聯發科
        "3661",  # 世芯-KY
        "3443",  # 創意

        # Cooling
        "3017",  # 奇鋐
        "3324",  # 雙鴻
        "3653",  # 健策
        "6230",  # 尼得科超眾

        # Power
        "2308",  # 台達電
        "6412",  # 群電
        "6409",  # 旭隼

        # PCB / CCL
        "2383",  # 台光電
        "6213",  # 聯茂
        "6274",  # 台燿
        "3037",  # 欣興
        "8046",  # 南電
        "2368",  # 金像電

        # Network / High Speed
        "2345",  # 智邦
        "3081",  # 聯亞
        "3363",  # 上詮
        "4979",  # 華星光
        "3163",  # 波若威

        # Connector / Cable
        "3533",  # 嘉澤
        "3665",  # 貿聯-KY
    ],


    # ==========================================
    # Technology
    # ==========================================

    "科技": [

        # Electronics
        "2317",  # 鴻海
        "2308",  # 台達電
        "2382",  # 廣達
        "3231",  # 緯創
        "6669",  # 緯穎
        "2357",  # 華碩
        "2376",  # 技嘉
        "2356",  # 英業達

        # Apple / Consumer Electronics
        "3008",  # 大立光
        "4938",  # 和碩
        "2474",  # 可成
        "2392",  # 正崴

        # Optical / Network
        "2345",  # 智邦
        "6285",  # 啟碁
        "5388",  # 中磊
        "3596",  # 智易

        # PCB
        "3037",  # 欣興
        "8046",  # 南電
        "2368",  # 金像電
        "2383",  # 台光電
        "6274",  # 台燿

        # Cooling / Components
        "3017",  # 奇鋐
        "3324",  # 雙鴻
        "3653",  # 健策
        "3533",  # 嘉澤
    ],


    # ==========================================
    # ETF
    # ==========================================

    "ETF": [

        # Taiwan broad market
        "0050",    # 元大台灣50
        "006208",  # 富邦台50

        # Technology / Semiconductor
        "0052",    # 富邦科技
        "0053",    # 元大電子
        "00881",   # 國泰台灣科技龍頭
        "00891",   # 中信關鍵半導體
        "00927",   # 群益半導體收益
        "00935",   # 野村臺灣新科技50

        # Dividend / broad ETF
        "0056",    # 元大高股息
        "00878",   # 國泰永續高股息
        "00919",   # 群益台灣精選高息
    ],
}


def get_stock_codes():
    """
    Return unique stock codes.
    """

    codes = set()

    for stocks in STOCK_GROUPS.values():
        codes.update(stocks)

    return sorted(codes)


def get_groups(code):
    """
    Return all groups for a stock.
    Example:
        2330 -> ["半導體", "AI"]
    """

    groups = []

    for group, stocks in STOCK_GROUPS.items():

        if code in stocks:
            groups.append(group)

    return groups


if __name__ == "__main__":

    codes = get_stock_codes()

    print(
        "Total unique symbols:",
        len(codes)
    )

    for group, stocks in STOCK_GROUPS.items():

        print(
            group,
            len(stocks)
        )
