import os
import sys
import time
import logging
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd
import requests


# ============================================================
# 用户配置
# ============================================================

# 连续跌停至少多少个交易日
MIN_CONSECUTIVE_DT = 3

# 最多向前检查多少个交易日
# 例如 20 表示最多识别：
# 3连跌、4连跌、5连跌……20连跌
LOOKBACK_TRADE_DAYS = 10

# 上市不足多少个交易日的新股排除
MIN_LISTING_TRADE_DAYS = 60

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# 每次 AKShare 请求之间稍微等待一下
REQUEST_SLEEP = 0.8

# AKShare 网络请求失败时的重试次数
API_RETRY_TIMES = 3

# 每次重试之间等待秒数
API_RETRY_SLEEP = 3

# Telegram 请求超时
TELEGRAM_TIMEOUT = 20


# ============================================================
# 日志
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("limit_down_monitor")


# ============================================================
# Telegram
# ============================================================

def send_telegram_message(text: str):
    """
    Telegram Bot 推送。
    """

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "环境变量 TELEGRAM_BOT_TOKEN 未设置"
        )

    if not TELEGRAM_CHAT_ID:
        raise RuntimeError(
            "环境变量 TELEGRAM_CHAT_ID 未设置"
        )

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    response = requests.post(
        url,
        json=payload,
        timeout=TELEGRAM_TIMEOUT,
    )

    response.raise_for_status()

    result = response.json()

    if not result.get("ok"):
        raise RuntimeError(
            f"Telegram API 返回失败：{result}"
        )


# ============================================================
# 交易日历
# ============================================================

def get_trade_calendar():
    """
    获取完整交易日历。

    返回：
        DatetimeIndex
    """

    logger.info("获取交易日历")

    df = ak.tool_trade_date_hist_sina()

    if df is None or df.empty:
        raise RuntimeError(
            "获取交易日历失败"
        )

    # AKShare 常见字段：trade_date
    possible_columns = [
        "trade_date",
        "交易日期",
    ]

    date_col = None

    for col in possible_columns:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        # 最后尝试第一列
        date_col = df.columns[0]

    dates = pd.to_datetime(
        df[date_col],
        errors="coerce",
    ).dropna()

    return pd.DatetimeIndex(
        sorted(dates.unique())
    )


def get_recent_trade_dates(
    count: int,
):
    """
    获取最近 count 个交易日。

    注意：
    如果今天是交易日，则包含今天。
    如果今天不是交易日，则自动取最近一个交易日。
    """

    calendar = get_trade_calendar()

    today = pd.Timestamp(
        datetime.now().date()
    )

    dates = calendar[
        calendar <= today
    ]

    if len(dates) < count:
        raise RuntimeError(
            f"交易日历只有 {len(dates)} 个交易日，"
            f"不足 {count} 个"
        )

    result = dates[-count:]

    return [
        d.strftime("%Y%m%d")
        for d in result
    ]


# ============================================================
# 主板判断
# ============================================================

def is_main_board(code: str) -> bool:
    """
    只保留沪深主板。

    保留：

    上海主板：
        600xxx
        601xxx
        603xxx
        605xxx

    深圳主板：
        000xxx
        001xxx
        002xxx
        003xxx

    排除：

        300xxx    创业板
        301xxx    创业板
        688xxx    科创板
        689xxx    科创板
        4xxxxx    北交所
        8xxxxx    北交所
        43xxxxx   北交所历史代码等

    """

    code = str(code).strip()

    if len(code) != 6:
        return False

    return code.startswith(
        (
            "000",
            "001",
            "002",
            "003",
            "600",
            "601",
            "603",
            "605",
        )
    )


# ============================================================
# 获取股票基本信息
# ============================================================

def get_stock_basic_info_from_history(history):
    """
    从跌停池历史数据中取得股票代码和名称。

    不再调用 stock_zh_a_spot_em()。
    这样可以避免 GitHub Actions 中访问东方财富实时行情
    接口时出现 RemoteDisconnected。
    """
    rows = []

    for item in history.values():
        if item is None or item.empty:
            continue

        rows.append(item[["code", "name"]].copy())

    if not rows:
        return pd.DataFrame(columns=["code", "name"])

    result = pd.concat(rows, ignore_index=True)

    result["code"] = result["code"].astype(str).str.strip()
    result["name"] = result["name"].astype(str).str.strip()

    # 只保留沪深主板
    result = result[
        result["code"].apply(is_main_board)
    ].copy()

    # 同一股票可能在多个交易日出现，只保留一条
    result = result.drop_duplicates(
        subset=["code"],
        keep="last",
    )

    return result.reset_index(drop=True)


# ============================================================
# 跌停池接口兼容
# ============================================================

def get_limit_down_pool(
    trade_date: str,
):
    """
    获取某个交易日的跌停池。

    兼容：
        stock_zt_pool_dtgc
        stock_zt_pool_dtgc_em

    网络异常会自动重试，避免一次 RemoteDisconnected
    直接导致 GitHub Actions 失败。
    """

    funcs = []

    # 用户原环境中的接口优先
    if hasattr(ak, "stock_zt_pool_dtgc"):
        funcs.append(
            (
                "stock_zt_pool_dtgc",
                getattr(ak, "stock_zt_pool_dtgc"),
            )
        )

    # 新版 AKShare 接口
    if hasattr(ak, "stock_zt_pool_dtgc_em"):
        funcs.append(
            (
                "stock_zt_pool_dtgc_em",
                getattr(ak, "stock_zt_pool_dtgc_em"),
            )
        )

    if not funcs:
        raise RuntimeError(
            "当前 AKShare 没有找到："
            "stock_zt_pool_dtgc / "
            "stock_zt_pool_dtgc_em"
        )

    last_error = None

    for name, func in funcs:
        for attempt in range(1, API_RETRY_TIMES + 1):
            try:
                logger.info(
                    "调用 %s，日期=%s，第 %d/%d 次",
                    name,
                    trade_date,
                    attempt,
                    API_RETRY_TIMES,
                )

                df = func(date=trade_date)

                if df is None:
                    return pd.DataFrame()

                return df

            except Exception as e:
                last_error = e

                logger.warning(
                    "%s 调用失败（日期=%s，第 %d/%d 次）：%s",
                    name,
                    trade_date,
                    attempt,
                    API_RETRY_TIMES,
                    e,
                )

                if attempt < API_RETRY_TIMES:
                    time.sleep(API_RETRY_SLEEP)

        # 当前接口连续失败后，再尝试另一个兼容接口
        time.sleep(REQUEST_SLEEP)

    raise RuntimeError(
        f"所有跌停池接口均调用失败：{last_error}"
    )


# ============================================================
# 规范化跌停池
# ============================================================

def normalize_limit_down_pool(
    df: pd.DataFrame,
):
    """
    将跌停池统一成：

        code
        name
    """

    if df is None or df.empty:

        return pd.DataFrame(
            columns=[
                "code",
                "name",
            ]
        )

    code_columns = [
        "代码",
        "股票代码",
        "证券代码",
        "code",
    ]

    name_columns = [
        "名称",
        "股票名称",
        "证券名称",
        "name",
    ]

    code_col = None
    name_col = None

    for col in code_columns:

        if col in df.columns:
            code_col = col
            break

    for col in name_columns:

        if col in df.columns:
            name_col = col
            break

    if code_col is None:

        raise RuntimeError(
            "无法识别跌停池股票代码字段。\n"
            f"实际字段：{df.columns.tolist()}"
        )

    result = pd.DataFrame()

    result["code"] = (
        df[code_col]
        .astype(str)
        .str.extract(r"(\d{6})")[0]
    )

    if name_col is not None:

        result["name"] = (
            df[name_col]
            .astype(str)
            .str.strip()
        )

    else:

        result["name"] = ""

    result = result.dropna(
        subset=["code"]
    )

    result = result.drop_duplicates(
        subset=["code"]
    )

    return result.reset_index(
        drop=True
    )


# ============================================================
# 一次性获取最近 N 个交易日跌停池
# ============================================================

def collect_limit_down_history(
    trade_dates,
):
    """
    抓取最近 N 个交易日的跌停池。

    返回：
        history: {交易日: 股票代码集合}
        stock_info: 最近 N 个交易日跌停池中出现过的
                    股票代码和名称

    不再请求全市场实时股票列表。
    """
    history = {}
    stock_info_parts = []

    total = len(trade_dates)

    for index, trade_date in enumerate(
        trade_dates,
        start=1,
    ):
        logger.info(
            "[%d/%d] 获取跌停池：%s",
            index,
            total,
            trade_date,
        )

        raw = get_limit_down_pool(trade_date)

        normalized = normalize_limit_down_pool(raw)

        if not normalized.empty:
            stock_info_parts.append(
                normalized[["code", "name"]].copy()
            )

        codes = set(
            normalized["code"].tolist()
        )

        history[trade_date] = codes

        logger.info(
            "%s 跌停股票：%d",
            trade_date,
            len(codes),
        )

        time.sleep(REQUEST_SLEEP)

    if stock_info_parts:
        stock_info = pd.concat(
            stock_info_parts,
            ignore_index=True,
        )
        stock_info = stock_info.drop_duplicates(
            subset=["code"],
            keep="last",
        ).reset_index(drop=True)
        stock_info = stock_info[
            stock_info["code"].apply(is_main_board)
        ].copy()
    else:
        stock_info = pd.DataFrame(
            columns=["code", "name"]
        )

    return history, stock_info


# ============================================================
# 计算连续跌停
# ============================================================

def calculate_consecutive_days(
    history,
    trade_dates,
):
    """
    从最近交易日开始，计算每只股票连续跌停天数。

    例如：

        09-08 跌停
        09-07 跌停
        09-04 跌停
        09-03 跌停
        09-02 未跌停

    结果：

        连续跌停 = 4

    完全在本地计算。
    """

    if not trade_dates:
        return {}

    latest_date = trade_dates[-1]

    latest_codes = history.get(
        latest_date,
        set(),
    )

    if not latest_codes:
        return {}

    logger.info(
        "最近交易日跌停股票：%d",
        len(latest_codes),
    )

    result = {}

    # 只从最近交易日的跌停股开始检查
    for code in latest_codes:

        count = 0

        # 从后往前逐交易日检查
        for trade_date in reversed(
            trade_dates
        ):

            codes = history.get(
                trade_date,
                set(),
            )

            if code in codes:
                count += 1
            else:
                break

        if count >= MIN_CONSECUTIVE_DT:

            result[code] = count

    return result


# ============================================================
# 新股过滤
# ============================================================

def get_listing_date(code: str):
    """
    查询单只股票上市日期。

    注意：

    这里只对已经满足“连续跌停 >= N”
    的极少数股票查询。

    不会对全部股票查询。
    """

    try:

        df = ak.stock_individual_info_em(
            symbol=code
        )

        if df is None or df.empty:
            return None

        # 常见字段：
        # item / value
        # 项目 / 值

        item_col = None
        value_col = None

        for col in df.columns:

            if str(col) in (
                "item",
                "项目",
            ):
                item_col = col

            if str(col) in (
                "value",
                "值",
            ):
                value_col = col

        if (
            item_col is None
            or value_col is None
        ):
            return None

        for _, row in df.iterrows():

            item = str(
                row[item_col]
            )

            if "上市时间" not in item:
                continue

            value = row[value_col]

            dt = pd.to_datetime(
                value,
                errors="coerce",
            )

            if pd.notna(dt):

                return dt.normalize()

    except Exception as e:

        logger.warning(
            "获取 %s 上市时间失败：%s",
            code,
            e,
        )

    return None


def count_listing_trade_days(
    listing_date,
    latest_date,
    calendar,
):
    """
    计算上市日至最近交易日之间的交易日数量。
    """

    if listing_date is None:
        return None

    listing_date = pd.Timestamp(
        listing_date
    )

    latest_date = pd.Timestamp(
        latest_date
    )

    dates = calendar[
        (calendar >= listing_date)
        & (calendar <= latest_date)
    ]

    return len(dates)


def filter_new_stocks(
    result,
    latest_trade_date,
    calendar,
):
    """
    排除上市交易日不足 N 天的新股。

    只对已经进入候选名单的股票查询上市日期，
    不会遍历全部 A 股。
    """

    if result.empty:
        return result

    latest_date = pd.to_datetime(
        latest_trade_date
    )

    kept_rows = []

    for _, row in result.iterrows():

        code = row["code"]
        name = row["name"]

        listing_date = get_listing_date(
            code
        )

        # 无法取得上市日期：
        # 为避免误删，保留
        if listing_date is None:

            logger.warning(
                "%s %s 无法取得上市日期，暂不排除",
                code,
                name,
            )

            kept_rows.append(row)
            continue

        trade_days = count_listing_trade_days(
            listing_date,
            latest_date,
            calendar,
        )

        if (
            trade_days is not None
            and trade_days
            >= MIN_LISTING_TRADE_DAYS
        ):

            kept_rows.append(row)

        else:

            logger.info(
                "排除新股：%s %s，上市交易日=%s",
                code,
                name,
                trade_days,
            )

        time.sleep(
            REQUEST_SLEEP
        )

    if not kept_rows:

        return pd.DataFrame(
            columns=result.columns
        )

    return pd.DataFrame(
        kept_rows
    ).reset_index(
        drop=True
    )


# ============================================================
# 构建结果
# ============================================================

def build_result(
    consecutive,
    stock_info,
):
    """
    将：

        code -> 连续跌停天数

    转换成 DataFrame。
    """

    if not consecutive:
        return pd.DataFrame(
            columns=[
                "code",
                "name",
                "consecutive_days",
            ]
        )

    result = pd.DataFrame(
        [
            {
                "code": code,
                "consecutive_days": days,
            }
            for code, days
            in consecutive.items()
        ]
    )

    result = result.merge(
        stock_info,
        on="code",
        how="left",
    )

    result["name"] = (
        result["name"]
        .fillna("")
    )

    return result[
        [
            "code",
            "name",
            "consecutive_days",
        ]
    ]


# ============================================================
# Telegram 消息
# ============================================================

def build_telegram_message(
    result,
    latest_trade_date,
):
    """
    生成 Telegram HTML。
    """

    date_fmt = pd.to_datetime(
        latest_trade_date
    ).strftime("%Y-%m-%d")

    title = (
        "📉 <b>连续跌停监控</b>"
    )

    condition = (
        f"连续跌停 ≥ "
        f"{MIN_CONSECUTIVE_DT} 个交易日"
    )

    if result.empty:

        return (
            f"{title}\n\n"
            f"交易日：{date_fmt}\n"
            f"条件：{condition}\n\n"
            f"✅ 没有符合条件的股票。"
        )

    lines = [
        title,
        "",
        f"交易日：{date_fmt}",
        f"条件：{condition}",
        "",
    ]

    for _, row in result.iterrows():

        code = row["code"]
        name = row["name"]

        days = int(
            row["consecutive_days"]
        )

        lines.append(
            f"🔻 <b>{name}</b> "
            f"<code>{code}</code> "
            f"连续跌停 <b>{days}</b> 天"
        )

    lines.extend(
        [
            "",
            f"共 <b>{len(result)}</b> 只",
            "",
            "排除：创业板、科创板、北交所、"
            f"上市不足 {MIN_LISTING_TRADE_DAYS} 个交易日的新股。",
        ]
    )

    return "\n".join(lines)


# ============================================================
# 主程序
# ============================================================

def main():

    start_time = time.time()

    logger.info("=" * 70)
    logger.info(
        "连续跌停监控启动"
    )
    logger.info("=" * 70)

    # --------------------------------------------------------
    # 1. 检查 Telegram
    # --------------------------------------------------------

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "未设置 TELEGRAM_BOT_TOKEN"
        )

    if not TELEGRAM_CHAT_ID:
        raise RuntimeError(
            "未设置 TELEGRAM_CHAT_ID"
        )

    # --------------------------------------------------------
    # 2. 获取最近交易日
    # --------------------------------------------------------

    trade_dates = get_recent_trade_dates(
        LOOKBACK_TRADE_DAYS
    )

    latest_trade_date = trade_dates[-1]

    logger.info(
        "最近交易日：%s",
        latest_trade_date,
    )

    logger.info(
        "检查交易日：%s ~ %s",
        trade_dates[0],
        trade_dates[-1],
    )

    # --------------------------------------------------------
    # 3. 获取交易日历
    # --------------------------------------------------------

    calendar = get_trade_calendar()

    # --------------------------------------------------------
    # 4. 获取最近20个交易日跌停池
    #
    # 核心：
    # 不再调用 stock_zh_a_spot_em() 获取全市场实时行情。
    # 直接从跌停池取得代码和名称。
    # --------------------------------------------------------

    history, stock_info = collect_limit_down_history(
        trade_dates
    )

    logger.info(
        "最近 %d 个交易日跌停池中出现的沪深主板股票：%d",
        len(trade_dates),
        len(stock_info),
    )

    # --------------------------------------------------------
    # 5. 本地计算连续跌停
    # --------------------------------------------------------

    consecutive = calculate_consecutive_days(
        history,
        trade_dates,
    )

    logger.info(
        "连续跌停 >= %d 天股票：%d",
        MIN_CONSECUTIVE_DT,
        len(consecutive),
    )

    # --------------------------------------------------------
    # 7. 构造结果
    # --------------------------------------------------------

    result = build_result(
        consecutive,
        stock_info,
    )

    # --------------------------------------------------------
    # 8. 主板过滤
    #
    # 实际上股票列表已经过滤过一次，
    # 这里再次保险过滤
    # --------------------------------------------------------

    if not result.empty:

        result = result[
            result["code"].apply(
                is_main_board
            )
        ].copy()

    # --------------------------------------------------------
    # 9. 排除上市不足 N 个交易日的新股
    #
    # 只检查候选股票，不检查全部股票
    # --------------------------------------------------------

    result = filter_new_stocks(
        result,
        latest_trade_date,
        calendar,
    )

    # --------------------------------------------------------
    # 10. 排序
    # --------------------------------------------------------

    if not result.empty:

        result = result.sort_values(
            by=[
                "consecutive_days",
                "code",
            ],
            ascending=[
                False,
                True,
            ],
        ).reset_index(
            drop=True
        )

    # --------------------------------------------------------
    # 11. Telegram
    # --------------------------------------------------------

    message = build_telegram_message(
        result,
        latest_trade_date,
    )

    send_telegram_message(
        message
    )

    # --------------------------------------------------------
    # 12. 完成
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    logger.info(
        "Telegram 推送完成"
    )

    logger.info(
        "最终符合条件：%d 只",
        len(result),
    )

    logger.info(
        "总耗时：%.1f 秒",
        elapsed,
    )

    logger.info("=" * 70)
    logger.info(
        "连续跌停监控结束"
    )
    logger.info("=" * 70)


# ============================================================
# 异常处理
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        logger.exception(
            "程序运行失败：%s",
            e,
        )

        # 程序失败也尝试 Telegram 报警
        try:

            if (
                TELEGRAM_BOT_TOKEN
                and TELEGRAM_CHAT_ID
            ):

                send_telegram_message(
                    "❌ <b>连续跌停监控程序运行失败</b>\n\n"
                    f"<code>{str(e)}</code>"
                )

        except Exception:
            pass

        sys.exit(1)