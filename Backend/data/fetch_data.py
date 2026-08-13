import yfinance as yf
import pandas as pd

INDIA_VIX_TICKER = "^INDIAVIX"


def _extract_price_data(raw):
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        price_level = raw.columns.get_level_values(0)
        if "Adj Close" in price_level:
            data = raw["Adj Close"]
        elif "Close" in price_level:
            data = raw["Close"]
        else:
            return pd.DataFrame()
    else:
        if "Adj Close" in raw.columns:
            data = raw[["Adj Close"]].copy()
        elif "Close" in raw.columns:
            data = raw[["Close"]].copy()
        else:
            return pd.DataFrame()

    data = data.copy()
    data = data.dropna(axis=1, how="all")
    if data.empty:
        return pd.DataFrame()

    return data.dropna(how="any")


def fetch_data(stocks, period="2y", interval="1d"):
    requested = list(stocks) if isinstance(stocks, (list, tuple, set, pd.Index)) else [stocks]
    requested = [str(s) for s in requested]

    raw = yf.download(
        requested,
        period=period,
        interval=interval
    )
    price_data = _extract_price_data(raw)

    if price_data.empty:
        failed_tickers = requested
        for ticker in failed_tickers:
            print(f"Warning: market data unavailable for {ticker}")
        return pd.DataFrame(), [], failed_tickers

    if len(requested) == 1 and len(price_data.columns) == 1 and requested[0] not in price_data.columns:
        price_data = price_data.rename(columns={price_data.columns[0]: requested[0]})

    valid_tickers = []
    for ticker in requested:
        if ticker in price_data.columns and not price_data[ticker].dropna().empty:
            valid_tickers.append(ticker)
        else:
            print(f"Warning: market data unavailable for {ticker}")

    failed_tickers = [ticker for ticker in requested if ticker not in valid_tickers]

    if not valid_tickers:
        return pd.DataFrame(), [], failed_tickers

    filtered_data = price_data[valid_tickers].copy()
    filtered_data = filtered_data.dropna(how="any")

    if filtered_data.empty:
        for ticker in valid_tickers:
            print(f"Warning: market data unavailable for {ticker}")
        failed_tickers = requested
        return pd.DataFrame(), [], failed_tickers

    return filtered_data, valid_tickers, failed_tickers


def fetch_benchmark_data(ticker="^NSEI"):
    raw = yf.download(
        ticker,
        period="2y",
        interval="1d"
    )
    data = _extract_price_data(raw)
    if data.empty:
        return pd.Series(dtype=float)

    series = data.iloc[:, 0].copy()
    series.name = "benchmark"
    return series


def fetch_india_vix(period="2y", interval="1d"):
    """
    India VIX measures the market's expectation of near-term volatility,
    derived from NIFTY option prices. It's a leading, forward-looking risk
    signal (unlike historical volatility, which is backward-looking) --
    used in the recommendation engine (Phase 8) and to auto-tighten the
    optimizer's strategy when the market is signalling turbulence ahead,
    not just reacting to volatility that already happened.

    Returns a single pd.Series of closing VIX values, name="india_vix".
    Empty Series if the ticker is unavailable for some reason.
    """
    raw = yf.download(
        INDIA_VIX_TICKER,
        period=period,
        interval=interval
    )
    data = _extract_price_data(raw)
    if data.empty:
        print(f"Warning: India VIX data unavailable ({INDIA_VIX_TICKER})")
        return pd.Series(dtype=float)

    series = data.iloc[:, 0].copy()
    series.name = "india_vix"
    return series


def fetch_ohlcv_data(stocks, period="1y", interval="1d"):
    """
    Full OHLCV (Open/High/Low/Close/Volume) per stock, as a dict of
    {ticker: DataFrame}. fetch_data() above only keeps Close/Adj Close,
    which is enough for returns and lagging indicators (RSI, SMA) but NOT
    enough for the leading indicators added in Phase 8 -- Stochastic
    Oscillator and Williams %R need High/Low, and On-Balance Volume needs
    Volume. This function is the source feed for
    signals/leading_indicators.py.

    Returns: (ohlcv_by_ticker: dict[str, pd.DataFrame], valid_tickers: list, failed_tickers: list)
    Each DataFrame has columns ["Open", "High", "Low", "Close", "Volume"].
    """
    requested = list(stocks) if isinstance(stocks, (list, tuple, set, pd.Index)) else [stocks]
    requested = [str(s) for s in requested]

    raw = yf.download(
        requested,
        period=period,
        interval=interval,
        group_by="ticker"
    )

    if raw.empty:
        for ticker in requested:
            print(f"Warning: OHLCV data unavailable for {ticker}")
        return {}, [], requested

    ohlcv_by_ticker = {}
    needed_cols = ["Open", "High", "Low", "Close", "Volume"]

    # yfinance returns a flat frame (no MultiIndex) when only one ticker is requested
    if not isinstance(raw.columns, pd.MultiIndex):
        ticker = requested[0]
        df = raw[needed_cols].dropna(how="any") if set(needed_cols).issubset(raw.columns) else pd.DataFrame()
        if not df.empty:
            ohlcv_by_ticker[ticker] = df
    else:
        top_level = raw.columns.get_level_values(0)
        for ticker in requested:
            if ticker not in top_level:
                continue
            df = raw[ticker]
            if not set(needed_cols).issubset(df.columns):
                continue
            df = df[needed_cols].dropna(how="any")
            if not df.empty:
                ohlcv_by_ticker[ticker] = df

    valid_tickers = list(ohlcv_by_ticker.keys())
    failed_tickers = [t for t in requested if t not in valid_tickers]
    for ticker in failed_tickers:
        print(f"Warning: OHLCV data unavailable for {ticker}")

    return ohlcv_by_ticker, valid_tickers, failed_tickers
