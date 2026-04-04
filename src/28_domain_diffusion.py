"""28 – URL Domain Diffusion (H21).

Extends the bridge analysis (H14) to the *domain* level.  Rather than
matching exact URLs, we track which *domains* IM users share in
Siege-relevant posts and measure whether those same domains later
appear in /pol/'s Siege-filtered corpus.

Key outputs
-----------
- Domain-level temporal priority (which platform posted a domain first).
- Per-domain lag distribution.
- "Gateway domain" identification: domains introduced by IM that later
  become popular on /pol/.
- Weekly domain-overlap coefficient time series.
"""

from __future__ import annotations

import datetime
import json
import re
from collections import Counter

import numpy as np
import polars as pl

from utils import (
    DATA_PROCESSED, RESULTS_DIR, ZEIGER_MEMBER_ID,
    setup_plot_style, save_figure, FIGSIZE_WIDE, CB_PALETTE,
)
import matplotlib.pyplot as plt

# Reuse URL extraction from bridges module
_URL_PATTERN = re.compile(r'https?://[^\s<>"\')\]]+', re.IGNORECASE)
_NOISE_DOMAINS = frozenset({
    "imgur.com", "i.imgur.com", "youtube.com", "youtu.be",
    "twitter.com", "facebook.com", "google.com", "wikipedia.org",
    "reddit.com", "boards.4chan.org", "4chan.org",
    "archive.org", "web.archive.org", "en.wikipedia.org",
    "t.co", "bit.ly", "goo.gl", "tinyurl.com",
})


def _extract_domain(url: str) -> str | None:
    """Extract domain from URL, stripping www."""
    match = re.match(r'https?://(?:www\.)?([^/]+)', url)
    return match.group(1).lower() if match else None


def extract_domains_with_dates(
    df: pl.DataFrame,
) -> dict[str, datetime.datetime]:
    """Return {domain: earliest_date} from a scored dataframe."""
    domain_dates: dict[str, datetime.datetime] = {}
    for row in df.iter_rows(named=True):
        text = row.get("text") or ""
        date = row.get("date")
        if not date or not text:
            continue
        for url in _URL_PATTERN.findall(text):
            url = url.rstrip(".,;:!?)>")
            domain = _extract_domain(url)
            if domain and domain not in _NOISE_DOMAINS:
                if domain not in domain_dates or date < domain_dates[domain]:
                    domain_dates[domain] = date
    return domain_dates


def domain_temporal_priority(
    im_domains: dict[str, datetime.datetime],
    pol_domains: dict[str, datetime.datetime],
) -> dict:
    """Determine which platform posted each shared domain first."""
    shared = set(im_domains.keys()) & set(pol_domains.keys())
    im_first = 0
    pol_first = 0
    simultaneous = 0
    domain_records = []

    for domain in shared:
        im_date = im_domains[domain]
        pol_date = pol_domains[domain]
        diff_days = (pol_date - im_date).total_seconds() / 86400

        if diff_days > 1:
            source = "im"
            im_first += 1
        elif diff_days < -1:
            source = "pol"
            pol_first += 1
        else:
            source = "simultaneous"
            simultaneous += 1

        domain_records.append({
            "domain": domain,
            "im_first": str(im_date),
            "pol_first": str(pol_date),
            "lag_days": round(diff_days, 1),
            "source": source,
        })

    domain_records.sort(key=lambda r: r["lag_days"])

    return {
        "total_im_domains": len(im_domains),
        "total_pol_domains": len(pol_domains),
        "shared_domains": len(shared),
        "im_first": im_first,
        "pol_first": pol_first,
        "simultaneous": simultaneous,
        "domain_records": domain_records,
    }


def identify_gateway_domains(
    im_domains: dict[str, datetime.datetime],
    pol_posts: pl.DataFrame,
    min_pol_count: int = 5,
) -> list[dict]:
    """Find domains introduced by IM that became popular on /pol/.

    A 'gateway domain' is one that:
    - Appeared on IM first
    - Appears ≥ min_pol_count times on /pol/
    """
    # Count domain frequency on /pol/
    pol_domain_counts: Counter[str] = Counter()
    pol_first_dates: dict[str, datetime.datetime] = {}

    for row in pol_posts.iter_rows(named=True):
        text = row.get("text") or ""
        date = row.get("date")
        if not date or not text:
            continue
        for url in _URL_PATTERN.findall(text):
            url = url.rstrip(".,;:!?)>")
            domain = _extract_domain(url)
            if domain and domain not in _NOISE_DOMAINS:
                pol_domain_counts[domain] += 1
                if domain not in pol_first_dates or date < pol_first_dates[domain]:
                    pol_first_dates[domain] = date

    gateways = []
    for domain, im_date in im_domains.items():
        pol_count = pol_domain_counts.get(domain, 0)
        if pol_count < min_pol_count:
            continue
        pol_date = pol_first_dates.get(domain)
        if pol_date and pol_date > im_date:
            lag = (pol_date - im_date).total_seconds() / 86400
            gateways.append({
                "domain": domain,
                "im_first": str(im_date),
                "pol_first": str(pol_date),
                "lag_days": round(lag, 1),
                "pol_count": pol_count,
            })

    gateways.sort(key=lambda r: r["pol_count"], reverse=True)
    return gateways


def weekly_domain_overlap(
    im: pl.DataFrame, pol: pl.DataFrame,
) -> pl.DataFrame:
    """Compute weekly Jaccard overlap of domains between platforms."""
    weeks_data = []

    # Build weekly domain sets for each platform
    im_weekly: dict[str, set[str]] = {}
    for row in im.filter(pl.col("date").is_not_null()).iter_rows(named=True):
        text = row.get("text") or ""
        date = row.get("date")
        if not date or not text:
            continue
        week = date.strftime("%G-W%V")
        if week not in im_weekly:
            im_weekly[week] = set()
        for url in _URL_PATTERN.findall(text):
            d = _extract_domain(url.rstrip(".,;:!?)>"))
            if d and d not in _NOISE_DOMAINS:
                im_weekly[week].add(d)

    pol_weekly: dict[str, set[str]] = {}
    for row in pol.filter(pl.col("date").is_not_null()).iter_rows(named=True):
        text = row.get("text") or ""
        date = row.get("date")
        if not date or not text:
            continue
        week = date.strftime("%G-W%V")
        if week not in pol_weekly:
            pol_weekly[week] = set()
        for url in _URL_PATTERN.findall(text):
            d = _extract_domain(url.rstrip(".,;:!?)>"))
            if d and d not in _NOISE_DOMAINS:
                pol_weekly[week].add(d)

    common_weeks = sorted(set(im_weekly.keys()) & set(pol_weekly.keys()))
    for week in common_weeks:
        im_set = im_weekly[week]
        pol_set = pol_weekly[week]
        union = im_set | pol_set
        inter = im_set & pol_set
        jaccard = len(inter) / len(union) if union else 0
        weeks_data.append({
            "week": week,
            "im_domains": len(im_set),
            "pol_domains": len(pol_set),
            "shared": len(inter),
            "jaccard": jaccard,
        })

    return pl.DataFrame(weeks_data)


def plot_domain_lags(priority: dict, filename: str):
    """Histogram of domain-level appearance lags."""
    setup_plot_style()
    records = priority.get("domain_records", [])
    if not records:
        return

    lags = [r["lag_days"] for r in records if abs(r["lag_days"]) < 1000]
    if not lags:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.hist(lags, bins=40, color=CB_PALETTE[0], alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Simultaneous")
    median_lag = float(np.median(lags))
    ax.axvline(median_lag, color=CB_PALETTE[2], linestyle="-",
               linewidth=2, label=f"Median: {median_lag:.0f} d")
    ax.set_xlabel("Lag (days; positive = IM first)")
    ax.set_ylabel("Domains")
    ax.set_title(f"Domain Appearance Lag (IM-first: {priority['im_first']}, "
                 f"/pol/-first: {priority['pol_first']})")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, filename)


def plot_gateway_domains(gateways: list[dict], filename: str):
    """Bar chart of top gateway domains by /pol/ frequency."""
    setup_plot_style()
    if not gateways:
        return

    top = gateways[:20]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        [g["domain"] for g in top][::-1],
        [g["pol_count"] for g in top][::-1],
        color=CB_PALETTE[0], alpha=0.8,
    )
    ax.set_xlabel("/pol/ Post Count")
    ax.set_title("Gateway Domains: IM-Introduced, Popular on /pol/")
    fig.tight_layout()
    save_figure(fig, filename)


def main():
    print("=" * 60)
    print("H21: URL Domain Diffusion")
    print("=" * 60)

    im_path = DATA_PROCESSED / "siege_scores.parquet"
    pol_path = DATA_PROCESSED / "pol_siege_scores.parquet"
    if not im_path.exists() or not pol_path.exists():
        print("  ✗ Missing scored data.")
        return

    im = pl.read_parquet(im_path).filter(
        (pl.col("channel") == "forum")
        & (pl.col("author_id") != ZEIGER_MEMBER_ID)
    )
    pol = pl.read_parquet(pol_path)
    print(f"  IM posts: {im.height:,}  |  /pol/ posts: {pol.height:,}")

    # Extract domains
    print("\n  Extracting domains from IM…")
    im_domains = extract_domains_with_dates(im)
    print(f"    IM unique domains: {len(im_domains):,}")

    print("  Extracting domains from /pol/…")
    pol_domains = extract_domains_with_dates(pol)
    print(f"    /pol/ unique domains: {len(pol_domains):,}")

    # Temporal priority
    print("\n  Domain temporal priority…")
    priority = domain_temporal_priority(im_domains, pol_domains)
    print(f"    Shared: {priority['shared_domains']}  "
          f"(IM-first: {priority['im_first']}, "
          f"/pol/-first: {priority['pol_first']}, "
          f"simultaneous: {priority['simultaneous']})")

    # Gateway domains
    print("\n  Identifying gateway domains…")
    gateways = identify_gateway_domains(im_domains, pol)
    print(f"    Gateway domains (IM→/pol/, ≥5 /pol/ posts): {len(gateways)}")
    for g in gateways[:10]:
        print(f"      {g['domain']}: {g['pol_count']} posts, "
              f"lag={g['lag_days']:.0f} d")

    # Plots
    plot_domain_lags(priority, "domain_diffusion_lags")
    plot_gateway_domains(gateways, "gateway_domains")

    # Save
    results = {
        "im_unique_domains": len(im_domains),
        "pol_unique_domains": len(pol_domains),
        "temporal_priority": {
            k: v for k, v in priority.items() if k != "domain_records"
        },
        "domain_records": priority.get("domain_records", [])[:100],
        "gateway_domains": gateways[:50],
    }

    with open(RESULTS_DIR / "domain_diffusion_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n✓ Domain diffusion results saved.")


if __name__ == "__main__":
    main()
