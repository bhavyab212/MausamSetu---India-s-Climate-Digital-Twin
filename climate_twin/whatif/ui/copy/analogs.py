"""whatif.ui.copy.analogs — strings for the analog narrative panel."""
from __future__ import annotations

HEADLINE_TEMPLATE = "This scenario resembles {n} past years:"

CARD_TEMPLATE = (
    "**{year}** — Ya = {ya:.2f} t/ha ({pct_of_ymax:.0f}% of Ymax). "
    "SPI-3 JJAS = {spi:+.1f}, tmax anom {tmax:+.1f} °C, "
    "onset {onset:+.0f} d."
)

QUALITY_WORDS = {
    "strong": "Strong",
    "fair":   "Fair",
    "poor":   "Poor",
}

NO_STRONG_ANALOG_BODY = (
    "The scenario's outcome distribution shown here is unusually "
    "uncertain — the recommendation above should be treated as "
    "low-confidence."
)
