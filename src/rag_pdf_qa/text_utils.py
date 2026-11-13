"""Utilità sul testo condivise tra i moduli."""

from __future__ import annotations

import re

WORD = re.compile(r"\w+", re.UNICODE)

# Parole molto frequenti che non aiutano a capire di cosa parla un testo.
ITALIAN_STOPWORDS: frozenset[str] = frozenset(
    """
    a ad al alla alle allo agli ai anche c che chi ci come con cosa cui da dal dalla dalle dei del
    della delle dello degli di dove e ed è era essere gli ha hanno ho i il in io la le lei li lo
    loro lui ma mi ne nei nel nella nelle nello negli noi non o per perché più può puo qual quale
    quali quando quanta quante quanti quanto quella quelle quelli quello questa queste questi
    questo se si sia siamo sono su sua sue sui sul sulla suo suoi ti tra fra tu tua tue tuo tuoi
    un una uno vi voi the of and to is are
    """.split()
)


def keywords(text: str) -> set[str]:
    """Parole significative (minuscole, senza stopword e senza lettere singole)."""
    return {w for w in (m.lower() for m in WORD.findall(text)) if w not in ITALIAN_STOPWORDS and len(w) > 1}
