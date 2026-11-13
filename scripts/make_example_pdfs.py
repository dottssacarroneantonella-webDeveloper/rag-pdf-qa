"""Genera i PDF di esempio in examples/ (documenti fittizi, usati per demo e test).

Uso:  python scripts/make_example_pdfs.py   (richiede: pip install reportlab)
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent.parent / "examples"

REGOLAMENTO = [
    ("Regolamento della Biblioteca Civica di Valverde", None),
    ("Documento fittizio creato per dimostrare il progetto rag-pdf-qa.", "i"),
    ("Art. 1 – Orari di apertura", "h"),
    (
        "La Biblioteca Civica di Valverde è aperta dal lunedì al venerdì dalle 9:00 alle 19:00 "
        "e il sabato dalle 9:00 alle 13:00. La biblioteca resta chiusa la domenica, nei giorni "
        "festivi e nelle due settimane centrali di agosto.",
        None,
    ),
    ("Art. 2 – Iscrizione", "h"),
    (
        "L'iscrizione è gratuita e aperta a tutti. Per iscriversi è necessario presentare un "
        "documento di identità valido e il codice fiscale. I minori di 14 anni si iscrivono con "
        "la firma di un genitore. La tessera ha validità di tre anni e si rinnova allo sportello.",
        None,
    ),
    "BREAK",
    ("Art. 3 – Prestito", "h"),
    (
        "Ogni utente iscritto può prendere in prestito fino a cinque libri contemporaneamente. "
        "La durata del prestito è di 30 giorni. Il prestito può essere rinnovato una sola volta "
        "per altri 15 giorni, purché il libro non sia stato prenotato da un altro utente. "
        "I DVD hanno un prestito di 7 giorni e non sono rinnovabili.",
        None,
    ),
    ("Art. 4 – Ritardi e sanzioni", "h"),
    (
        "In caso di ritardo nella restituzione il prestito viene sospeso per un numero di giorni "
        "pari ai giorni di ritardo. Dopo 60 giorni di ritardo l'utente riceve un sollecito scritto. "
        "I libri smarriti o danneggiati devono essere sostituiti con una copia identica o rimborsati.",
        None,
    ),
    "BREAK",
    ("Art. 5 – Sala studio e servizi digitali", "h"),
    (
        "La sala studio dispone di 40 posti e si prenota online fino a sette giorni prima. "
        "La rete Wi-Fi gratuita è disponibile con le credenziali della tessera. "
        "Il prestito di ebook avviene tramite la piattaforma digitale della rete bibliotecaria "
        "provinciale, con un massimo di due ebook al mese per utente.",
        None,
    ),
    ("Art. 6 – Contatti", "h"),
    (
        "Per informazioni è possibile scrivere all'indirizzo email info@biblioteca-valverde.example "
        "oppure rivolgersi allo sportello durante l'orario di apertura.",
        None,
    ),
]


def build(path: Path, content: list) -> None:
    styles = getSampleStyleSheet()
    story = []
    for item in content:
        if item == "BREAK":
            story.append(PageBreak())
            continue
        text, kind = item
        style = {None: styles["BodyText"], "h": styles["Heading2"], "i": styles["Italic"]}[kind]
        if kind is None and not story:
            style = styles["Title"]
        story.append(Paragraph(text, style))
        story.append(Spacer(1, 6))
    SimpleDocTemplate(str(path), pagesize=A4, title=content[0][0]).build(story)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build(OUT / "regolamento_biblioteca.pdf", REGOLAMENTO)
    print(f"Creato {OUT / 'regolamento_biblioteca.pdf'}")
