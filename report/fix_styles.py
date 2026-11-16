"""Post-process the pandoc-built docx so it renders the same in Word, Google Docs and LibreOffice.

The sprint template defines none of pandoc's default styles (BodyText, Compact, Table, ...), and
references to missing styles break table layout in LibreOffice. This drops those references and
gives tables explicit borders and widths.

Build:  pandoc report.md --reference-doc=<template.docx> --resource-path=.:.. -o out.docx
        python fix_styles.py out.docx
"""
import re, shutil, sys, zipfile

src = sys.argv[1]
BORDERS = "".join(f'<w:{k} w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
                  for k in ("top", "left", "bottom", "right", "insideH", "insideV"))
with zipfile.ZipFile(src) as z:
    defined = set(re.findall(r'w:styleId="([^"]+)"', z.read("word/styles.xml").decode()))


def fix(x):
    x = x.replace('<w:rStyle w:val="VerbatimChar" />', '<w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" />')
    x = re.sub(r'<w:tblStyle w:val="Table" /><w:tblW [^>]*/>(<w:tblLayout [^>]*/>)?',
               f'<w:tblW w:type="dxa" w:w="9360" /><w:tblBorders>{BORDERS}</w:tblBorders>'
               '<w:tblCellMar><w:left w:w="100" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tblCellMar>', x)
    x = x.replace('<w:gridCol w:w="3960" /><w:gridCol w:w="3960" />', '<w:gridCol w:w="2200" /><w:gridCol w:w="7160" />')
    x = x.replace('<w:gridCol w:w="1320" />', '<w:gridCol w:w="1560" />')
    def drop(m):  # undefined style -> Normal, keeping paragraph spacing for body text and captions
        if m.group(1) in defined:
            return m.group(0)
        if m.group(1) == "TableCaption":
            return '<w:keepNext /><w:spacing w:before="160" w:after="60" />'
        return "" if m.group(1) == "Compact" else '<w:spacing w:before="60" w:after="140" />'
    return re.sub(r'<w:(?:pStyle|rStyle) w:val="([^"]+)" />', drop, x)


with zipfile.ZipFile(src) as zi, zipfile.ZipFile(src + ".tmp", "w", zipfile.ZIP_DEFLATED) as zo:
    for it in zi.infolist():
        data = zi.read(it.filename)
        zo.writestr(it, fix(data.decode()).encode() if it.filename == "word/document.xml" else data)
shutil.move(src + ".tmp", src)
