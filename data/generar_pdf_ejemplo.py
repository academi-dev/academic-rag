"""Script utilitario para generar un archivo PDF académico de muestra.

Crea el archivo 'data/ejemplo_apuntes.pdf' con 2 páginas de contenido universitario
(Programación Orientada a Objetos y Complejidad Algorítmica) para pruebas del RAG.
"""

from io import BytesIO
from pathlib import Path
from typing import List


def generar_pdf(pages_text: List[str], output_path: Path) -> None:
    """Construye un archivo PDF 1.4 canónico sin dependencias externas."""
    objects: List[bytes] = []

    font_content = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    font_id = 3

    page_ids: List[int] = []
    content_ids: List[int] = []

    current_obj = 4
    for _ in pages_text:
        content_ids.append(current_obj)
        current_obj += 1
        page_ids.append(current_obj)
        current_obj += 1

    catalog_obj = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids_refs = b" ".join([f"{pid} 0 R".encode("ascii") for pid in page_ids])
    pages_obj = (
        b"<< /Type /Pages /Kids [ "
        + kids_refs
        + b" ] /Count "
        + str(len(pages_text)).encode("ascii")
        + b" >>"
    )

    objects_list: List[bytes] = [catalog_obj, pages_obj, font_content]

    for text, c_id, p_id in zip(pages_text, content_ids, page_ids):
        # Limpieza básica para streams PDF
        safe_text = text.replace("(", "\\(").replace(")", "\\)")
        stream_data = (
            f"BT\n/F1 11 Tf\n40 730 Td\n15 TL\n({safe_text}) Tj\nET\n".encode("utf-8")
        )
        stream_obj = (
            f"<< /Length {len(stream_data)} >>\nstream\n".encode("ascii")
            + stream_data
            + b"endstream"
        )
        objects_list.append(stream_obj)

        page_dict = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {c_id} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        ).encode("ascii")
        objects_list.append(page_dict)

    buffer = BytesIO()
    buffer.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    offsets: List[int] = []
    for idx, obj in enumerate(objects_list, start=1):
        offsets.append(buffer.tell())
        buffer.write(f"{idx} 0 obj\n".encode("ascii"))
        buffer.write(obj)
        buffer.write(b"\nendobj\n")

    xref_pos = buffer.tell()
    buffer.write(f"xref\n0 {len(objects_list) + 1}\n".encode("ascii"))
    buffer.write(b"0000000000 65535 f \n")
    for off in offsets:
        buffer.write(f"{off:010d} 00000 n \n".encode("ascii"))

    trailer = (
        f"trailer\n<< /Size {len(objects_list) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode("ascii")
    buffer.write(trailer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(buffer.getvalue())


if __name__ == "__main__":
    paginas = [
        "Unidad 1: Programacion Orientada a Objetos. Pilares fundamentales: encapsulamiento, herencia, polimorfismo y abstraccion.",
        "Unidad 2: Analisis de Algoritmos. Complejidad asintotica y Notacion Big O para tiempo y espacio.",
    ]
    destino = Path(__file__).parent / "ejemplo_apuntes.pdf"
    generar_pdf(paginas, destino)
    print(f"Archivo PDF de ejemplo generado en: {destino}")
