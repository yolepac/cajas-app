# CAJAS App v3 (OCR con Tesseract y EasyOCR)

- OCR de nombres con **Tesseract** (si está) o **EasyOCR** como fallback (pip-only).
- PDF → imagen con **PyMuPDF** (no requiere binarios).
- Clasificación de celdas por **muestra de color**.

## Despliegue (Streamlit Cloud)
1. Crea repo en GitHub con `app.py`, `requirements.txt`, `README.md`.
2. En Streamlit Cloud, crea app apuntando a `app.py`.
3. Usa la app: sube PDF/imagen, ajusta recortes, define horario, sube muestra de color, procesa, descarga CSV.
