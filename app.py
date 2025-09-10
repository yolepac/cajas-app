
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
import datetime as dt
import fitz  # PyMuPDF for PDFs

# OCR backends
import pytesseract
try:
    import easyocr
    EASY_AVAILABLE = True
except Exception:
    EASY_AVAILABLE = False

st.set_page_config(page_title="Detector de CAJAS por color (OCR nombres)", layout="wide")
st.title("Detector de horas de **CAJAS** con **OCR de nombres**")

with st.expander("Cómo funciona", expanded=True):
    st.markdown('''
1) Sube el horario (PDF/JPG/PNG).  
2) Ajusta recortes: **columna de nombres** (izquierda) y **zona de la rejilla** (derecha).  
3) Define **rango horario** y **paso** (30 min).  
4) Sube una **muestra de color** de *CAJAS*.  
5) Procesa → exporta **CSV**.
- El OCR de nombres usa **Tesseract** si está disponible; si no, intenta **EasyOCR**; si tampoco, podrás editar los nombres a mano.
''')

uploaded = st.file_uploader("Sube el horario (PDF/JPG/PNG)", type=["pdf","jpg","jpeg","png"])

col1, col2, col3 = st.columns(3)
with col1:
    start_time = st.time_input("Hora inicio", value=dt.time(9,0))
with col2:
    end_time   = st.time_input("Hora fin", value=dt.time(22,0))
with col3:
    slot_mins  = st.selectbox("Paso (min)", [30, 15], index=0)

slot_count = int((dt.datetime.combine(dt.date.today(), end_time) - dt.datetime.combine(dt.date.today(), start_time)).seconds // 60 // slot_mins)
if slot_count <= 0:
    st.error("Revisa el rango horario: debe ser mayor que 0.")
    st.stop()

def build_time_slots(start, end, step):
    t = pd.date_range(pd.to_datetime(start.strftime("%H:%M")), pd.to_datetime(end.strftime("%H:%M")), freq=f"{step}min", inclusive="left")
    return [f"{t[i].strftime('%H:%M')}-{t[i+1].strftime('%H:%M')}" for i in range(len(t)-1)]

slots = build_time_slots(start_time, end_time, slot_mins)

def pdf_first_page_to_pil(pdf_bytes, zoom=2.0):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

def ocr_names_from_image(img_rgb):
    \"\"\"
    Devuelve lista de nombres leídos de la columna izquierda.
    Intenta: 1) pytesseract, 2) easyocr (si está instalado).
    \"\"\"
    names = []

    # Preprocesado suave para OCR
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    # Escalado para mejorar OCR
    scale = 1.5
    resized = cv2.resize(gray, (int(gray.shape[1]*scale), int(gray.shape[0]*scale)), interpolation=cv2.INTER_LINEAR)

    # 1) Tesseract
    try:
        cfg = "--oem 3 --psm 6"
        txt = pytesseract.image_to_string(resized, config=cfg, lang="spa")
        for line in txt.splitlines():
            line = line.strip()
            if len(line) < 2: continue
            if sum(c.isalpha() for c in line) < 2: continue
            names.append(line)
        if len(names) >= 1:
            return names
    except Exception:
        pass

    # 2) EasyOCR
    if EASY_AVAILABLE:
        try:
            reader = easyocr.Reader(['es', 'en'], gpu=False, verbose=False)
            result = reader.readtext(resized)
            for item in result:
                # item: [bbox, text, conf]
                text = item[1]
                line = text.strip()
                if len(line) < 2: continue
                if sum(c.isalpha() for c in line) < 2: continue
                names.append(line)
            if len(names) >= 1:
                return names
        except Exception:
            pass

    return names  # puede estar vacío

def rgb_to_lab_mean(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    mean_lab = img_lab.reshape(-1, 3).mean(axis=0)
    return mean_lab

if uploaded is not None:
    # Cargar imagen base
    if uploaded.type == "application/pdf":
        base_pil = pdf_first_page_to_pil(uploaded.read(), zoom=2.0)
    else:
        base_pil = Image.open(uploaded).convert("RGB")
    base = np.array(base_pil)  # RGB
    h, w = base.shape[:2]

    st.subheader("1) Recortes (calibración visual)")
    top_pct    = st.slider("Recorte superior (%)", 0, 30, 0, 1)
    bottom_pct = st.slider("Recorte inferior (%)", 0, 30, 0, 1)
    top_px     = int(h * top_pct / 100.0)
    bot_px     = int(h * (1 - bottom_pct/100.0))
    crop_v     = base[top_px:bot_px, :, :]
    ch, cw = crop_v.shape[:2]

    names_width_pct = st.slider("Ancho columna de nombres (%)", 5, 50, 25, 1)
    sep_px = int(cw * names_width_pct / 100.0)

    names_roi = crop_v[:, :sep_px, :]
    grid_roi  = crop_v[:, sep_px:, :]

    prev1, prev2 = st.columns(2)
    with prev1:
        st.image(names_roi, caption="Columna de nombres (ROI)", use_container_width=True)
    with prev2:
        st.image(grid_roi, caption="Zona de rejilla (ROI)", use_container_width=True)

    st.subheader("2) Nombres (OCR automático con edición)")
    st.caption("Intento leer con **Tesseract**; si no, con **EasyOCR**; si no, edítalos a mano.")
    names = ocr_names_from_image(names_roi)

    if len(names) == 0:
        names = ["(Nombre 1)", "(Nombre 2)"]

    names_text = st.text_area("Corrige si hace falta (uno por línea):", value="\\n".join(names), height=150)
    names = [n.strip() for n in names_text.splitlines() if n.strip()]
    row_count = len(names)

    st.subheader("3) Muestra de color para **CAJAS**")
    st.caption("Sube un recorte pequeñito (PNG/JPG) de una celda que sepas que es de CAJAS.")
    sample = st.file_uploader("Muestra de color (imagen pequeña)", type=["png","jpg","jpeg"], key="sample")
    lab_target = None
    th = st.slider("Umbral de similitud (ΔE aprox.)", 5, 50, 15, 1)

    if sample is not None:
        samp_img = Image.open(sample).convert("RGB")
        samp_np = np.array(samp_img)
        lab_target = rgb_to_lab_mean(samp_np)
        st.image(samp_np, caption="Muestra de color CAJAS", width=200)

    st.subheader("4) Generar tabla")
    st.caption("Filas = nº de nombres; columnas = tramos. Clasificación por color con tu muestra.")
    gen = st.button("Procesar")

    if gen:
        if lab_target is None:
            st.warning("Sube una muestra de color para CAJAS antes de procesar.")
        else:
            gh, gw = grid_roi.shape[:2]
            row_heights = np.linspace(0, gh, row_count+1).astype(int)
            col_widths = np.linspace(0, gw, len(slots)+1).astype(int)

            def delta_e(lab1, lab2):
                return np.linalg.norm(lab1 - lab2)

            data = []
            for r in range(row_count):
                row_label = names[r]
                row_cells_flags = []
                for c in range(len(slots)):
                    y0, y1 = row_heights[r], row_heights[r+1]
                    x0, x1 = col_widths[c], col_widths[c+1]
                    cell = grid_roi[y0:y1, x0:x1, :]
                    if cell.size == 0 or min(cell.shape[:2]) < 3:
                        row_cells_flags.append(False)
                        continue
                    # grid_roi está en RGB
                    lab_cell = rgb_to_lab_mean(cell)
                    d = delta_e(lab_cell, lab_target)
                    row_cells_flags.append(d <= th)
                # Unir tramos contiguos
                tramos = []
                start_idx = None
                for idx, flag in enumerate(row_cells_flags):
                    if flag and start_idx is None:
                        start_idx = idx
                    if (not flag or idx == len(row_cells_flags)-1) and start_idx is not None:
                        end_idx = idx if not flag else idx+1
                        tramos.append(f"{slots[start_idx].split('-')[0]}-{slots[end_idx-1].split('-')[1]}")
                        start_idx = None
                data.append((row_label, ", ".join(tramos)))

            df = pd.DataFrame(data, columns=["Gerente", "Tramos CAJAS"])
            st.success("Proceso completado.")
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar CSV", csv, "cajas_diario.csv", "text/csv")
