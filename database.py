import sqlite3
import os
import time
from pathlib import Path

DB_NAME = os.path.join(os.path.dirname(__file__), 'car_history.db')
# hidden directory for saved images (dot-prefixed so it's hidden on Unix-like systems)
IMAGE_DIR = os.path.join(os.path.dirname(__file__), '.db_images')

def init_db(db_path=None):
    dbp = db_path or DB_NAME
    conn = sqlite3.connect(dbp)
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT,
        saved_image TEXT,
        predicted TEXT,
        confidence REAL,
        correct INTEGER,
        processing_time REAL,
        timestamp INTEGER
    )
    ''')
    conn.commit()
    conn.close()
    # ensure hidden image dir exists
    try:
        os.makedirs(IMAGE_DIR, exist_ok=True)
    except Exception:
        pass
    return dbp


def save_image_copy(src_path, prefix=None):
    """Copy an image file into the hidden IMAGE_DIR and return the new path.

    This keeps saved images out of the user's visible project folder listing.
    """
    try:
        os.makedirs(IMAGE_DIR, exist_ok=True)
    except Exception:
        pass
    base = os.path.basename(src_path)
    ts = int(time.time())
    if prefix:
        name = f"{ts}_{prefix}_{base}"
    else:
        name = f"{ts}_{base}"
    dst = os.path.join(IMAGE_DIR, name)
    try:
        import shutil
        shutil.copy(src_path, dst)
        return dst
    except Exception:
        return dst

def insert_record(image_path, saved_image, predicted, confidence, correct, processing_time, timestamp, db_path=None):
    dbp = db_path or DB_NAME
    conn = sqlite3.connect(dbp)
    cur = conn.cursor()
    cur.execute('''INSERT INTO records (image_path, saved_image, predicted, confidence, correct, processing_time, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)''', (image_path, saved_image, predicted, confidence, int(bool(correct)), processing_time, int(timestamp)))
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid

def get_all_records(db_path=None):
    dbp = db_path or DB_NAME
    conn = sqlite3.connect(dbp)
    cur = conn.cursor()
    cur.execute('SELECT id, image_path, saved_image, predicted, confidence, correct, processing_time, timestamp FROM records ORDER BY timestamp DESC')
    rows = cur.fetchall()
    conn.close()
    keys = ['id','image_path','saved_image','predicted','confidence','correct','processing_time','timestamp']
    return [dict(zip(keys, r)) for r in rows]

def get_record(rid, db_path=None):
    dbp = db_path or DB_NAME
    conn = sqlite3.connect(dbp)
    cur = conn.cursor()
    cur.execute('SELECT id, image_path, saved_image, predicted, confidence, correct, processing_time, timestamp FROM records WHERE id=?', (rid,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    keys = ['id','image_path','saved_image','predicted','confidence','correct','processing_time','timestamp']
    return dict(zip(keys, row))

def export_record_pdf(rid, out_pdf_path, db_path=None):
    # uses reportlab to generate a simple PDF with the image and metadata
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.utils import ImageReader
    except Exception as e:
        raise RuntimeError('reportlab is required to export PDF')
    rec = get_record(rid, db_path=db_path)
    if rec is None:
        raise ValueError('Record not found')
    c = canvas.Canvas(out_pdf_path, pagesize=A4)
    w, h = A4
    margin = 40
    text_x = margin
    text_y = h - margin
    c.setFont('Helvetica', 12)
    c.drawString(text_x, text_y, f"Record ID: {rec['id']}")
    text_y -= 18
    c.drawString(text_x, text_y, f"Predicted: {rec['predicted']}")
    text_y -= 18
    try:
        cval = float(rec.get('confidence') or 0.0)
    except Exception:
        cval = 0.0
    c.drawString(text_x, text_y, f"Confidence: {cval:.2f}")
    text_y -= 18
    c.drawString(text_x, text_y, f"Correct: {bool(rec['correct'])}")
    text_y -= 18
    try:
        pt = float(rec.get('processing_time') or 0.0)
    except Exception:
        pt = 0.0
    c.drawString(text_x, text_y, f"Processing time (s): {pt:.3f}")
    text_y -= 18
    import datetime
    dt = datetime.datetime.fromtimestamp(rec['timestamp']).isoformat()
    c.drawString(text_x, text_y, f"Timestamp: {dt}")
    # draw image below
    try:
        img_path = rec.get('saved_image') or rec.get('image_path')
        if img_path and os.path.isfile(img_path):
            img_reader = ImageReader(img_path)
            # fit image into page width - 2*margin and remaining height
            max_w = w - 2*margin
            max_h = text_y - margin
            c.drawImage(img_reader, margin, margin, width=max_w, height=max_h, preserveAspectRatio=True, anchor='sw')
    except Exception:
        pass
    # do not add an extra blank page
    c.save()
    return out_pdf_path

def export_records_pdf(rids, out_pdf_path, db_path=None):
    """Export multiple records into a single PDF. Layout adapts to page size (A4) and places items in a 2-column grid."""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.utils import ImageReader
        from reportlab.lib.units import mm
    except Exception:
        raise RuntimeError('reportlab is required to export PDF')
    recs = []
    for rid in rids:
        r = get_record(rid, db_path=db_path)
        if r:
            recs.append(r)
    if not recs:
        raise ValueError('No valid records to export')

    c = canvas.Canvas(out_pdf_path, pagesize=A4)
    w, h = A4
    margin = 20 * mm
    usable_w = w - 2 * margin
    usable_h = h - 2 * margin
    cols = 2
    col_w = usable_w / cols
    # reserve space for caption under each image
    caption_h = 28 * mm
    # compute rows per page
    box_h = 80 * mm  # image + caption approx
    rows = max(1, int(usable_h // box_h))
    items_per_page = cols * rows

    def draw_record_at(rec, page_idx, pos_idx):
        col = pos_idx % cols
        row = pos_idx // cols
        x = margin + col * col_w
        y_top = h - margin - row * box_h
        # space for image
        img_max_w = col_w - 10 * mm
        img_max_h = box_h - caption_h - 6 * mm
        img_path = rec.get('saved_image') or rec.get('image_path')
        if img_path and os.path.isfile(img_path):
            try:
                ir = ImageReader(img_path)
                # fit while preserving aspect
                iw, ih = ir.getSize()
                scale = min(img_max_w / iw, img_max_h / ih)
                draw_w = iw * scale
                draw_h = ih * scale
                img_x = x + (col_w - draw_w) / 2
                img_y = y_top - draw_h - caption_h
                c.drawImage(ir, img_x, img_y, width=draw_w, height=draw_h, preserveAspectRatio=True)
            except Exception:
                pass
        # draw caption
        txt_x = x + 6 * mm
        txt_y = y_top - box_h + caption_h - 6 * mm
        c.setFont('Helvetica', 10)
        try:
            cval = float(rec.get('confidence') or 0.0)
        except Exception:
            cval = 0.0
        try:
            pt = float(rec.get('processing_time') or 0.0)
        except Exception:
            pt = 0.0
        c.drawString(txt_x, txt_y + 14, f"ID: {rec['id']}  Predicted: {rec.get('predicted')} ({cval:.2f}%)")
        c.drawString(txt_x, txt_y, f"Correct: {bool(rec.get('correct'))}  Time: {pt:.3f}s")

    idx = 0
    for i, rec in enumerate(recs):
        page_idx = idx // items_per_page
        pos_idx = idx % items_per_page
        if pos_idx == 0 and idx != 0:
            c.showPage()
        draw_record_at(rec, page_idx, pos_idx)
        idx += 1

    # do not add an extra blank page at the end
    c.save()
    return out_pdf_path


def update_record_correct(rid, correct, db_path=None):
    """Update the 'correct' flag for a record."""
    dbp = db_path or DB_NAME
    conn = sqlite3.connect(dbp)
    cur = conn.cursor()
    cur.execute('UPDATE records SET correct=? WHERE id=?', (int(bool(correct)), rid))
    conn.commit()
    conn.close()
    return True

def export_records_table_pdf(rids, out_pdf_path, db_path=None):
    """Export multiple records into a single PDF as a clean, paginated table.

    Columns: ID, Predykcja, Pewność (%), Poprawny, Czas (s), Znacznik czasu
    Uses ReportLab Platypus for automatic pagination and header repetition.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.utils import ImageReader
        from reportlab.lib.units import mm
    except Exception:
        raise RuntimeError('reportlab is required to export PDF')

    recs = []
    for rid in rids:
        r = get_record(rid, db_path=db_path)
        if r:
            recs.append(r)
    if not recs:
        raise ValueError('No valid records to export')

    doc = SimpleDocTemplate(out_pdf_path, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    title = Paragraph('Historia predykcji – eksport (z miniaturami)', styles['Heading2'])
    story.append(title)
    story.append(Spacer(1, 8))

    data = [[
        'Obraz', 'ID', 'Predykcja', 'Pewność (%)', 'Poprawny', 'Czas (s)', 'Znacznik czasu'
    ]]
    thumb_w = 28 * mm
    thumb_h = 20 * mm
    for rec in recs:
        try:
            cval = float(rec.get('confidence') or 0.0)
        except Exception:
            cval = 0.0
        try:
            pt = float(rec.get('processing_time') or 0.0)
        except Exception:
            pt = 0.0
        try:
            import datetime
            ts = rec.get('timestamp')
            ts_val = int(ts) if ts is not None else 0
            dt = datetime.datetime.fromtimestamp(ts_val).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            dt = ''
        ok = 'TAK' if bool(rec.get('correct')) else 'NIE'
        # build thumbnail from saved_image (temp) or fallback to image_path
        img_path = rec.get('saved_image') or rec.get('image_path')
        thumb = Paragraph('—', styles['Normal'])
        if img_path and os.path.isfile(img_path):
            try:
                ir = ImageReader(img_path)
                iw, ih = ir.getSize()
                scale = min(thumb_w / iw, thumb_h / ih)
                tw = max(1, iw * scale)
                th = max(1, ih * scale)
                thumb = RLImage(img_path, width=tw, height=th)
            except Exception:
                pass
        data.append([
            thumb,
            str(rec.get('id') or ''),
            str(rec.get('predicted') or ''),
            f"{cval:.2f}",
            ok,
            f"{pt:.3f}",
            dt,
        ])

    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e7f2ff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0b1f44')),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor('#f8fafc')]),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 1), (0, -1), 'CENTER'),  # image column center
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))

    story.append(table)
    doc.build(story)
    return out_pdf_path
