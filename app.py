import requests
import time
import os
import json
import io
import uuid
import threading
import re
import logging
from fpdf import FPDF
from collections import Counter
from flask import Flask, request, render_template_string, send_from_directory, redirect, url_for, session, jsonify
from PIL import Image, ImageOps

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# KONFIGURATION (alle Pfade via Umgebungsvariablen steuerbar)
# Auf Render wird die Persistent Disk unter /data gemountet.
# Lokal kann DATA_DIR auf ein beliebiges Verzeichnis gesetzt werden.
# ---------------------------------------------------------------------------
DATA_DIR      = os.environ.get('DATA_DIR', '/data')
CARDS_DIR     = os.path.join(DATA_DIR, 'card_images')
OUTPUT_DIR    = os.path.join(DATA_DIR, 'output_pdfs')
CARD_BACKS_DIR= os.path.join(DATA_DIR, 'card_backs')
UPLOADS_DIR   = os.path.join(DATA_DIR, 'user_uploads')
MODELS_DIR    = os.path.join(DATA_DIR, 'models')

# Upscaling ein-/ausschalten (z.B. im Render-Dashboard als Env-Var setzen)
UPSCALE_ENABLED = os.environ.get('UPSCALE_ENABLED', 'true').lower() == 'true'

LANGUAGES = {
    'de': 'Deutsch', 'en': 'Englisch', 'es': 'Spanisch', 'fr': 'Französisch',
    'it': 'Italienisch', 'ja': 'Japanisch', 'ko': 'Koreanisch', 'pt': 'Portugiesisch',
    'ru': 'Russisch', 'zhs': 'Vereinfachtes Chinesisch',
}

# --- Globale Variable für Hintergrund-Tasks ---
tasks = {}

# --- Initialisierung der Flask-App ---
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'bitte-aendern-vor-deployment')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024


# ===========================================================================
# ██████████  REAL-ESRGAN UPSCALER  ████████████████████████████████████████
# ===========================================================================
# Verwendetes Modell: RealESRGAN_x2plus
#   - Speziell auf reale Fotos/Scans trainiert → ideal für MTG-Karten
#   - x2 Upscaling reicht aus: Scryfall low-res (~488×680) → ~976×1360 px
#   - Tile-basierte Verarbeitung: hält RAM-Verbrauch auf ~1-1.5 GB begrenzt
#   - CPU-Modus: kein GPU nötig, läuft auf jedem Render-Plan
#
# Voraussetzung (wird im Build-Befehl installiert):
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
#   pip install basicsr realesrgan opencv-python-headless
# ===========================================================================

_upscaler_instance = None
_upscaler_lock     = threading.Lock()
_upscaler_ready    = False   # True sobald Modell geladen

def _init_upscaler():
    """
    Lädt das Real-ESRGAN-Modell einmalig, thread-safe.
    Wird beim App-Start in einem Daemon-Thread aufgerufen.
    """
    global _upscaler_instance, _upscaler_ready
    with _upscaler_lock:
        if _upscaler_instance is not None:
            return _upscaler_instance
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model_path = os.path.join(MODELS_DIR, 'RealESRGAN_x2plus.pth')
            os.makedirs(MODELS_DIR, exist_ok=True)

            # Modell herunterladen falls nicht vorhanden (~67 MB)
            if not os.path.exists(model_path):
                logger.info("Lade Real-ESRGAN x2plus Modell herunter (~67 MB)...")
                url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
                r = requests.get(url, stream=True, timeout=180)
                r.raise_for_status()
                with open(model_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
                logger.info("Modell-Download abgeschlossen.")

            # RRDBNet-Architektur für x2-Modell
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=23, num_grow_ch=32,
                scale=2
            )
            _upscaler_instance = RealESRGANer(
                scale=2,
                model_path=model_path,
                model=model,
                # tile=256: verarbeitet Bild in 256×256 Kacheln → ~1 GB RAM statt 4+ GB
                tile=256,
                tile_pad=10,
                pre_pad=0,
                half=False,    # half=True nur mit CUDA; CPU braucht float32
                device='cpu',
            )
            _upscaler_ready = True
            logger.info("✅ Real-ESRGAN x2plus erfolgreich geladen.")
            return _upscaler_instance

        except ImportError as e:
            logger.warning(
                f"⚠️  Real-ESRGAN-Pakete nicht gefunden ({e}).\n"
                "    Installiere: pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                "                 pip install basicsr realesrgan opencv-python-headless\n"
                "    Fallback: PIL LANCZOS wird verwendet."
            )
            return None
        except Exception as e:
            logger.warning(f"⚠️  Real-ESRGAN konnte nicht geladen werden: {e}. Fallback: PIL LANCZOS.")
            return None


def upscale_image(source_path: str, dest_path: str) -> bool:
    """
    Upscaled ein Kartenbild mit Real-ESRGAN x2.
    Fallback: PIL LANCZOS (immer noch deutlich besser als keine Skalierung).
    Gibt True zurück wenn eine upscaled-Version gespeichert wurde.
    """
    upscaler = _init_upscaler()

    # ---- Real-ESRGAN (bestes Ergebnis) ----
    if upscaler is not None:
        try:
            import cv2
            import numpy as np

            img_bgr = cv2.imread(source_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError(f"cv2 konnte Bild nicht öffnen: {source_path}")

            logger.info(f"  🔬 Real-ESRGAN Upscaling: {os.path.basename(source_path)}")
            output_bgr, _ = upscaler.enhance(img_bgr, outscale=2)
            cv2.imwrite(dest_path, output_bgr, [cv2.IMWRITE_JPEG_QUALITY, 97])
            logger.info(f"  ✅ Upscaling fertig: {os.path.basename(dest_path)}")
            return True
        except Exception as e:
            logger.error(f"  Real-ESRGAN Fehler ({e}), falle auf PIL zurück.")

    # ---- PIL LANCZOS Fallback ----
    try:
        img = Image.open(source_path)
        w, h = img.size
        # Ziel: mindestens 745×1040 (Scryfall high-res Standard)
        TARGET_W, TARGET_H = 976, 1360
        if w < TARGET_W or h < TARGET_H:
            scale = max(TARGET_W / w, TARGET_H / h)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        img.save(dest_path, 'JPEG', quality=95)
        logger.info(f"  ✅ PIL LANCZOS Fallback: {os.path.basename(dest_path)}")
        return True
    except Exception as e:
        logger.error(f"  PIL Fallback fehlgeschlagen: {e}")
        return False


def _preload_upscaler_background():
    """Startet die Modell-Initialisierung beim App-Start im Hintergrund."""
    if UPSCALE_ENABLED:
        logger.info("Starte Upscaler-Vorladung im Hintergrund...")
        _init_upscaler()
    else:
        logger.info("Upscaling deaktiviert (UPSCALE_ENABLED=false).")


# ===========================================================================
# HILFSFUNKTIONEN (unverändert + Upscaling-Integration in get_image_by_id)
# ===========================================================================

def fetch_all_pages(api_url):
    all_cards = []
    while api_url:
        try:
            response = requests.get(api_url, timeout=15)
            response.raise_for_status()
            data = response.json()
            all_cards.extend(data.get('data', []))
            api_url = data.get('next_page') if data.get('has_more') else None
            if api_url: time.sleep(0.1)
        except requests.exceptions.RequestException as e:
            logger.error(f"Fehler beim Abrufen der Scryfall-Daten: {e}")
            return []
    return all_cards


def find_specific_card_printing(card_name, set_code, lang='de'):
    """Sucht eine Karte in einer spezifischen Edition mit robuster Sprachlogik."""
    logger.info(f"Suche gezielt nach '{card_name}' in Edition '{set_code.upper()}'...")
    search_query = f'!"{card_name}" set:{set_code}'
    params = {'q': search_query}
    api_url = "https://api.scryfall.com/cards/search?" + requests.compat.urlencode(params)
    base_prints = fetch_all_pages(api_url)

    if not base_prints:
        return None, f"Karte '{card_name}' in Edition '{set_code.upper()}' nicht gefunden."

    base_print = base_prints[0]
    final_print = None

    if lang != 'en':
        logger.info(f"-> Versuche '{lang}'-Version für '{base_print['name']}' ({base_print['set'].upper()})...")
        try:
            lang_url = f"https://api.scryfall.com/cards/{base_print['set']}/{base_print['collector_number']}/{lang}"
            res = requests.get(lang_url, timeout=10)
            res.raise_for_status()
            final_print = res.json()
            logger.info(f"-> '{lang}'-Version erfolgreich gefunden.")
        except requests.exceptions.RequestException as e:
            status = e.response.status_code if hasattr(e, 'response') and e.response else 'N/A'
            logger.warning(f"-> Keine '{lang}'-Version (Fehler: {status}). Nutze EN als Fallback.")
            final_print = base_print
    else:
        final_print = base_print

    is_highres = final_print.get('image_status') == 'highres_scan'
    final_print['quality'] = 'H' if is_highres else 'L'
    final_print['en_highres_fallback'] = None
    return [final_print], None


def find_card_printings(card_name, lang='de', filter_by_artwork=True):
    logger.info(f"Suche alle Drucke für '{card_name}' (Artwork-Filter: {filter_by_artwork})...")
    search_query = f'!"{card_name}" unique:prints'
    params = {'q': f'{search_query} lang:{lang}', 'order': 'released', 'dir': 'desc'}
    initial_url = "https://api.scryfall.com/cards/search?" + requests.compat.urlencode(params)
    card_data = fetch_all_pages(initial_url)
    valid_prints_raw = [p for p in card_data if p.get('image_status') in ['highres_scan', 'lowres'] or 'card_faces' in p]

    if not valid_prints_raw and lang != 'en':
        logger.info(f"-> Keine Drucke in '{lang}' gefunden. Wechsle zu EN.")
        lang = 'en'
        params['q'] = f'{search_query} lang:en'
        initial_url = "https://api.scryfall.com/cards/search?" + requests.compat.urlencode(params)
        card_data = fetch_all_pages(initial_url)
        valid_prints_raw = [p for p in card_data if p.get('image_status') in ['highres_scan', 'lowres'] or 'card_faces' in p]

    if not valid_prints_raw:
        return None, f"Karte '{card_name}' nicht gefunden"

    for p in valid_prints_raw:
        is_highres = p.get('image_status') == 'highres_scan'
        p['quality'] = 'H' if is_highres else 'L'
        p['en_highres_fallback'] = None

    if not filter_by_artwork:
        logger.info(f"-> {len(valid_prints_raw)} Drucke gefunden (ohne Artwork-Filter).")
        return valid_prints_raw, None

    unique_prints_by_artwork = {}
    for print_ in reversed(valid_prints_raw):
        illustration_id = (print_.get('card_faces', [{}])[0].get('illustration_id') or print_.get('illustration_id'))
        if not illustration_id: continue

        is_highres = print_['quality'] == 'H'

        if lang != 'en':
            try:
                res_en = requests.get(f"https://api.scryfall.com/cards/{print_['set']}/{print_['collector_number']}/en")
                time.sleep(0.1)
                if res_en.status_code == 200:
                    data_en = res_en.json()
                    if data_en.get('image_status') == 'highres_scan':
                        print_['en_highres_fallback'] = data_en
            except requests.exceptions.RequestException:
                pass

        unique_prints_by_artwork[illustration_id] = print_

    final_prints = list(unique_prints_by_artwork.values())
    logger.info(f"-> {len(final_prints)} einzigartige Artworks für '{card_name}' gefunden.")
    return final_prints, None


def get_image_by_id(card_id):
    """
    Lädt eine Karte per ID herunter.
    Low-Res-Karten werden automatisch mit Real-ESRGAN upscaled,
    um gestochen scharfe Drucke zu gewährleisten.
    """
    metadata_path = os.path.join(CARDS_DIR, f"{card_id}.json")
    try:
        card_data = None
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                card_data = json.load(f)
        else:
            api_url = f"https://api.scryfall.com/cards/{card_id}?format=json"
            for attempt in range(3):
                try:
                    response = requests.get(api_url, timeout=15)
                    response.raise_for_status()
                    card_data = response.json()
                    time.sleep(0.1)
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(card_data, f)
                    break
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Download-Versuch {attempt + 1}/3 für ID {card_id} fehlgeschlagen: {e}")
                    if attempt < 2:
                        time.sleep(3)
                    else:
                        raise e

        if not card_data:
            return None, None, f"Keine Kartendaten für ID {card_id} nach 3 Versuchen."

        image_uris = []
        if 'card_faces' in card_data and 'image_uris' in card_data['card_faces'][0]:
            image_uris.extend(
                face['image_uris'].get('png', face['image_uris'].get('large'))
                for face in card_data['card_faces']
            )
        elif 'image_uris' in card_data:
            image_uris.append(card_data['image_uris'].get('png', card_data['image_uris'].get('large')))
        else:
            return None, None, f"Keine Bild-URIs für ID {card_id} gefunden."

        is_lowres = card_data.get('image_status') == 'lowres'

        downloaded_paths = []
        for i, url in enumerate(image_uris):
            suffix = f"_face_{i}" if len(image_uris) > 1 else ""
            raw_path      = os.path.join(CARDS_DIR, f"{card_id}{suffix}.jpg")
            upscaled_path = os.path.join(CARDS_DIR, f"{card_id}{suffix}_upscaled.jpg")

            # --- Bild herunterladen (falls noch nicht vorhanden) ---
            if not os.path.exists(raw_path):
                img_res = requests.get(url, timeout=30)
                img_res.raise_for_status()
                img = Image.open(io.BytesIO(img_res.content))
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
                    background.paste(img, img.getchannel('A'))
                    img = background
                img.save(raw_path, 'JPEG', quality=95)

            # --- Upscaling für Low-Res-Karten ---
            # High-Res-Karten (745×1040 PNG) sind bereits druckoptimal → kein Upscaling nötig
            # Low-Res-Karten (~488×680) werden auf ~976×1360 upscaled → 300 DPI Druckqualität
            final_path = raw_path
            if is_lowres and UPSCALE_ENABLED:
                if not os.path.exists(upscaled_path):
                    logger.info(f"Low-Res erkannt ({card_data.get('name', card_id)}) – starte Upscaling...")
                    success = upscale_image(raw_path, upscaled_path)
                    if success:
                        final_path = upscaled_path
                else:
                    final_path = upscaled_path  # Cache: bereits upscaled

            downloaded_paths.append(final_path)

        result = downloaded_paths if len(downloaded_paths) > 1 else downloaded_paths[0]
        return result, card_data, None

    except Exception as e:
        return None, None, f"Fehler beim Download für ID {card_id}: {e}"


def process_card_back(source_path, scaling_method='fit'):
    try:
        img = Image.open(source_path)
    except Exception as e:
        return None, f"Fehler beim Öffnen des Bildes: {e}"
    target_size = (744, 1039)
    output_path = os.path.join(UPLOADS_DIR, f"processed_{uuid.uuid4().hex}.jpg")
    if img.mode in ('RGBA', 'LA'):
        background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
        background.paste(img, img.getchannel('A'))
        img = background
    if scaling_method == 'fit':
        processed_img = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
    else:
        processed_img = img.resize(target_size, Image.Resampling.LANCZOS)
    processed_img.save(output_path, 'JPEG', quality=95)
    return output_path, None


def create_blank_image():
    blank_path = os.path.join(UPLOADS_DIR, "blank_card.jpg")
    if not os.path.exists(blank_path):
        img = Image.new('RGB', (744, 1039), (255, 255, 255))
        img.save(blank_path, 'JPEG')
    return blank_path


def create_pdf_from_images(image_list, output_path, cols=3, rows=3, mirror_layout=False):
    if not image_list: return
    A4_WIDTH, A4_HEIGHT   = 210, 297
    CARD_WIDTH_MM, CARD_HEIGHT_MM = 63, 88
    MARGIN_X = (A4_WIDTH  - cols * CARD_WIDTH_MM)  / 2
    MARGIN_Y = (A4_HEIGHT - rows * CARD_HEIGHT_MM) / 2
    if MARGIN_X < 0 or MARGIN_Y < 0:
        MARGIN_X, MARGIN_Y = 1, 1

    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(False)
    pdf.set_margins(0, 0, 0)
    page_size = cols * rows
    for i, path in enumerate(image_list):
        if i % page_size == 0: pdf.add_page()
        col_index = i % cols
        row_index = (i % page_size) // cols
        final_col_index = cols - 1 - col_index if mirror_layout else col_index
        x = MARGIN_X + final_col_index * CARD_WIDTH_MM
        y = MARGIN_Y + row_index * CARD_HEIGHT_MM
        pdf.image(path, x=x, y=y, w=CARD_WIDTH_MM, h=CARD_HEIGHT_MM)
    pdf.output(output_path)
    logger.info(f"PDF erstellt: {output_path}")


def create_duplex_pdf(front_images, back_images, output_path, cols=3, rows=3):
    if not front_images or not back_images or len(front_images) != len(back_images):
        logger.error("Fehler: Listen der Vorder- und Rückseiten leer oder ungleich lang.")
        return

    A4_WIDTH, A4_HEIGHT   = 210, 297
    CARD_WIDTH_MM, CARD_HEIGHT_MM = 65.5, 90.9
    MARGIN_X = (A4_WIDTH  - cols * CARD_WIDTH_MM)  / 2
    MARGIN_Y = (A4_HEIGHT - rows * CARD_HEIGHT_MM) / 2
    if MARGIN_X < 0 or MARGIN_Y < 0:
        MARGIN_X, MARGIN_Y = 1, 1

    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(False)
    pdf.set_margins(0, 0, 0)
    page_size  = cols * rows
    num_pages  = (len(front_images) + page_size - 1) // page_size

    for page_index in range(num_pages):
        pdf.add_page()
        front_chunk = front_images[page_index * page_size : (page_index + 1) * page_size]
        for i, path in enumerate(front_chunk):
            col_index = i % cols
            row_index = (i % page_size) // cols
            x = MARGIN_X + col_index * CARD_WIDTH_MM
            y = MARGIN_Y + row_index * CARD_HEIGHT_MM
            pdf.image(path, x=x, y=y, w=CARD_WIDTH_MM, h=CARD_HEIGHT_MM)

        pdf.add_page()
        back_chunk = back_images[page_index * page_size : (page_index + 1) * page_size]
        for i, path in enumerate(back_chunk):
            col_index = i % cols
            row_index = (i % page_size) // cols
            final_col_index = cols - 1 - col_index
            x = MARGIN_X + final_col_index * CARD_WIDTH_MM
            y = MARGIN_Y + row_index * CARD_HEIGHT_MM
            pdf.image(path, x=x, y=y, w=CARD_WIDTH_MM, h=CARD_HEIGHT_MM)

    pdf.output(output_path)
    logger.info(f"Duplex-PDF erstellt: {output_path}")


# ===========================================================================
# HTML-VORLAGEN (identisch zum Original)
# ===========================================================================

HOME_TEMPLATE = """<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>MTG Proxy Generator - Schritt 1</title><style>body{font-family:sans-serif;background-color:#333;color:#eee;margin:2rem;line-height:1.6}.container{max-width:800px;margin:auto}textarea,input,select{width:100%;padding:12px;margin-top:5px;margin-bottom:15px;background-color:#444;color:#eee;border:1px solid #666;border-radius:5px;box-sizing:border-box}input[type="submit"]{background-color:#007bff;color:white;font-weight:bold;cursor:pointer;font-size:1.1em;transition:background-color .2s}input[type="submit"]:hover{background-color:#0056b3}label{font-weight:bold}.error{color:#ff8a8a;background-color:#5e3333;padding:10px;border-radius:5px}.info{background-color:#2a3f50;padding:10px;border-radius:5px;border-left:5px solid #007bff;margin-bottom:15px}.upscale-badge{background-color:#1a3a1a;border-left:5px solid #28a745;padding:10px;border-radius:5px;margin-bottom:15px}</style></head><body><div class="container"><h1>MTG Proxy Generator</h1><p><b>Schritt 1:</b> Deckliste, Sprache und Dateiname festlegen.</p>{% if error %}<p class="error">{{ error }}</p>{% endif %}
<div class="upscale-badge"><p><b>🔬 KI-Upscaling aktiv (Real-ESRGAN x2)</b><br>Low-Res Karten werden automatisch auf ~976×1360 px hochskaliert – gestochen scharf im PDF.</p></div>
<div class="info"><p><b>Tipp:</b> Sie können eine bestimmte Edition angeben, indem Sie den 3- bis 5-stelligen Editions-Code in Klammern hinter den Kartennamen schreiben.</p><p><b>Beispiele:</b><br>4 Sol Ring (CM2)<br>1 Counterspell (EMA)<br>10 Island (UNF)</p><p>Wenn keine Edition angegeben wird, werden alle verfügbaren Versionen der Karte zur Auswahl angezeigt.</p></div>
<form id="decklistForm"><label for="decklist">Deckliste hier einfügen:</label><textarea name="decklist" id="decklist" rows="15" placeholder="4 Sol Ring&#10;1 Command Tower&#10;..."></textarea><label for="lang">Sprache der Karten:</label><select name="lang" id="lang">{% for code, name in languages.items() %}<option value="{{ code }}" {% if code == 'de' %}selected{% endif %}>{{ name }}</option>{% endfor %}</select><label for="filename">Gewünschter PDF-Dateiname (ohne .pdf):</label><input type="text" name="filename" id="filename" value="deck_proxies"><input type="submit" value="Editionen suchen →"></form></div>
<script>
document.getElementById('decklistForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const submitButton = this.querySelector('input[type="submit"]');
    submitButton.value = 'Starte Suche...';
    submitButton.disabled = true;
    const formData = new FormData(this);
    fetch("{{ url_for('start_card_search') }}", {method: 'POST', body: formData})
    .then(response => response.json())
    .then(data => {
        if (data.task_id) {
            window.location.href = `/loading-search/${data.task_id}`;
        } else {
            alert('Fehler beim Starten der Suche: ' + (data.error || 'Unbekanntes Problem'));
            submitButton.value = 'Editionen suchen →';
            submitButton.disabled = false;
        }
    })
    .catch(err => {
        alert('Netzwerkfehler: ' + err);
        submitButton.value = 'Editionen suchen →';
        submitButton.disabled = false;
    });
});
</script>
</body></html>"""

LOADING_TEMPLATE = """<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Suche Karten...</title><style>body{font-family:monospace;background-color:#333;color:#eee;display:flex;justify-content:center;align-items:center;height:100vh;margin:0}.container{width:80%;max-width:600px;text-align:center}.progress-bar-container{border:2px solid #eee;padding:3px;margin:20px 0}.progress-bar{background-color:#007bff;width:0%;height:25px;transition:width .2s ease-in-out;text-align:right;line-height:25px;color:#000;font-weight:bold;overflow:hidden}.status-text{margin-top:15px;height:2em;font-size:1.1em}</style></head><body><div class="container"><h1>Suche nach Karten-Editionen...</h1><p>Dies kann einen Moment dauern.</p><div class="progress-bar-container"><div id="progressBar" class="progress-bar"></div></div><p id="progressText">[....................] 0%</p><div id="statusText">Initialisiere...</div></div>
<script>
const taskId = "{{ task_id }}";
const progressBar = document.getElementById("progressBar");
const progressText = document.getElementById("progressText");
const statusText = document.getElementById("statusText");
function checkStatus() {
    fetch(`/status/${taskId}`)
    .then(response => response.json())
    .then(data => {
        if (!data || data.status === 'error') { statusText.innerText = `Ein Fehler ist aufgetreten: ${data.message || 'Unbekannter Fehler'}`; clearInterval(interval); return; }
        const percent = data.progress || 0;
        progressBar.style.width = percent + "%";
        const filledChars = Math.round(percent / 5);
        const emptyChars = 20 - filledChars;
        progressText.innerText = `[${'#'.repeat(filledChars)}${'.'.repeat(emptyChars)}] ${Math.round(percent)}%`;
        statusText.innerText = data.message || "";
        if (data.status === 'complete') { clearInterval(interval); statusText.innerText = "Suche abgeschlossen! Leite zur Auswahl weiter..."; window.location.href = `/selection/${taskId}`; }
    })
    .catch(err => { statusText.innerText = "Verbindung zum Server verloren."; clearInterval(interval); });
}
const interval = setInterval(checkStatus, 1500);
checkStatus();
</script></body></html>"""

SELECTION_TEMPLATE = """<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>MTG Proxy Generator - Schritt 2 & 3</title><style>body{font-family:sans-serif;background-color:#333;color:#eee;margin:0;padding:2rem}.container{max-width:1200px;margin:auto}.card-group,.back-selection-group{background-color:#444;padding:20px;border-radius:8px;margin-bottom:20px}.group-title{border-bottom:2px solid #666;padding-bottom:10px;margin-bottom:15px;font-size:1.5em}div.option-group p{margin-top:0}.option-group label{display:block;margin-bottom:8px}.print-options{display:flex;flex-wrap:wrap;gap:20px}.print-option{cursor:pointer;position:relative;display:flex;flex-direction:column;align-items:center}.print-option input[type="radio"]{display:none}.print-option img{width:150px;border-radius:7px;border:3px solid transparent;transition:all .2s}.print-option .no-image{width:150px;height:209px;border-radius:7px;border:3px solid #666;background-color:#3a3a3a;display:flex;align-items:center;justify-content:center;text-align:center;font-size:.9em;color:#aaa;padding:5px;box-sizing:border-box}input[type="radio"]:checked+.card-display-wrapper{border-color:#007bff;box-shadow:0 0 15px #007bff;border-radius:10px}input[type="radio"][value="do_not_print"]:checked ~ .no-image{border-color:#dc3545;box-shadow:0 0 15px #dc3545}.card-display-wrapper{border:3px solid transparent;padding:5px;display:flex;flex-direction:column;align-items:center;border-radius:10px}.card-display-wrapper.double-face{flex-direction:row;gap:5px}.card-display-wrapper.double-face img{width:120px}.print-option .quality-badge{position:absolute;top:10px;right:10px;padding:2px 5px;font-size:.8em;font-weight:bold;border-radius:4px;z-index:10}.quality-H{background-color:#28a745;color:white}.quality-L{background-color:#dc3545;color:white}.print-option p{text-align:center;margin:5px 0 0;font-size:.9em;color:#ccc}.submit-button{display:block;width:100%;padding:15px;margin-top:20px;background-color:#28a745;color:white;font-weight:bold;cursor:pointer;font-size:1.2em;border:none;border-radius:5px;transition:background-color .2s}.submit-button:hover{background-color:#218838}.fallback-option{border-left:2px dotted #007bff;padding-left:15px}.preview-modal{display:none;position:fixed;z-index:1000;left:0;top:0;width:100%;height:100%;background-color:rgba(0,0,0,.85);justify-content:center;align-items:center}.preview-modal img{max-width:90%;max-height:90%;border-radius:15px}.preview-modal .close-btn{position:absolute;top:20px;right:35px;color:#f1f1f1;font-size:40px;font-weight:bold;cursor:pointer}.upload-section{flex-grow:1}.load-more-btn{background-color:#5a6268;color:white;border:none;padding:10px 15px;border-radius:5px;cursor:pointer;margin-top:10px;transition:background-color .2s}.load-more-btn:hover{background-color:#4a4f54}.load-more-btn:disabled{background-color:#333;cursor:not-allowed}</style></head><body><div class="container">
<form action="{{ url_for('generate_pdf', task_id=task_id) }}" method="post" enctype="multipart/form-data"><h1 class="group-title">Schritt 2: Editionen & Optionen</h1>{% if errors %}<p style="color:red;">Folgende Karten wurden nicht gefunden: {{ errors|join(', ') }}</p>{% endif %}<p>Wähle für jede Karte das gewünschte Artwork aus. (H) = High-Res, (L) = Low-Res → wird automatisch mit Real-ESRGAN upscaled.<br><strong>Tipp: Rechtsklick auf ein Bild für eine hochauflösende Vorschau!</strong></p>
<div class="card-group option-group"><h2 class="group-title" style="font-size:1.3em;border-bottom-style:dotted">Optionen für doppelseitige Karten (DFCs)</h2><p>Wie sollen DFCs im PDF platziert werden?</p><label><input type="radio" name="dfc_handling" value="side_by_side" checked> <strong>Nebeneinander</strong></label><label><input type="radio" name="dfc_handling" value="true_backside"> <strong>Echte Rückseiten (Duplex)</strong></label><label><input type="radio" name="dfc_handling" value="dfc_only_backside"> <strong>Nur DFC-Rückseiten (Duplex)</strong></label></div>
{% for card_name, data in cards.items() %}<div class="card-group"><h2 class="card-title">{{ data.count }}x {{ data.original_name }}{% if data.set_code %} ({{ data.set_code.upper() }}){% endif %}</h2><div class="print-options">
<label class="print-option"><input type="radio" name="{{ data.original_name|replace(' ', '_') }}_{{ data.set_code or '' }}" value="do_not_print"><div class="card-display-wrapper"><div class="no-image" style="border-color:#dc3545; color:#ff8a8a;">Nicht drucken</div></div><p>Diese Karte überspringen</p></label>
{% for print in data.printings %}<label class="print-option"><input type="radio" name="{{ data.original_name|replace(' ', '_') }}_{{ data.set_code or '' }}" value="{{ print.id }}" {% if loop.first %}checked{% endif %}><div class="card-display-wrapper {% if 'card_faces' in print %}double-face{% endif %}">{% if 'card_faces' in print and print.card_faces[0].get('image_uris') %}<img src="{{ print.card_faces[0].image_uris.small }}" data-large-url="{{ print.card_faces[0].image_uris.png or print.card_faces[0].image_uris.large }}" alt="{{ print.set_name }} - Front"><img src="{{ print.card_faces[1].image_uris.small }}" data-large-url="{{ print.card_faces[1].image_uris.png or print.card_faces[1].image_uris.large }}" alt="{{ print.set_name }} - Back">{% elif print.image_uris %}<img src="{{ print.image_uris.small }}" data-large-url="{{ print.image_uris.png or print.image_uris.large }}" alt="{{ print.set_name }}">{% else %}<div class="no-image">Bild nicht verfügbar</div>{% endif %}</div><span class="quality-badge quality-{{ print.quality }}">{{ print.quality }}</span><p>{{ print.set_name }} ({{ print.released_at[:4] }})</p></label>{% if print.en_highres_fallback %}<label class="print-option fallback-option"><input type="radio" name="{{ data.original_name|replace(' ', '_') }}_{{ data.set_code or '' }}" value="{{ print.en_highres_fallback.id }}"><div class="card-display-wrapper {% if 'card_faces' in print.en_highres_fallback %}double-face{% endif %}">{% if 'card_faces' in print.en_highres_fallback and print.en_highres_fallback.card_faces[0].get('image_uris') %}<img src="{{ print.en_highres_fallback.card_faces[0].image_uris.small }}" data-large-url="{{ print.en_highres_fallback.card_faces[0].image_uris.png }}" alt="{{ print.set_name }} (EN) - Front"><img src="{{ print.en_highres_fallback.card_faces[1].image_uris.small }}" data-large-url="{{ print.en_highres_fallback.card_faces[1].image_uris.png }}" alt="{{ print.set_name }} (EN) - Back">{% elif print.en_highres_fallback.image_uris %}<img src="{{ print.en_highres_fallback.image_uris.small }}" data-large-url="{{ print.en_highres_fallback.image_uris.png }}" alt="{{ print.set_name }} (EN)">{% else %}<div class="no-image">Bild nicht verfügbar</div>{% endif %}</div><span class="quality-badge quality-H">H</span><p>{{ print.set_name }} (EN - High-Res)</p></label>{% endif %}{% endfor %}</div>
<button type="button" class="load-more-btn" data-card-name="{{ data.original_name }}" data-input-name="{{ data.original_name|replace(' ', '_') }}_{{ data.set_code or '' }}">Mehr laden (Englisch)</button>
</div>{% endfor %}
<div class="back-selection-group"><h1 class="group-title">Schritt 3: Kartenrücken auswählen</h1><label><input type="radio" name="back_choice_type" value="none" checked> Keine allgemeine Rückseite</label><hr style="border-color:#666;margin:15px 0"><label><input type="radio" name="back_choice_type" value="standard"> Standard-Rückseite:</label><div class="print-options back-options">{% for back_img in standard_backs %}<label class="print-option"><input type="radio" name="standard_back" value="{{ back_img }}"><div class="card-display-wrapper"><img src="{{ url_for('serve_card_back', filename=back_img) }}"></div></label>{% endfor %}</div><hr style="border-color:#666;margin:15px 0"><label><input type="radio" name="back_choice_type" value="custom"> Eigene Rückseite:</label><div class="upload-section"><input type="file" name="custom_back_file"><p>Anpassung: <label><input type="radio" name="scaling_method" value="fit" checked> Zuschneiden</label> <label><input type="radio" name="scaling_method" value="stretch"> Strecken</label></p></div></div>
<input type="submit" value="Fertig! PDFs jetzt erstellen" class="submit-button"></form></div>
<div id="previewModal" class="preview-modal"><span class="close-btn">&times;</span><img id="previewImage"></div>
<script>
document.addEventListener('contextmenu', function(e) {
    const img = e.target.closest('img[data-large-url]');
    if (img) { e.preventDefault(); document.getElementById('previewImage').src = img.dataset.largeUrl; document.getElementById('previewModal').style.display = 'flex'; }
});
document.querySelector('.preview-modal .close-btn').addEventListener('click', function() { document.getElementById('previewModal').style.display = 'none'; });
document.getElementById('previewModal').addEventListener('click', function(e) { if (e.target.id === 'previewModal') this.style.display = 'none'; });
document.querySelectorAll('.load-more-btn').forEach(button => {
    button.addEventListener('click', function() {
        const cardName = this.dataset.cardName;
        const inputName = this.dataset.inputName;
        const optionsContainer = this.closest('.card-group').querySelector('.print-options');
        this.textContent = 'Lade...';
        this.disabled = true;
        fetch("{{ url_for('load_more_prints') }}", {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ card_name: cardName })})
        .then(response => { if (!response.ok) throw new Error('Network response was not ok'); return response.json(); })
        .then(data => {
            if (data.printings) {
                const existingIds = new Set(Array.from(optionsContainer.querySelectorAll('input[type="radio"]')).map(input => input.value));
                data.printings.forEach(print => {
                    if (existingIds.has(print.id)) return;
                    const isDfc = print.card_faces && print.card_faces.length > 1;
                    const frontImg = isDfc ? print.card_faces[0].image_uris : print.image_uris;
                    const backImg  = isDfc ? print.card_faces[1].image_uris : null;
                    let imagesHtml = '<div class="no-image">Bild nicht verfügbar</div>';
                    if (frontImg) {
                        imagesHtml = `<img src="${frontImg.small}" data-large-url="${frontImg.png || frontImg.large}" alt="${print.set_name} - Front">`;
                        if (backImg) imagesHtml += `<img src="${backImg.small}" data-large-url="${backImg.png || backImg.large}" alt="${print.set_name} - Back">`;
                    }
                    const newOptionHtml = `<label class="print-option"><input type="radio" name="${inputName}" value="${print.id}"><div class="card-display-wrapper ${isDfc ? 'double-face' : ''}">${imagesHtml}</div><span class="quality-badge quality-${print.quality}">${print.quality}</span><p>${print.set_name} (${print.released_at.substring(0, 4)}) (EN)</p></label>`;
                    optionsContainer.insertAdjacentHTML('beforeend', newOptionHtml);
                    existingIds.add(print.id);
                });
                this.textContent = 'Alle englischen Versionen geladen';
            } else { this.textContent = data.error || 'Laden fehlgeschlagen'; }
        })
        .catch(err => { console.error('Fehler:', err); this.textContent = 'Fehler!'; });
    });
});
</script></body></html>"""

RESULT_TEMPLATE = """<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>MTG Proxy Generator - Fertig!</title><style>body{font-family:sans-serif;background-color:#333;color:#eee;margin:2rem;text-align:center}.container{max-width:800px;margin:auto;background-color:#444;padding:30px;border-radius:8px}a.download-btn{display:inline-block;background-color:#28a745;color:white;padding:15px 30px;text-decoration:none;font-size:1.2em;border-radius:5px;margin:10px;transition:background-color .2s}a.download-btn:hover{background-color:#218838}a.download-btn.duplex{background-color:#007bff}a.download-btn.duplex:hover{background-color:#0056b3}.error{color:#ff8a8a}ul{list-style:none;padding:0}.download-section{border-bottom:1px solid #666;padding-bottom:15px;margin-bottom:15px}</style></head><body><div class="container"><h1>PDF-Erstellung abgeschlossen</h1>
{% if pdf_duplex_path %}<div class="download-section"><p><strong>Für den beidseitigen Druck (empfohlen):</strong><br>Diese Datei enthält Vorder- und Rückseiten. Drucke sie mit der Option "Beidseitiger Druck, an der langen Kante spiegeln".</p><a href="{{ url_for('download_file', filename=filename_duplex) }}" class="download-btn duplex">"{{ filename_duplex }}" herunterladen</a></div>{% endif %}
{% if pdf_front_path %}<div class="download-section"><p><strong>Nur Vorderseiten:</strong></p><a href="{{ url_for('download_file', filename=filename_front) }}" class="download-btn">"{{ filename_front }}" herunterladen</a></div>{% endif %}
{% if pdf_back_path %}<div class="download-section"><p><strong>Nur Rückseiten (für manuellen Duplex-Druck):</strong><br>Die Seiten sind bereits für den Druck auf die Rückseite der Vorderseiten-Bögen gespiegelt.</p><a href="{{ url_for('download_file', filename=filename_back) }}" class="download-btn">"{{ filename_back }}" herunterladen</a></div>{% endif %}
{% if not pdf_front_path and not pdf_back_path and not pdf_duplex_path %}<p class="error">Es konnten keine PDFs erstellt werden, da keine Karten ausgewählt oder gefunden wurden.</p>{% endif %}
{% if errors %}<h3>Einige Karten konnten nicht geladen werden:</h3><ul>{% for error in errors %}<li>{{ error }}</li>{% endfor %}</ul>{% endif %}
<p style="margin-top:30px;"><a href="{{ url_for('home') }}">Neue PDF erstellen</a></p></div></body></html>"""


# ===========================================================================
# WEB-ROUTEN
# ===========================================================================

@app.route('/')
def home():
    return render_template_string(HOME_TEMPLATE, languages=LANGUAGES)


def _run_card_search(task_id, card_requests, lang, filename):
    """Sucht im Hintergrund nach Karten."""
    try:
        tasks[task_id] = {'status': 'processing', 'progress': 0, 'message': 'Starte Kartensuche...'}
        cards_for_selection = {}
        error_messages = []
        total_cards = len(card_requests)

        for i, req in enumerate(card_requests):
            card_name = req['name']
            set_code  = req['set']
            count     = req['count']
            tasks[task_id]['message'] = (
                f"Suche nach: {card_name}" +
                (f" ({set_code.upper()})" if set_code else "") +
                f" ({i+1}/{total_cards})"
            )

            if set_code:
                printings, error = find_specific_card_printing(card_name, set_code, lang)
            else:
                printings, error = find_card_printings(card_name, lang)

            if error:
                error_messages.append(error)
            else:
                unique_key = f"{card_name}_{set_code}" if set_code else card_name
                cards_for_selection[unique_key] = {
                    'count': count,
                    'printings': printings,
                    'set_code': set_code,
                    'original_name': card_name
                }

            tasks[task_id]['progress'] = (i + 1) / total_cards * 100
            time.sleep(0.1)

        tasks[task_id]['result_data'] = {
            'cards_for_selection': cards_for_selection,
            'card_requests':       card_requests,
            'filename_base':       filename,
            'error_messages':      error_messages,
        }
        tasks[task_id]['status'] = 'complete'

    except Exception as e:
        tasks[task_id]['status']  = 'error'
        tasks[task_id]['message'] = f"Ein schwerwiegender Fehler ist aufgetreten: {e}"
        logger.exception(f"Schwerwiegender Fehler in Task {task_id}")


@app.route('/start-card-search', methods=['POST'])
def start_card_search():
    decklist_text = request.form.get('decklist')
    if not decklist_text:
        return jsonify({'error': 'Die Deckliste darf nicht leer sein.'})

    line_regex    = re.compile(r"^\s*(\d*)\s*([^(]+?)\s*(?:\(([^)]+)\))?\s*$")
    card_requests = []
    lines = [line.strip() for line in decklist_text.strip().split('\n') if line.strip()]

    for line in lines:
        match = line_regex.match(line)
        if match:
            count, name, set_code = match.groups()
            card_requests.append({
                'count': int(count) if count.isdigit() else 1,
                'name':  name.strip(),
                'set':   set_code.strip().lower() if set_code else None,
            })

    if not card_requests:
        return jsonify({'error': "Keine gültigen Karten gefunden."})

    task_id  = uuid.uuid4().hex
    lang     = request.form.get('lang', 'de')
    filename = request.form.get('filename', 'deck_proxies').replace('.pdf', '')

    thread = threading.Thread(
        target=_run_card_search,
        args=(task_id, card_requests, lang, filename),
        daemon=True,
    )
    thread.start()
    return jsonify({'task_id': task_id})


@app.route('/loading-search/<task_id>')
def loading_search_page(task_id):
    return render_template_string(LOADING_TEMPLATE, task_id=task_id)


@app.route('/status/<task_id>')
def task_status(task_id):
    task = tasks.get(task_id, {})
    return jsonify({k: v for k, v in task.items() if k != 'result_data'})


@app.route('/selection/<task_id>')
def show_selection_page(task_id):
    task = tasks.get(task_id)
    if not task or task.get('status') != 'complete':
        if task and task.get('status') == 'processing':
            return redirect(url_for('loading_search_page', task_id=task_id))
        return redirect(url_for('home'))

    result_data        = task.get('result_data', {})
    cards_for_selection = result_data.get('cards_for_selection', {})
    sorted_cards       = sorted(cards_for_selection.items(), key=lambda item: item[1]['original_name'])
    standard_backs     = (
        [f for f in os.listdir(CARD_BACKS_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        if os.path.exists(CARD_BACKS_DIR) else []
    )
    return render_template_string(
        SELECTION_TEMPLATE,
        cards=dict(sorted_cards),
        errors=result_data.get('error_messages'),
        standard_backs=standard_backs,
        task_id=task_id,
    )


@app.route('/load-more', methods=['POST'])
def load_more_prints():
    card_name = request.json.get('card_name')
    if not card_name:
        return jsonify({'error': 'Kartenname fehlt.'}), 400
    printings, error = find_card_printings(card_name, lang='en', filter_by_artwork=False)
    if error:
        return jsonify({'error': error}), 404
    return jsonify({'printings': printings})


@app.route('/generate/<task_id>', methods=['POST'])
def generate_pdf(task_id):
    task = tasks.get(task_id)
    if not task:
        return redirect(url_for('home'))

    selections    = request.form
    result_data   = task.get('result_data', {})
    card_requests = result_data.get('card_requests', [])
    filename_base = result_data.get('filename_base', 'proxies')
    failed_cards  = result_data.get('error_messages', [])

    final_card_ids = []
    for req in card_requests:
        form_field_name = f"{req['name'].replace(' ', '_')}_{req['set'] or ''}"
        selected_id = selections.get(form_field_name)
        if selected_id and selected_id != 'do_not_print':
            final_card_ids.extend([selected_id] * int(req['count']))

    id_to_path_map, id_to_metadata_map = {}, {}
    for card_id in sorted(set(final_card_ids)):
        paths, metadata, error = get_image_by_id(card_id)
        if paths and metadata:
            id_to_path_map[card_id]     = paths
            id_to_metadata_map[card_id] = metadata
        else:
            failed_cards.append(f"Download fehlgeschlagen (ID: {card_id}): {error}")

    card_back_path = None
    if request.form.get('back_choice_type') == 'custom':
        if custom_back_file := request.files.get('custom_back_file'):
            if custom_back_file.filename != '':
                temp_path = os.path.join(UPLOADS_DIR, f"{uuid.uuid4().hex}_{custom_back_file.filename}")
                custom_back_file.save(temp_path)
                processed_path, error = process_card_back(temp_path, request.form.get('scaling_method', 'fit'))
                if error: failed_cards.append(f"Verarbeitung der Rückseite fehlgeschlagen: {error}")
                else:     card_back_path = processed_path
    elif request.form.get('back_choice_type') == 'standard':
        if back_filename := request.form.get('standard_back'):
            card_back_path = os.path.join(CARD_BACKS_DIR, back_filename)

    dfc_handling    = selections.get('dfc_handling', 'side_by_side')
    blank_image_path = create_blank_image() if dfc_handling == 'dfc_only_backside' else None
    needs_back_pdf  = (card_back_path is not None) or (dfc_handling in ['true_backside', 'dfc_only_backside'])

    all_front_image_paths, all_back_image_paths = [], []

    for card_id in final_card_ids:
        path_or_paths = id_to_path_map.get(card_id)
        metadata      = id_to_metadata_map.get(card_id)
        if not path_or_paths or not metadata: continue

        is_dfc = 'card_faces' in metadata and isinstance(path_or_paths, list) and len(path_or_paths) > 1
        front_paths_to_add = []
        if is_dfc and dfc_handling in ['true_backside', 'dfc_only_backside']:
            front_paths_to_add.append(path_or_paths[0])
        else:
            front_paths_to_add.extend(path_or_paths if isinstance(path_or_paths, list) else [path_or_paths])
        all_front_image_paths.extend(front_paths_to_add)

        if needs_back_pdf:
            if is_dfc and dfc_handling in ['true_backside', 'dfc_only_backside']:
                all_back_image_paths.append(path_or_paths[1])
            else:
                back_to_use = card_back_path if card_back_path else blank_image_path
                if back_to_use:
                    all_back_image_paths.extend([back_to_use] * len(front_paths_to_add))

    pdf_front_path = pdf_back_path = pdf_duplex_path = None
    filename_front  = f"{filename_base}_front.pdf"
    filename_back   = f"{filename_base}_back.pdf"
    filename_duplex = f"{filename_base}_duplex.pdf"

    if all_front_image_paths:
        pdf_front_path = os.path.join(OUTPUT_DIR, filename_front)
        create_pdf_from_images(all_front_image_paths, pdf_front_path, mirror_layout=False)

    if all_back_image_paths:
        if len(all_front_image_paths) == len(all_back_image_paths):
            pdf_back_path = os.path.join(OUTPUT_DIR, filename_back)
            create_pdf_from_images(all_back_image_paths, pdf_back_path, mirror_layout=True)
            pdf_duplex_path = os.path.join(OUTPUT_DIR, filename_duplex)
            create_duplex_pdf(all_front_image_paths, all_back_image_paths, pdf_duplex_path)
        else:
            failed_cards.append("Fehler: Anzahl Vorder-/Rückseiten stimmt nicht. Duplex-PDF nicht erstellt.")

    if task_id in tasks:
        del tasks[task_id]
    session.clear()

    return render_template_string(
        RESULT_TEMPLATE,
        pdf_front_path=pdf_front_path,   pdf_back_path=pdf_back_path,   pdf_duplex_path=pdf_duplex_path,
        filename_front=filename_front if pdf_front_path else None,
        filename_back=filename_back   if pdf_back_path  else None,
        filename_duplex=filename_duplex if pdf_duplex_path else None,
        errors=list(set(failed_cards)),
    )


@app.route('/backs/<path:filename>')
def serve_card_back(filename):
    return send_from_directory(CARD_BACKS_DIR, filename)


@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


# ===========================================================================
# APP-START
# ===========================================================================
if __name__ == '__main__':
    # Verzeichnisse anlegen
    for dir_path in [CARDS_DIR, OUTPUT_DIR, CARD_BACKS_DIR, UPLOADS_DIR, MODELS_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    # Real-ESRGAN-Modell im Hintergrund vorladen
    threading.Thread(target=_preload_upscaler_background, daemon=True).start()

    # Lokaler Dev-Server
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
else:
    # Gunicorn-Start (Render Produktion)
    for dir_path in [CARDS_DIR, OUTPUT_DIR, CARD_BACKS_DIR, UPLOADS_DIR, MODELS_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    # Modell im Hintergrund vorladen sobald Gunicorn den Worker startet
    threading.Thread(target=_preload_upscaler_background, daemon=True).start()
