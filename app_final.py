import os
import csv
import glob
import json
import tempfile
import importlib.util
from pathlib import Path
import ast
import sys
import io
import time
from contextlib import contextmanager

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# Stable image-annotation component for manual bounding boxes.
try:
    from streamlit_image_annotation import detection as image_detection
    IMAGE_ANNOTATION_AVAILABLE = True
except ImportError:
    image_detection = None
    IMAGE_ANNOTATION_AVAILABLE = False


# ============================================================
# STREAMLIT DRAWABLE CANVAS COMPATIBILITY
# ============================================================
# streamlit-drawable-canvas 0.9.3 imports:
#
#     streamlit.elements.image.image_to_url
#
# Newer Streamlit versions moved image_to_url to:
#
#     streamlit.elements.lib.image_utils
#
# Without this compatibility shim the manual segmentation
# editor crashes when it tries to display the background image.
try:
    import streamlit.elements.image as _st_image

    # streamlit-drawable-canvas 0.9.3 calls the OLD Streamlit
    # image_to_url API positionally:
    #
    #   image_to_url(image, width, clamp, channels, output_format, image_id)
    #
    # New Streamlit versions moved image_to_url and changed its
    # second argument to a LayoutConfig object.  Simply aliasing
    # the new function is therefore NOT enough: the old integer
    # width gets passed as layout_config, producing:
    #   AttributeError: 'int' object has no attribute 'width'
    #
    # This wrapper converts the old call signature to the new API.
    from streamlit.elements.lib import image_utils as _image_utils
    from streamlit.elements.lib.layout_utils import LayoutConfig

    _new_image_to_url = _image_utils.image_to_url

    def _drawable_canvas_image_to_url(
        image,
        width=None,
        clamp=False,
        channels="RGB",
        output_format="auto",
        image_id=None,
        *args,
        **kwargs,
    ):
        # Support both the old positional API used by
        # streamlit-drawable-canvas and a possible keyword call.
        if "layout_config" in kwargs:
            return _new_image_to_url(
                image,
                layout_config=kwargs["layout_config"],
                clamp=clamp,
                channels=channels,
                output_format=output_format,
                image_id=image_id,
            )

        layout_config = LayoutConfig(width=width)

        return _new_image_to_url(
            image,
            layout_config=layout_config,
            clamp=clamp,
            channels=channels,
            output_format=output_format,
            image_id=image_id,
        )

    _st_image.image_to_url = _drawable_canvas_image_to_url

except Exception:
    # The actual canvas import below will provide the useful
    # error/warning if the installed Streamlit version differs.
    pass


try:
    from streamlit_drawable_canvas import st_canvas
    DRAWABLE_CANVAS_AVAILABLE = True
except ImportError:
    st_canvas = None
    DRAWABLE_CANVAS_AVAILABLE = False


# ============================================================
# NeoLipi - One Click Brahmi Inscription Analyzer
# ============================================================
#
# USER:
#   Upload image -> Analyze
#
# PIPELINE:
#   1. Script classification
#   2. Binarization
#   3. Line/character segmentation
#   4. Character extraction
#   5. Save extracted characters
#   6. Preprocess complete extracted folder
#   7. All-class CNN recognition
#   8. Brahmi class -> Unicode/glyph mapping
#   7. Digital Brahmi text
#   8. CSV class-name -> Kannada mapping
#
# Kannada conversion is performed directly from the recognized class names.
#
# The mapping source is the user's CSV:
#   brahmi_mapping.csv
#
# No technical file paths are shown in the UI.
# ============================================================
st.set_page_config(
    page_title="Stone Inscription OCR System",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mildly zoom out slightly (92%) and add comfortable margin padding
# Mildly zoom out more (85%) and add comfortable margin padding
st.markdown(
    """
    <style>
    .stApp {
        zoom: 85%;
    }
    .main .block-container {
        max-width: 92% !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

ROOT = Path(__file__).resolve().parent

# ============================================================
# STABLE GREEN MULTI-BOX MANUAL EDITOR
# ============================================================
# A tiny local Streamlit component is embedded directly in this
# Python file. This avoids streamlit-drawable-canvas rerun issues
# and avoids the extra red/Transform/Class UI of image annotation.
MANUAL_COMPONENT_DIR = ROOT / ".neolipi_manual_box_component_v24"
MANUAL_COMPONENT_DIR.mkdir(parents=True, exist_ok=True)
MANUAL_COMPONENT_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<script>
// Minimal local Streamlit component protocol shim.
// Kept inline so this component does NOT depend on internet/CDN access.
window.Streamlit = {
  RENDER_EVENT: "streamlit:render",
  setComponentReady: function(){
    window.parent.postMessage({isStreamlitMessage:true,type:"streamlit:componentReady",apiVersion:1}, "*");
  },
  setFrameHeight: function(height){
    window.parent.postMessage({isStreamlitMessage:true,type:"streamlit:setFrameHeight",height:Number(height)||0}, "*");
  },
  setComponentValue: function(value){
    window.parent.postMessage({isStreamlitMessage:true,type:"streamlit:setComponentValue",value:value}, "*");
  },
  events: {
    addEventListener: function(type, callback){
      window.addEventListener("message", function(event){
        const data=event.data||{};
        if(data && data.type===type){ callback({detail:data}); }
      });
    }
  }
};
</script>
<style>
html,body{margin:0;padding:0;background:transparent;font-family:Arial,sans-serif;color:#ddd}
#wrap{width:100%;display:flex;flex-direction:column;align-items:center;gap:7px}
#toolbar{display:flex;gap:6px;align-items:center;flex-wrap:wrap;width:100%}
button{background:#262b35;color:#fff;border:1px solid #596170;border-radius:5px;padding:5px 10px;cursor:pointer}
button:hover{background:#343b48}
#count{font-size:13px;margin-left:auto;color:#aaa}
#stage{position:relative;display:inline-block;max-width:100%;border:1px solid #555;background:#000}
#img{display:block;max-width:100%;height:auto}
#canvas{position:absolute;left:0;top:0;cursor:crosshair}
.hint{font-size:12px;color:#999}
</style>
</head>
<body>
<div id="wrap">
  <div id="toolbar">
    <button id="undo">Undo</button>
    <button id="clear">Clear</button>
    <button id="delete">Delete selected</button>
    <span id="count">Boxes currently drawn: 0</span>
  </div>
  <div id="stage">
    <canvas id="canvas"></canvas>
  </div>
  <div class="hint">Drag a green rectangle around each character. Click and drag a box to move it.</div>
</div>
<script>
(function(){
  const Streamlit = window.Streamlit;
  const canvas=document.getElementById("canvas");
  const ctx=canvas.getContext("2d");
  const countEl=document.getElementById("count");
  const stage=document.getElementById("stage");
  let boxes=[], selected=-1, drawing=false, moving=false;
  let startX=0,startY=0,moveDX=0,moveDY=0,cfg={};
  let sourceImage=null, naturalW=1, naturalH=1, shownW=760, shownH=300;

  function sendValue(){
    Streamlit.setComponentValue(boxes.map(b=>({
      x:Math.round(b.x), y:Math.round(b.y),
      w:Math.round(b.w), h:Math.round(b.h)
    })));
  }

  function fit(){
    const maxW=Number(cfg.width)||760;
    const nw=naturalW||760, nh=naturalH||300;
    shownW=Math.min(maxW,nw);
    shownH=Math.max(1,Math.round(nh*shownW/nw));

    canvas.width=shownW;
    canvas.height=shownH;
    canvas.style.width=shownW+"px";
    canvas.style.height=shownH+"px";
    stage.style.width=shownW+"px";
    stage.style.height=shownH+"px";

    draw();
    Streamlit.setFrameHeight(Math.max(180, shownH+105));
  }

  function draw(){
    ctx.clearRect(0,0,canvas.width,canvas.height);

    // Always paint the actual binary image first.
    ctx.fillStyle="#000";
    ctx.fillRect(0,0,canvas.width,canvas.height);

    if(sourceImage){
      ctx.imageSmoothingEnabled=false;
      ctx.drawImage(sourceImage,0,0,canvas.width,canvas.height);
    }

    const sx=canvas.width/naturalW;
    const sy=canvas.height/naturalH;

    boxes.forEach((b,i)=>{
      const x=b.x*sx,y=b.y*sy,w=b.w*sx,h=b.h*sy;
      ctx.fillStyle="rgba(0,255,0,0.08)";
      ctx.fillRect(x,y,w,h);
      ctx.strokeStyle="#00ff00";
      ctx.lineWidth=i===selected?3:2;
      ctx.strokeRect(x+0.5,y+0.5,w,h);
      ctx.fillStyle="#00ff00";
      ctx.font="bold 13px Arial";
      ctx.fillText(String(i+1),x+3,Math.max(14,y-3));
      if(i===selected){
        ctx.fillRect(x+w-5,y+h-5,10,10);
      }
    });

    countEl.textContent="Boxes currently drawn: "+boxes.length;
  }

  function pos(e){
    const r=canvas.getBoundingClientRect();
    const px=(e.clientX-r.left), py=(e.clientY-r.top);
    return {
      x:px/(r.width||1)*naturalW,
      y:py/(r.height||1)*naturalH
    };
  }

  function hit(p){
    for(let i=boxes.length-1;i>=0;i--){
      const b=boxes[i];
      if(p.x>=b.x&&p.x<=b.x+b.w&&p.y>=b.y&&p.y<=b.y+b.h) return i;
    }
    return -1;
  }

  canvas.addEventListener("pointerdown",e=>{
    e.preventDefault();
    const p=pos(e), h=hit(p);
    if(h>=0){
      selected=h; moving=true;
      moveDX=p.x-boxes[h].x; moveDY=p.y-boxes[h].y;
    }else{
      selected=-1; drawing=true; startX=p.x; startY=p.y;
    }
    canvas.setPointerCapture(e.pointerId); draw();
  });

  canvas.addEventListener("pointermove",e=>{
    if(!drawing&&!moving) return;
    e.preventDefault();
    const p=pos(e);
    if(drawing){
      draw();
      const sx=canvas.width/naturalW;
      const sy=canvas.height/naturalH;
      const x=Math.min(startX,p.x), y=Math.min(startY,p.y);
      const w=Math.abs(p.x-startX), h=Math.abs(p.y-startY);
      ctx.fillStyle="rgba(0,255,0,0.08)";
      ctx.fillRect(x*sx,y*sy,w*sx,h*sy);
      ctx.strokeStyle="#00ff00"; ctx.lineWidth=2;
      ctx.strokeRect(x*sx+0.5,y*sy+0.5,w*sx,h*sy);
    }else if(moving&&selected>=0){
      const b=boxes[selected];
      const W=naturalW, H=naturalH;
      b.x=Math.max(0,Math.min(W-b.w,p.x-moveDX));
      b.y=Math.max(0,Math.min(H-b.h,p.y-moveDY));
      draw();
    }
  });

  canvas.addEventListener("pointerup",e=>{
    e.preventDefault();
    if(drawing){
      const p=pos(e);
      const x=Math.min(startX,p.x), y=Math.min(startY,p.y);
      const w=Math.abs(p.x-startX), h=Math.abs(p.y-startY);
      if(w>=3&&h>=3) boxes.push({x,y,w,h});
      selected=boxes.length-1;
    }
    drawing=false; moving=false; draw();
    // Send the latest boxes only after the user finishes a draw/move.
    // This avoids reruns while the mouse is moving, but keeps Streamlit
    // synchronized so Apply can use the current boxes directly.
    sendValue();
  });

  document.getElementById("undo").onclick=()=>{
    if(boxes.length){boxes.pop();selected=-1;draw();sendValue();}
  };
  document.getElementById("clear").onclick=()=>{
    boxes=[];selected=-1;draw();sendValue();
  };
  document.getElementById("delete").onclick=()=>{
    if(selected>=0){boxes.splice(selected,1);selected=-1;draw();sendValue();}
  };

  window.addEventListener("resize",fit);

  Streamlit.events.addEventListener(Streamlit.RENDER_EVENT,event=>{
    cfg=event.detail.args||{};
    boxes=(cfg.initial_boxes||[]).map(b=>({
      x:Number(b[0]),y:Number(b[1]),w:Number(b[2]),h:Number(b[3])
    }));
    selected=-1;

    sourceImage=new Image();
    sourceImage.onload=()=>{
      naturalW=sourceImage.naturalWidth||Number(cfg.width)||760;
      naturalH=sourceImage.naturalHeight||Number(cfg.height)||300;
      fit();
    };
    sourceImage.onerror=()=>{
      // Keep a visible black canvas even if the browser cannot decode the
      // data URI, and report a useful component height.
      naturalW=Number(cfg.width)||760;
      naturalH=Number(cfg.height)||300;
      fit();
    };
    sourceImage.src=cfg.image_data||"";
  });

  Streamlit.setComponentReady();
  Streamlit.setFrameHeight(350);
})();
</script>
</body>
</html>"""
(MANUAL_COMPONENT_DIR / "index.html").write_text(MANUAL_COMPONENT_HTML, encoding="utf-8")
manual_box_component = components.declare_component(
    "neolipi_manual_box_editor_v24",
    path=str(MANUAL_COMPONENT_DIR),
)

MODELS = ROOT / "Model"
# Backward-compatible fallback for projects that still use lowercase models/.
LEGACY_MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)


@contextmanager
def working_directory(path):
    """Temporarily change CWD so legacy script relative paths work."""
    previous = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(previous)

# ============================================================
# MANUALLY EXTRACTED CHARACTER OUTPUT
# ============================================================
# Character crops created through the manual segmentation editor
# are saved separately from the automatic "characters" folder.
#
# Output:
#   segmentation_brahmi_v6/
#       manual_characters/
#           line_01_char_001.png
#           line_01_char_002.png
#           ...
#
# Existing binarization, segmentation, CNN, dataset similarity,
# and decision-engine behavior are not changed by this folder.
MANUAL_CHARACTERS_DIR = (
    ROOT / "segmentation_brahmi_v6" / "manual_characters"
)
MANUAL_CHARACTERS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# BRAHMI RECOGNIZER DATASET
# ============================================================
# The dataset shown in the project screenshot is organized as:
#
#   RecognizerDataset_150_210/
#       a/
#       ba/
#       ca/
#       va/
#       ...
#
# Each folder name is the class label.
BRAHMI_DATASET = Path(
    r"D:\NeoLipi_Dataset\RecognizerDataset_150_210\RecognizerDataset_150_210"
)


# ============================================================
# DISCOVERY
# ============================================================

def find_first(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def find_project_file(*names):
    for name in names:
        path = ROOT / name
        if path.exists():
            return path

    # Search recursively, because the CSV may be in a dataset folder.
    for name in names:
        matches = list(ROOT.rglob(name))
        if matches:
            return matches[0]

    return None


def find_script_model():
    # The unified 3-class classifier is stored in Model/ in the current
    # NeoLipi project. Keep models/ as a backward-compatible fallback.
    model_dirs = [MODELS]
    if LEGACY_MODELS != MODELS:
        model_dirs.append(LEGACY_MODELS)

    exact_names = (
        "script_classifier.keras",
        "script_classifier.h5",
        "script_classifier_model.keras",
        "script_classifier_model.h5",
    )

    for model_dir in model_dirs:
        exact = find_first([model_dir / name for name in exact_names])
        if exact:
            return exact

    files = []
    for model_dir in model_dirs:
        for pattern in (
            model_dir / "*script*.keras",
            model_dir / "*script*.h5",
            model_dir / "*classifier*.keras",
            model_dir / "*classifier*.h5",
        ):
            files.extend(glob.glob(str(pattern)))

    return Path(sorted(set(files))[0]) if files else None


def find_brahmi_model():
    exact = find_first([
        MODELS / "best_brahmi_final_v2.keras",
        MODELS / "best_brahmi_final_v2.h5",
        MODELS / "brahmi_model.keras",
        MODELS / "brahmi_model.h5",
    ])
    if exact:
        return exact

    files = []
    for pattern in (
        MODELS / "*brahmi*.keras",
        MODELS / "*brahmi*.h5",
    ):
        files.extend(glob.glob(str(pattern)))

    return Path(sorted(set(files))[0]) if files else None


def find_brahmi_classes():
    return find_first([
        MODELS / "brahmi_classes.json",
        MODELS / "classes.json",
    ])


def find_mapping_csv():
    exact = [
        ROOT / "brahmi_mapping.csv",
        ROOT / "brahmi_mapping",
    ]

    path = find_first(exact)
    if path:
        return path

    # Screenshot showed the filename without .csv in Explorer.
    # Search recursively for any likely CSV mapping file.
    candidates = []

    for csv_path in ROOT.rglob("*.csv"):
        name = csv_path.name.lower()
        if (
            "brahmi" in name
            and (
                "mapping" in name
                or "unicode" in name
            )
        ):
            candidates.append(csv_path)

    if candidates:
        candidates.sort(
            key=lambda p: (
                "brahmi_mapping" not in p.name.lower(),
                len(p.parts),
                str(p).lower(),
            )
        )
        return candidates[0]

    return None


# ============================================================
# MODULE / IMAGE HELPERS
# ============================================================

def load_module(path, name):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name,
        str(path)
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_kannada_translation(records, mapping):
    """Build Kannada output directly from brahmi_mapping.csv.

    The CNN prediction is a class name.  The mapping CSV then provides
    the Kannada equivalent through its ``kannada_glyph`` column.

    No machine translation or external translation service is used here.
    """
    if not records:
        return ""

    output_lines = []
    missing = []

    line_numbers = sorted({int(r["line"]) for r in records})

    for line_no in line_numbers:
        line_records = [
            r for r in records
            if int(r["line"]) == line_no
        ]
        line_records.sort(
            key=lambda r: (
                int(r.get("x", 0)),
                int(r.get("char", 0)),
            )
        )

        kannada_chars = []
        for record in line_records:
            label = str(record.get("label", "")).strip()
            item = map_class_to_glyph(label, mapping)
            row = item.get("row", {}) or {}
            kannada = str(row.get("kannada_glyph", "")).strip()

            if kannada:
                kannada_chars.append(kannada)
            else:
                # Keep the output visibly aligned with the recognized class
                # if the CSV has no Kannada glyph for that class.
                kannada_chars.append(f"[{label}]")
                if label:
                    missing.append(label)

        output_lines.append("".join(kannada_chars))

    return "\n".join(output_lines), sorted(set(missing))


def load_gray(path):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            f"Cannot read image: {path}"
        )

    return image


def load_binary(path):
    image = load_gray(path)

    _, binary = cv2.threshold(
        image,
        127,
        255,
        cv2.THRESH_BINARY
    )

    if np.mean(binary > 0) > 0.60:
        binary = cv2.bitwise_not(binary)

    return binary


def save_uploaded(uploaded):
    folder = Path(
        tempfile.mkdtemp(
            prefix="neolipi_"
        )
    )

    suffix = (
        Path(uploaded.name).suffix
        or ".png"
    )

    path = (
        folder /
        f"inscription{suffix}"
    )

    path.write_bytes(
        uploaded.getbuffer()
    )

    return path


# ============================================================
# SCRIPT CLASSIFIER
# ============================================================

def load_script_classes():
    candidates = [
        ROOT / "script_classes.json",
        ROOT / "class_indices.json",
        MODELS / "script_classes.json",
        MODELS / "class_indices.json",
    ]

    path = find_first(candidates)

    if path is None:
        return [
            "Brahmi",
            "Devanagari",
            "Tamil",
        ]

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    if isinstance(data, list):
        return [str(x) for x in data]

    if isinstance(data, dict):
        try:
            return [
                str(v)
                for _, v in sorted(
                    data.items(),
                    key=lambda item: int(item[0])
                )
            ]
        except (ValueError, TypeError):
            pass

        if all(
            isinstance(v, (int, float))
            for v in data.values()
        ):
            return [
                str(k)
                for k, _ in sorted(
                    data.items(),
                    key=lambda item: item[1]
                )
            ]

    raise ValueError(
        "Unsupported script class file."
    )


@st.cache_resource(show_spinner=False)
def _load_script_classifier_model(model_path):
    import tensorflow as tf
    return tf.keras.models.load_model(str(model_path))


def classify_script(image_path):
    """
    EXACT 3-script classifier inference used by the working classifier.

    Original uploaded image -> BGR to RGB -> 224x224 -> float32 ->
    EfficientNet preprocess_input -> batch -> 3-class model.

    IMPORTANT: do NOT grayscale, threshold, binarize, /255, or apply
    OCR preprocessing before this classifier.
    """
    model_path = find_script_model()
    if model_path is None:
        raise FileNotFoundError(
            "Script classifier model was not found. Checked Model/ and models/."
        )

    from tensorflow.keras.applications.efficientnet import preprocess_input

    model = _load_script_classifier_model(model_path)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"Could not read classifier input image: {image_path}")

    # EXACT preprocessing: original image, RGB, 224x224, EfficientNet.
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    rgb = rgb.astype(np.float32)
    rgb = preprocess_input(rgb)
    x = np.expand_dims(rgb, axis=0)

    prediction = model.predict(x, verbose=0)[0]
    if len(prediction) != 3:
        raise ValueError(f"Expected 3 classifier outputs, got {len(prediction)}")

    # This is the class order used by the working 3-script classifier.
    classes = ["Brahmi", "Devanagari", "Tamil"]
    order = np.argsort(prediction)[::-1]
    best = int(order[0])

    top = [
        {
            "label": classes[int(i)],
            "confidence": float(prediction[int(i)]),
        }
        for i in order
    ]

    return {
        "label": classes[best],
        "confidence": float(prediction[best]),
        "top": top,
        "input_shape": "original image → RGB → 224×224 → EfficientNet preprocess_input",
        "raw_scores": {classes[i]: float(prediction[i]) for i in range(3)},
    }

def load_brahmi_classes():
    path = find_brahmi_classes()

    if path is None:
        raise FileNotFoundError(
            "brahmi_classes.json was not found in models/."
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    if isinstance(data, list):
        return [
            str(x)
            for x in data
        ]

    if isinstance(data, dict):
        try:
            return [
                str(v)
                for _, v in sorted(
                    data.items(),
                    key=lambda item: int(item[0])
                )
            ]
        except (ValueError, TypeError):
            pass

    raise ValueError(
        "Unsupported Brahmi class JSON."
    )


# ============================================================
# BRAHMI MAPPING CSV
# ============================================================

def clean_column(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace("\ufeff", "")
        .replace(" ", "_")
        .replace("-", "_")
    )



def load_brahmi_mapping_csv():
    path = find_mapping_csv()

    if path is None:
        raise FileNotFoundError(
            "brahmi_mapping.csv "
            "was not found anywhere inside the project."
        )

    # Detect the delimiter. The file is CSV, but using the Sniffer
    # makes it tolerant to comma/semicolon/tab formatting.
    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        sample = f.read(4096)

        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(
                sample,
                delimiters=",;\t|"
            )
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(
            f,
            dialect=dialect
        )

        raw_rows = list(reader)

    if not raw_rows:
        raise ValueError(
            f"Mapping CSV is empty: {path}"
        )

    # Normalize column names.
    rows = []

    for raw in raw_rows:
        normalized = {
            clean_column(k): (
                str(v).strip()
                if v is not None
                else ""
            )
            for k, v in raw.items()
            if k is not None
        }

        rows.append(normalized)

    # Exact uploaded CSV schema after normalization:
    # class_name, unicode_glyph, unicode_hex,
    # unicode_character_name, script_category.
    possible_label_columns = [
        "class_name",
        "class_label",
        "label",
        "class",
        "character",
        "name",
        "symbol",
    ]

    possible_glyph_columns = [
        "unicode_glyph",
        "brahmi_glyph",
        "glyph",
        "unicode_character",
        "unicode",
        "text",
    ]

    possible_hex_columns = [
        "unicode_hex",
        "codepoint",
        "code_point",
        "unicode_codepoint",
    ]


    label_column = None

    for col in possible_label_columns:
        if col in rows[0]:
            label_column = col
            break

    if label_column is None:
        raise ValueError(
            "Could not identify the class-label column in "
            f"{path.name}. Columns found: {list(rows[0].keys())}"
        )

    glyph_column = next(
        (
            c for c in possible_glyph_columns
            if c in rows[0]
        ),
        None
    )

    hex_column = next(
        (
            c for c in possible_hex_columns
            if c in rows[0]
        ),
        None
    )

    mapping = {}

    for row in rows:
        label = row.get(
            label_column,
            ""
        ).strip()

        if not label:
            continue

        glyph = (
            row.get(glyph_column, "").strip()
            if glyph_column
            else ""
        )

        codepoint = (
            row.get(hex_column, "").strip()
            if hex_column
            else ""
        )

        # If the CSV stores only a Unicode hex codepoint,
        # convert it to the actual Brahmi Unicode character(s)
        # when possible.
        if not glyph and codepoint:
            glyph = codepoint_to_text(
                codepoint
            )

        mapping[label] = {
            "class_label": label,
            "glyph": glyph,
            "unicode_hex": codepoint,
            "character_name": row.get(
                "unicode_character_name",
                ""
            ),
            "script_category": row.get(
                "script_category",
                ""
            ),
            "kannada_glyph": row.get(
                "kannada_glyph",
                ""
            ),
            "row": row,
        }

    if not mapping:
        raise ValueError(
            f"No usable class labels were found in {path}."
        )

    return path, mapping


def codepoint_to_text(value):
    value = str(value).strip()

    if not value:
        return ""

    # Supports:
    # U+11003
    # U+11024-U+11038
    # 11003
    # 11024-11038
    parts = value.replace(
        " ",
        ""
    ).upper().split("-")

    chars = []

    for part in parts:
        if part.startswith("U+"):
            part = part[2:]

        try:
            chars.append(
                chr(
                    int(
                        part,
                        16
                    )
                )
            )
        except ValueError:
            return ""

    return "".join(chars)


def normalize_label(label):
    return (
        str(label)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def map_class_to_glyph(
    label,
    mapping
):
    if label in mapping:
        return mapping[label]

    normalized = normalize_label(
        label
    )

    for key, value in mapping.items():
        if normalize_label(key) == normalized:
            return value

    return {
        "class_label": label,
        "glyph": "",
        "unicode_hex": "",
        "row": {},
    }


# ============================================================
# CHARACTER PREPROCESSING / OCR
# ============================================================

def prepare_character(
    image,
    size=75
):
    _, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    ys, xs = np.where(
        binary > 0
    )

    if len(xs) == 0:
        return np.zeros(
            (size, size),
            dtype=np.uint8
        )

    x1 = int(
        xs.min()
    )

    x2 = int(
        xs.max()
    )

    y1 = int(
        ys.min()
    )

    y2 = int(
        ys.max()
    )

    binary = binary[
        y1:y2 + 1,
        x1:x2 + 1
    ]

    h, w = binary.shape

    pad = max(
        2,
        int(
            max(h, w) *
            0.15
        )
    )

    binary = cv2.copyMakeBorder(
        binary,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=0
    )

    h, w = binary.shape

    scale = (
        size *
        0.80
    ) / max(
        h,
        w
    )

    nw = max(
        1,
        int(
            round(
                w * scale
            )
        )
    )

    nh = max(
        1,
        int(
            round(
                h * scale
            )
        )
    )

    binary = cv2.resize(
        binary,
        (nw, nh),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    sx = (
        size -
        nw
    ) // 2

    sy = (
        size -
        nh
    ) // 2

    canvas[
        sy:sy + nh,
        sx:sx + nw
    ] = binary

    # Keep the existing model preprocessing behavior.
    density = (
        np.count_nonzero(
            canvas
        )
        /
        canvas.size
    )

    if density > 0.18:
        canvas = cv2.erode(
            canvas,
            np.ones(
                (2, 2),
                dtype=np.uint8
            ),
            iterations=1
        )

    return 255 - canvas


def recognize_character(
    crop,
    model,
    classes
):
    prepared = prepare_character(
        crop
    )

    input_shape = model.input_shape

    if isinstance(
        input_shape,
        list
    ):
        input_shape = input_shape[0]

    if len(input_shape) != 4:
        raise ValueError(
            f"Unsupported Brahmi model input shape: "
            f"{input_shape}"
        )

    target_h = int(
        input_shape[1]
        if input_shape[1] is not None
        else 75
    )

    target_w = int(
        input_shape[2]
        if input_shape[2] is not None
        else 75
    )

    channels = int(
        input_shape[3]
        if input_shape[3] is not None
        else 1
    )

    x = (
        prepared.astype(
            np.float32
        )
        /
        255.0
    )

    if x.shape != (
        target_h,
        target_w
    ):
        x = cv2.resize(
            x,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA
        )

    if channels == 1:
        x = x[..., np.newaxis]

    elif channels == 3:
        x = np.repeat(
            x[..., np.newaxis],
            3,
            axis=-1
        )

    else:
        raise ValueError(
            f"Unsupported Brahmi model channels: {channels}"
        )

    x = x[np.newaxis, ...]

    probabilities = model.predict(
        x,
        verbose=0
    )[0]

    if len(probabilities) != len(classes):
        raise ValueError(
            "Brahmi model output count does not match "
            "brahmi_classes.json."
        )

    order = np.argsort(
        probabilities
    )[::-1]

    top5 = [
        {
            "label": classes[int(i)],
            "confidence": float(
                probabilities[int(i)]
            )
        }
        for i in order[:5]
    ]

    best = int(
        order[0]
    )

    return (
        prepared,
        classes[best],
        float(
            probabilities[best]
        ),
        top5
    )
# ============================================================
# DATASET CHARACTER SIMILARITY
# ============================================================

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp",
    ".tif", ".tiff", ".webp"
}


def _normalize_similarity_image(image, size=75):
    """Normalize a character image to a centered 75x75 binary shape."""
    if image is None or image.size == 0:
        return None

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Make a binary foreground mask.
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Force the character to white and background to black.
    # This handles both white-on-black and black-on-white inputs.
    if np.mean(binary > 0) > 0.50:
        binary = cv2.bitwise_not(binary)

    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return np.zeros((size, size), dtype=np.uint8)

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    character = binary[y1:y2, x1:x2]

    h, w = character.shape
    available = int(size * 0.80)
    scale = min(
        available / max(w, 1),
        available / max(h, 1)
    )

    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    character = cv2.resize(
        character,
        (nw, nh),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((size, size), dtype=np.uint8)
    sx = (size - nw) // 2
    sy = (size - nh) // 2
    canvas[sy:sy + nh, sx:sx + nw] = character

    return canvas


@st.cache_resource(show_spinner=False)
def load_brahmi_dataset_gallery(dataset_path, max_images_per_class=40):
    """Load and normalize representative dataset images once per app run."""
    root = Path(dataset_path)

    if not root.exists():
        return {}

    gallery = {}

    for class_folder in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not class_folder.is_dir():
            continue

        paths = [
            p for p in sorted(class_folder.iterdir(), key=lambda p: p.name.lower())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ][:max_images_per_class]

        samples = []
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            normalized = _normalize_similarity_image(image)
            if normalized is not None:
                samples.append(normalized)

        if samples:
            gallery[class_folder.name] = samples

    return gallery


def _shape_similarity(a, b):
    """IoU/Jaccard similarity between two normalized binary shapes."""
    if a is None or b is None:
        return 0.0

    a_mask = a > 127
    b_mask = b > 127

    intersection = np.logical_and(a_mask, b_mask).sum()
    union = np.logical_or(a_mask, b_mask).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def compare_with_dataset(crop, candidate_classes=None, max_images_per_class=40):
    """Compare an extracted character with class-wise dataset examples.

    The folder name is the dataset class. The class score is the mean of
    the best five image similarities in that class. This avoids letting a
    single unusual training image decide the result.
    """
    if not BRAHMI_DATASET.exists():
        return {
            "label": None,
            "similarity": 0.0,
            "top": [],
            "available": False,
        }

    gallery = load_brahmi_dataset_gallery(
        str(BRAHMI_DATASET),
        max_images_per_class=max_images_per_class,
    )

    if not gallery:
        return {
            "label": None,
            "similarity": 0.0,
            "top": [],
            "available": False,
        }

    query = _normalize_similarity_image(crop)
    if query is None:
        return {
            "label": None,
            "similarity": 0.0,
            "top": [],
            "available": True,
        }

    # By default compare against every class. This is important because
    # a wrong CNN prediction can otherwise hide the correct class.
    allowed = None
    if candidate_classes:
        allowed = {
            str(x).strip().lower()
            for x in candidate_classes
        }

    results = []

    for class_name, samples in gallery.items():
        if allowed is not None and class_name.lower() not in allowed:
            # We intentionally do NOT use this filter by default in the
            # pipeline below. It is kept for optional experiments.
            continue

        scores = [
            _shape_similarity(query, sample)
            for sample in samples
        ]

        if not scores:
            continue

        scores.sort(reverse=True)
        best_scores = scores[:min(5, len(scores))]

        results.append({
            "label": class_name,
            "similarity": float(np.mean(best_scores)),
            "best_single": float(scores[0]),
            "samples_used": len(samples),
        })

    results.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    if not results:
        return {
            "label": None,
            "similarity": 0.0,
            "top": [],
            "available": True,
        }

    return {
        "label": results[0]["label"],
        "similarity": results[0]["similarity"],
        "top": results[:5],
        "available": True,
    }


# ============================================================
# MANUAL CHARACTER SEGMENTATION HELPERS
# ============================================================

def extract_manual_boxes_from_canvas(json_data, line_box, scale_x=1.0, scale_y=1.0):
    """
    Extract bounding boxes from st_canvas and scale back to global image coordinates (x1, y1, x2, y2),
    matching the robust parsing pattern used in the Tamil pipeline[cite: 9].
    """
    boxes = []
    if not json_data:
        return boxes

    if isinstance(json_data, str):
        try:
            json_data = json.loads(json_data)
        except Exception:
            return boxes

    if not isinstance(json_data, dict) or "objects" not in json_data:
        return boxes

    lx1, ly1, _, _ = line_box

    for obj in json_data.get("objects", []):
        obj_type = str(obj.get("type", "")).lower()
        if obj_type in ["rect", "rectangle"]:
            left = float(obj.get("left", 0.0))
            top = float(obj.get("top", 0.0))
            width = float(obj.get("width", 0.0))
            height = float(obj.get("height", 0.0))
            sx = float(obj.get("scaleX", 1.0))
            sy = float(obj.get("scaleY", 1.0))

            actual_w = abs(width * sx)
            actual_h = abs(height * sy)
            actual_x = left if sx >= 0 else left - actual_w
            actual_y = top if sy >= 0 else top - actual_h

            x = actual_x * scale_x
            y = actual_y * scale_y
            w = actual_w * scale_x
            h = actual_h * scale_y

            if w > 1 and h > 1:
                gx1 = int(round(lx1 + x))
                gy1 = int(round(ly1 + y))
                gx2 = int(round(lx1 + x + w))
                gy2 = int(round(ly1 + y + h))
                boxes.append((gx1, gy1, gx2, gy2))

    boxes.sort(key=lambda b: (b[0], b[1]))
    return boxes


def _records_from_manual_boxes(result, line_no, manual_boxes):
    """
    Re-run recognition for one manually corrected line using the all-class CNN pipeline.
    """
    if not manual_boxes:
        return result["records"]

    run_dir = result.get("run_dir")
    if not run_dir:
        run_dir, _, _ = create_character_run_dir()
        result["run_dir"] = str(run_dir)

    run_dir = Path(run_dir)
    extracted_dir = run_dir / "extracted"
    preprocessed_dir = run_dir / "preprocessed"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    binary = result["binary"]
    H, W = binary.shape[:2]

    # Remove old extracted/preprocessed images belonging to this line[cite: 12].
    for pattern in (
        f"line_{int(line_no):02d}_char_*.png",
        f"line_{int(line_no):02d}_char_*_preprocessed.png",
    ):
        for old_file in list(extracted_dir.glob(pattern)) + list(preprocessed_dir.glob(pattern)):
            try:
                old_file.unlink()
            except OSError:
                pass

    replacement = []

    for char_no, (x1, y1, x2, y2) in enumerate(manual_boxes, start=1):
        x1 = max(0, min(W - 1, int(x1)))
        y1 = max(0, min(H - 1, int(y1)))
        x2 = max(x1 + 1, min(W, int(x2)))
        y2 = max(y1 + 1, min(H, int(y2)))

        crop = binary[y1:y2, x1:x2].copy()
        save_extracted_character(
            run_dir,
            line_no,
            char_no,
            crop,
        )

    preprocess_extracted_folder(run_dir)
    predictions = predict_preprocessed_folder(run_dir)

    _, mapping = load_brahmi_mapping_csv()

    for char_no, (x1, y1, x2, y2) in enumerate(manual_boxes, start=1):
        key = _prediction_key(line_no, char_no)
        prediction = predictions.get(key)

        if prediction is None:
            continue

        final_label = prediction["label"]
        mapping_item = map_class_to_glyph(final_label, mapping)

        replacement.append({
            "global_index": None,
            "line": int(line_no),
            "char": int(char_no),
            "x": x1,
            "y": int(y1),
            "w": x2 - x1,
            "h": y2 - y1,
            "label": final_label,
            "cnn_label": final_label,
            "confidence": prediction["confidence"],
            "recognition_status": recognition_status(
                prediction["confidence"]
            ),
            "dataset_label": None,
            "dataset_similarity": 0.0,
            "dataset_top": [],
            "decision_source": "ALL_CLASS_CNN",
            "brahmi_glyph": mapping_item.get("glyph", ""),
            "unicode_hex": mapping_item.get("unicode_hex", ""),
            "unicode_character_name": mapping_item.get(
                "character_name", ""
            ),
            "raw": crop,
            "cnn": cv2.imread(
                str(preprocessed_dir / f"{key}_preprocessed.png"),
                cv2.IMREAD_GRAYSCALE,
            ),
            "top5": prediction["top5"],
            "extracted_path": str(
                extracted_dir / f"{key}.png"
            ),
            "preprocessed_path": prediction["preprocessed_path"],
        })

    other_records = [
        r for r in result["records"]
        if int(r["line"]) != int(line_no)
    ]

    merged = other_records + replacement
    merged.sort(
        key=lambda r: (
            int(r["line"]),
            int(r.get("x", 0)),
        )
    )

    for index, record in enumerate(merged, start=1):
        record["global_index"] = index

    return merged

# ============================================================
# MANUAL CHARACTER SEGMENTATION HELPERS
# ============================================================

def extract_manual_boxes_from_canvas(json_data, line_box, scale_x=1.0, scale_y=1.0):
    """
    Extract bounding boxes from st_canvas and scale back to global image coordinates (x1, y1, x2, y2),
    matching the robust parsing pattern.
    """
    boxes = []
    if not json_data:
        return boxes

    if isinstance(json_data, str):
        try:
            json_data = json.loads(json_data)
        except Exception:
            return boxes

    if not isinstance(json_data, dict) or "objects" not in json_data:
        return boxes

    lx1, ly1, _, _ = line_box

    for obj in json_data.get("objects", []):
        obj_type = str(obj.get("type", "")).lower()
        if obj_type in ["rect", "rectangle"]:
            left = float(obj.get("left", 0.0))
            top = float(obj.get("top", 0.0))
            width = float(obj.get("width", 0.0))
            height = float(obj.get("height", 0.0))
            sx = float(obj.get("scaleX", 1.0))
            sy = float(obj.get("scaleY", 1.0))

            actual_w = abs(width * sx)
            actual_h = abs(height * sy)
            actual_x = left if sx >= 0 else left - actual_w
            actual_y = top if sy >= 0 else top - actual_h

            x = actual_x * scale_x
            y = actual_y * scale_y
            w = actual_w * scale_x
            h = actual_h * scale_y

            if w > 1 and h > 1:
                gx1 = int(round(lx1 + x))
                gy1 = int(round(ly1 + y))
                gx2 = int(round(lx1 + x + w))
                gy2 = int(round(ly1 + y + h))
                boxes.append((gx1, gy1, gx2, gy2))

    boxes.sort(key=lambda b: (b[0], b[1]))
    return boxes


def _records_from_manual_boxes(result, line_no, manual_boxes):
    """
    Re-run recognition for one manually corrected line using the all-class CNN pipeline.
    """
    if not manual_boxes:
        return result["records"]

    run_dir = result.get("run_dir")
    if not run_dir:
        run_dir, _, _ = create_character_run_dir()
        result["run_dir"] = str(run_dir)

    run_dir = Path(run_dir)
    extracted_dir = run_dir / "extracted"
    preprocessed_dir = run_dir / "preprocessed"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    binary = result["binary"]
    H, W = binary.shape[:2]

    # Remove old extracted/preprocessed images belonging to this line.
    for pattern in (
        f"line_{int(line_no):02d}_char_*.png",
        f"line_{int(line_no):02d}_char_*_preprocessed.png",
    ):
        for old_file in list(extracted_dir.glob(pattern)) + list(preprocessed_dir.glob(pattern)):
            try:
                old_file.unlink()
            except OSError:
                pass

    replacement = []

    for char_no, (x1, y1, x2, y2) in enumerate(manual_boxes, start=1):
        x1 = max(0, min(W - 1, int(x1)))
        y1 = max(0, min(H - 1, int(y1)))
        x2 = max(x1 + 1, min(W, int(x2)))
        y2 = max(y1 + 1, min(H, int(y2)))

        crop = binary[y1:y2, x1:x2].copy()
        save_extracted_character(
            run_dir,
            line_no,
            char_no,
            crop,
        )

    preprocess_extracted_folder(run_dir)
    predictions = predict_preprocessed_folder(run_dir)

    _, mapping = load_brahmi_mapping_csv()

    for char_no, (x1, y1, x2, y2) in enumerate(manual_boxes, start=1):
        key = _prediction_key(line_no, char_no)
        prediction = predictions.get(key)

        if prediction is None:
            continue

        final_label = prediction["label"]
        mapping_item = map_class_to_glyph(final_label, mapping)

        replacement.append({
            "global_index": None,
            "line": int(line_no),
            "char": int(char_no),
            "x": x1,
            "y": int(y1),
            "w": x2 - x1,
            "h": y2 - y1,
            "label": final_label,
            "cnn_label": final_label,
            "confidence": prediction["confidence"],
            "recognition_status": recognition_status(
                prediction["confidence"]
            ),
            "dataset_label": None,
            "dataset_similarity": 0.0,
            "dataset_top": [],
            "decision_source": "ALL_CLASS_CNN",
            "brahmi_glyph": mapping_item.get("glyph", ""),
            "unicode_hex": mapping_item.get("unicode_hex", ""),
            "unicode_character_name": mapping_item.get(
                "character_name", ""
            ),
            "raw": crop,
            "cnn": cv2.imread(
                str(preprocessed_dir / f"{key}_preprocessed.png"),
                cv2.IMREAD_GRAYSCALE,
            ),
            "top5": prediction["top5"],
            "extracted_path": str(
                extracted_dir / f"{key}.png"
            ),
            "preprocessed_path": prediction["preprocessed_path"],
        })

    other_records = [
        r for r in result["records"]
        if int(r["line"]) != int(line_no)
    ]

    merged = other_records + replacement
    merged.sort(
        key=lambda r: (
            int(r["line"]),
            int(r.get("x", 0)),
        )
    )

    for index, record in enumerate(merged, start=1):
        record["global_index"] = index

    return merged


def render_manual_segmentation_editor(result):
    """Render an interactive per-line rectangle drawing canvas that updates downstream pipelines continuously."""
    if not DRAWABLE_CANVAS_AVAILABLE:
        st.warning(
            "Manual segmentation editor needs streamlit-drawable-canvas. "
            "Run: pip install streamlit-drawable-canvas"
        )
        return

    st.subheader("4. Manual Segmentation Correction")
    st.write(
        "Select an inscription line below, then draw green rectangles around each character. "
        "Recognition and subsequent pipeline stages will update continuously as you draw."
    )

    line_boxes = result["line_boxes"]
    if not line_boxes:
        st.warning("No line boxes found.")
        return

    line_no = st.selectbox(
        "Select inscription line to correct",
        options=list(range(1, len(line_boxes) + 1)),
        format_func=lambda n: f"Line {n}",
        key="manual_line_selector",
    )

    lx1, ly1, lx2, ly2 = line_boxes[line_no - 1]
    line = result["binary"][ly1:ly2, lx1:lx2]

    # Invert line for high-contrast display: black strokes on white background
    line_display = cv2.bitwise_not(line)

    orig_h, orig_w = line.shape[:2]
    target_canvas_w = 900
    target_canvas_h = max(140, int(target_canvas_w * (orig_h / float(max(1, orig_w)))))

    bg_display_img = cv2.resize(
        line_display,
        (target_canvas_w, target_canvas_h),
        interpolation=cv2.INTER_CUBIC,
    )
    pil_bg_img = Image.fromarray(cv2.cvtColor(bg_display_img, cv2.COLOR_GRAY2RGB))

    scale_x = orig_w / float(target_canvas_w)
    scale_y = orig_h / float(target_canvas_h)

    canvas_result = st_canvas(
        fill_color="rgba(0, 255, 0, 0.20)",
        stroke_width=2,
        stroke_color="#00FF00",
        background_image=pil_bg_img,
        update_streamlit=True,
        height=target_canvas_h,
        width=target_canvas_w,
        drawing_mode="rect",
        key=f"manual_canvas_line_{line_no}",
    )

    if canvas_result.json_data:
        boxes = extract_manual_boxes_from_canvas(
            canvas_result.json_data,
            (lx1, ly1, lx2, ly2),
            scale_x=scale_x,
            scale_y=scale_y,
        )

        if boxes:
            cache_key = f"processed_boxes_line_{line_no}"
            current_json_str = json.dumps(canvas_result.json_data.get("objects", []))

            if st.session_state.get(cache_key) != current_json_str:
                st.session_state[cache_key] = current_json_str
                with st.spinner(f"Updating recognition for Line {line_no}..."):
                    result["records"] = _records_from_manual_boxes(result, line_no, boxes)
                st.session_state["result"] = result
                st.rerun()

# ============================================================
# RECOGNITION CONFIDENCE STATUS
# ============================================================

def recognition_status(confidence):
    """
    Confidence is expected in the range 0.0 - 1.0.

    This does not alter the CNN prediction.
    It only labels how cautiously the prediction
    should be interpreted.
    """
    confidence = float(confidence)

    if confidence >= 0.85:
        return "HIGH"

    if confidence >= 0.65:
        return "MEDIUM"

    if confidence >= 0.45:
        return "LOW"

    return "UNCERTAIN"

# ============================================================
# BRAHMI LABEL -> READABLE TEXT
# ============================================================

def build_readable_text(records):

    if not records:
        return ""

    readable_lines = []

    line_numbers = sorted(
        set(
            int(r["line"])
            for r in records
        )
    )

    for line_no in line_numbers:

        line_records = [
            r
            for r in records
            if int(r["line"]) == line_no
        ]

        # IMPORTANT:
        # Sort characters from LEFT -> RIGHT
        line_records.sort(
            key=lambda r: (
                int(r.get("x", 0)),
                int(r.get("char", 0))
            )
        )

        labels = []

        for r in line_records:

            label = str(
                r.get("label", "")
            ).strip()

            if label:
                labels.append(label)

        if labels:
            readable_lines.append(
                " ".join(labels)
            )

    return "\n".join(
        readable_lines
    )



# ============================================================
# NEW ALL-CLASS CNN MAPPING PIPELINE
# ============================================================
# This block intentionally overrides only the recognition/storage
# functions. The existing binarization and segmentation modules are
# preserved unchanged.

ALL_CLASS_CNN_MODEL_PATH = next((p for p in [
    ROOT / "Model" / "cnn_brahmi.keras",
    ROOT / "Model" / "cnn_brahmi.h5",
    ROOT / "models" / "cnn_brahmi.keras",
    ROOT / "models" / "cnn_brahmi.h5",
    Path(r"D:\NeoLipi_VSCODE\NeoLipi_CRNN\cnn_all_classes_stone_augmented_best.keras"),
] if p.exists()), ROOT / "Model" / "cnn_brahmi.keras")

ALL_CLASS_NAMES_PATH = next((p for p in [
    ROOT / "Model" / "cnn_brahmi.txt",
    ROOT / "Model" / "all_class_names.txt",
    ROOT / "Model" / "brahmi_classes.json",
    ROOT / "models" / "cnn_brahmi.txt",
    ROOT / "models" / "all_class_names.txt",
    Path(r"D:\NeoLipi_VSCODE\NeoLipi_CRNN\all_class_names.txt"),
] if p.exists()), ROOT / "Model" / "cnn_brahmi.txt")

MODEL_IMG_SIZE = 64

MANUAL_CHARACTER_ROOT = (
    ROOT / "segmentation_brahmi_v6" / "manual_characters"
)
MANUAL_CHARACTER_ROOT.mkdir(parents=True, exist_ok=True)


def find_brahmi_model():
    candidates = [
        ALL_CLASS_CNN_MODEL_PATH,
        ROOT / "Model" / "cnn_brahmi.keras",
        ROOT / "Model" / "cnn_brahmi.h5",
        ROOT / "Model" / "best_brahmi_final_v2.keras",
        ROOT / "Model" / "best_brahmi_final_v2.h5",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for model_dir in (ROOT / "Model", ROOT / "models"):
        if model_dir.exists():
            matches = sorted(list(model_dir.glob("*brahmi*.keras")) + list(model_dir.glob("*brahmi*.h5")))
            if matches:
                return matches[0]
    raise FileNotFoundError(
        "Brahmi all-class CNN was not found. Expected Model\\cnn_brahmi.keras (or another *brahmi*.keras/.h5 model)."
    )


def load_brahmi_classes():
    candidates = [
        ALL_CLASS_NAMES_PATH,
        ROOT / "Model" / "cnn_brahmi.txt",
        ROOT / "Model" / "all_class_names.txt",
        ROOT / "Model" / "brahmi_classes.json",
        ROOT / "models" / "cnn_brahmi.txt",
        ROOT / "models" / "all_class_names.txt",
        ROOT / "models" / "brahmi_classes.json",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "Brahmi class-name file was not found. Expected Model\\cnn_brahmi.txt, "
            "Model\\all_class_names.txt, or brahmi_classes.json."
        )

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            classes = [str(x).strip() for x in data if str(x).strip()]
        elif isinstance(data, dict):
            try:
                classes = [str(v).strip() for _, v in sorted(data.items(), key=lambda item: int(item[0])) if str(v).strip()]
            except Exception:
                classes = [str(k).strip() for k in data.keys() if str(k).strip()]
        else:
            classes = []
    else:
        with open(path, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
    if not classes:
        raise ValueError(f"Brahmi class-name file is empty: {path}")
    return classes


@st.cache_resource(show_spinner=False)
def load_all_class_cnn():
    """Load the working all-class CNN once per Streamlit process."""
    import tensorflow as tf

    model_path = find_brahmi_model()
    model = tf.keras.models.load_model(str(model_path))
    classes = load_brahmi_classes()

    output_shape = model.output_shape
    if isinstance(output_shape, list):
        output_shape = output_shape[0]

    output_count = int(output_shape[-1])
    if output_count != len(classes):
        raise ValueError(
            "CNN output count does not match all_class_names.txt: "
            f"model={output_count}, classes={len(classes)}"
        )

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) != 4:
        raise ValueError(
            f"Unsupported CNN input shape: {input_shape}"
        )

    return model, classes


def create_character_run_dir():
    """Create a separate extraction/preprocessing folder per inscription run."""
    import datetime

    stamp = datetime.datetime.now().strftime("stone_%Y%m%d_%H%M%S_%f")[:-3]
    run_dir = MANUAL_CHARACTER_ROOT / stamp
    extracted_dir = run_dir / "extracted"
    preprocessed_dir = run_dir / "preprocessed"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, extracted_dir, preprocessed_dir


def save_extracted_character(run_dir, line_no, char_no, crop):
    """Save the exact binary crop produced by the existing segmentation."""
    extracted_dir = Path(run_dir) / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    path = extracted_dir / (
        f"line_{int(line_no):02d}_char_{int(char_no):03d}.png"
    )

    if not cv2.imwrite(str(path), crop):
        raise IOError(f"Could not save extracted character: {path}")

    return path


def preprocess_character_for_all_class_cnn(image, size=MODEL_IMG_SIZE):
    """Match the preprocessing used to prepare the augmented stone characters.

    Output:
        64x64, grayscale, black character on white background.
    """
    if image is None or image.size == 0:
        raise ValueError("Empty character image.")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Otsu threshold.
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Make the foreground the white character temporarily.
    # If white occupies most of the image, the character is assumed
    # to be black on white and is inverted.
    white_pixels = cv2.countNonZero(binary)
    total_pixels = binary.shape[0] * binary.shape[1]
    black_pixels = total_pixels - white_pixels

    if white_pixels > black_pixels:
        binary = cv2.bitwise_not(binary)

    # Find character foreground.
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.full((size, size), 255, dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary[y:y + h, x:x + w]

    # Padding used in the standalone preprocessing pipeline.
    padding = 8
    cropped = cv2.copyMakeBorder(
        cropped,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    h, w = cropped.shape
    square_size = max(h, w)
    square = np.zeros(
        (square_size, square_size),
        dtype=np.uint8,
    )

    y_offset = (square_size - h) // 2
    x_offset = (square_size - w) // 2

    square[
        y_offset:y_offset + h,
        x_offset:x_offset + w,
    ] = cropped

    resized = cv2.resize(
        square,
        (size, size),
        interpolation=cv2.INTER_AREA,
    )

    # Final format expected by the working model:
    # white background + black character.
    final_image = cv2.bitwise_not(resized)

    _, final_image = cv2.threshold(
        final_image,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    return final_image


def preprocess_extracted_folder(run_dir):
    """Preprocess every extracted image and save it to preprocessed/."""
    run_dir = Path(run_dir)
    extracted_dir = run_dir / "extracted"
    preprocessed_dir = run_dir / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        [
            p for p in extracted_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda p: p.name.lower(),
    )

    processed = {}

    for image_path in image_files:
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )
        if image is None:
            continue

        prepared = preprocess_character_for_all_class_cnn(image)
        output_path = preprocessed_dir / (
            image_path.stem + "_preprocessed.png"
        )

        if not cv2.imwrite(str(output_path), prepared):
            raise IOError(
                f"Could not save preprocessed character: {output_path}"
            )

        processed[image_path.name] = output_path

    return processed


def predict_preprocessed_folder(run_dir):
    """Run the all-class CNN on the saved preprocessed folder."""
    model, classes = load_all_class_cnn()
    preprocessed_dir = Path(run_dir) / "preprocessed"

    predictions = {}

    image_files = sorted(
        [
            p for p in preprocessed_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda p: p.name.lower(),
    )

    if not image_files:
        return predictions

    batch_images = []
    valid_paths = []

    for image_path in image_files:
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )
        if image is None:
            continue

        image = cv2.resize(
            image,
            (MODEL_IMG_SIZE, MODEL_IMG_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        x = image.astype(np.float32) / 255.0
        x = x[..., np.newaxis]

        batch_images.append(x)
        valid_paths.append(image_path)

    if not batch_images:
        return predictions

    batch = np.stack(batch_images, axis=0)
    probabilities = model.predict(batch, verbose=0)

    for image_path, probs in zip(valid_paths, probabilities):
        order = np.argsort(probs)[::-1]
        best = int(order[0])

        top5 = [
            {
                "label": classes[int(i)],
                "confidence": float(probs[int(i)]),
            }
            for i in order[:5]
        ]

        predictions[image_path.stem.replace("_preprocessed", "")] = {
            "label": classes[best],
            "confidence": float(probs[best]),
            "top5": top5,
            "preprocessed_path": str(image_path),
        }

    return predictions


def recognize_character(crop, model=None, classes=None):
    """Compatibility wrapper: preprocess one crop and classify it."""
    prepared = preprocess_character_for_all_class_cnn(crop)

    if model is None or classes is None:
        model, classes = load_all_class_cnn()

    x = prepared.astype(np.float32) / 255.0
    x = x[..., np.newaxis]
    x = x[np.newaxis, ...]

    probabilities = model.predict(x, verbose=0)[0]
    order = np.argsort(probabilities)[::-1]

    top5 = [
        {
            "label": classes[int(i)],
            "confidence": float(probabilities[int(i)]),
        }
        for i in order[:5]
    ]

    best = int(order[0])

    return (
        prepared,
        classes[best],
        float(probabilities[best]),
        top5,
    )


def _prediction_key(line_no, char_no):
    return f"line_{int(line_no):02d}_char_{int(char_no):03d}"


def _records_from_manual_boxes(result, line_no, manual_boxes):
    """Replace one line's extracted crops, preprocess the whole run folder,
    and map the resulting preprocessed images with the all-class CNN."""
    if not manual_boxes:
        return result["records"]

    run_dir = result.get("run_dir")
    if not run_dir:
        run_dir, _, _ = create_character_run_dir()
        result["run_dir"] = str(run_dir)

    run_dir = Path(run_dir)
    extracted_dir = run_dir / "extracted"
    preprocessed_dir = run_dir / "preprocessed"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    binary = result["binary"]
    H, W = binary.shape[:2]

    # Remove old extracted/preprocessed images belonging to this line.
    for pattern in (
        f"line_{int(line_no):02d}_char_*.png",
        f"line_{int(line_no):02d}_char_*_preprocessed.png",
    ):
        for old_file in list(extracted_dir.glob(pattern)) + list(preprocessed_dir.glob(pattern)):
            try:
                old_file.unlink()
            except OSError:
                pass

    replacement = []

    for char_no, (x1, y1, x2, y2) in enumerate(manual_boxes, start=1):
        x1 = max(0, min(W - 1, int(x1)))
        y1 = max(0, min(H - 1, int(y1)))
        x2 = max(x1 + 1, min(W, int(x2)))
        y2 = max(y1 + 1, min(H, int(y2)))

        crop = binary[y1:y2, x1:x2].copy()
        save_extracted_character(
            run_dir,
            line_no,
            char_no,
            crop,
        )

    # The requirement is that mapping always consumes the complete
    # preprocessed folder, not an in-memory crop.
    preprocess_extracted_folder(run_dir)
    predictions = predict_preprocessed_folder(run_dir)

    _, mapping = load_brahmi_mapping_csv()

    for char_no, (x1, y1, x2, y2) in enumerate(manual_boxes, start=1):
        key = _prediction_key(line_no, char_no)
        prediction = predictions.get(key)

        if prediction is None:
            continue

        final_label = prediction["label"]
        mapping_item = map_class_to_glyph(final_label, mapping)

        replacement.append({
            "global_index": None,
            "line": int(line_no),
            "char": int(char_no),
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "label": final_label,
            "cnn_label": final_label,
            "confidence": prediction["confidence"],
            "recognition_status": recognition_status(
                prediction["confidence"]
            ),
            "dataset_label": None,
            "dataset_similarity": 0.0,
            "dataset_top": [],
            "decision_source": "ALL_CLASS_CNN",
            "brahmi_glyph": mapping_item.get("glyph", ""),
            "unicode_hex": mapping_item.get("unicode_hex", ""),
            "unicode_character_name": mapping_item.get(
                "character_name", ""
            ),
            "raw": crop,
            "cnn": cv2.imread(
                str(preprocessed_dir / f"{key}_preprocessed.png"),
                cv2.IMREAD_GRAYSCALE,
            ),
            "top5": prediction["top5"],
            "extracted_path": str(
                extracted_dir / f"{key}.png"
            ),
            "preprocessed_path": prediction["preprocessed_path"],
        })

    other_records = [
        r for r in result["records"]
        if int(r["line"]) != int(line_no)
    ]

    merged = other_records + replacement
    merged.sort(
        key=lambda r: (
            int(r["line"]),
            int(r.get("x", 0)),
        )
    )

    for index, record in enumerate(merged, start=1):
        record["global_index"] = index

    return merged





# ============================================================
# Unified 3-script integration paths
# ============================================================
MODEL_DIR = MODELS if MODELS.exists() else (ROOT / "Model")
TAMIL_DIR = ROOT / "Tamil"
DEV_DIR = ROOT / "Devanagari"
MAPPING_DIR = ROOT / "Mapping"
UNIFIED_OUTPUT = ROOT / "Unified_Output"
UNIFIED_OUTPUT.mkdir(parents=True, exist_ok=True)

@contextmanager
def push_sys_path(path):
    path = str(Path(path).resolve())
    old = list(sys.path)
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = old

COMMON_GLOBALS = {
    "__name__": "__neolipi_pipeline__",
    "__file__": "",
    "os": os,
    "sys": sys,
    "json": json,
    "csv": csv,
    "glob": glob,
    "io": io,
    "time": time,
    "tempfile": tempfile,
    "Path": Path,
    "cv2": cv2,
    "np": np,
    "pd": pd,
    "Image": Image,
    "st": st,
}

def load_functions_only(source_path, namespace_name, extra_globals=None):
    source_path = Path(source_path)
    tree = ast.parse(source_path.read_text(encoding="utf-8"),
                     filename=str(source_path))
    ns = dict(COMMON_GLOBALS)
    ns["__file__"] = str(source_path)
    ns["__name__"] = namespace_name
    if extra_globals:
        ns.update(extra_globals)

    import_nodes = [
        n for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    with push_sys_path(source_path.parent):
        for node in import_nodes:
            try:
                exec(
                    compile(ast.Module([node], type_ignores=[]),
                            str(source_path), "exec"),
                    ns, ns
                )
            except Exception:
                pass

    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except Exception:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    ns[target.id] = value

    defs = [
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    with push_sys_path(source_path.parent):
        for node in defs:
            try:
                exec(
                    compile(ast.Module([node], type_ignores=[]),
                            str(source_path), "exec"),
                    ns, ns
                )
            except Exception as exc:
                ns[f"__load_error_{getattr(node, 'name', 'unknown')}"] = str(exc)
    return ns

# ============================================================
# Tamil pipeline (source algorithms preserved)
# ============================================================
_tamil = None
TAMIL_SOURCE = None
TAMIL_MODEL = MODEL_DIR / "verification_model.h5"
TAMIL_DATASET = Path(os.getenv(
    "TAMIL_DATASET_DIR",
    r"D:\MAJOR_DATASET\character_dataset\tamil_inscription_dataset"
))
TAMIL_STAGE_DIR = UNIFIED_OUTPUT / "tamil_characters"
TAMIL_STAGE_DIR.mkdir(parents=True, exist_ok=True)

def ensure_tamil_loaded():
    global _tamil, TAMIL_SOURCE
    if _tamil is not None:
        return
    candidates = [
        TAMIL_DIR / "tamil_final.py",
        TAMIL_DIR / "app.py",
        TAMIL_DIR / "app_final.py",
        TAMIL_DIR / "app_final3.py",
        TAMIL_DIR / "app_final3(1).py",
    ]
    TAMIL_SOURCE = next((p for p in candidates if p.exists()), None)
    if TAMIL_SOURCE is None:
        raise FileNotFoundError(
            "Tamil source file not found. Expected a working Tamil app "
            "inside the Tamil folder."
        )
    _tamil = load_functions_only(TAMIL_SOURCE, "neolipi_tamil_pipeline")
    _tamil["STAGE4_CROP_DIR"] = str(TAMIL_STAGE_DIR)

def load_h5_model_compat(model_path):
    """Load the legacy Tamil H5 model without deserializing its old config.

    The Tamil model was saved as an older Keras H5 model.  Its config contains
    initializer fields such as input_axes/output_axes that current Keras 3.x
    cannot deserialize.  The architecture is known from the saved config, so
    we rebuild that architecture explicitly and load the original trained
    weights from the H5 file.  This preserves the trained weights and avoids
    the incompatible JSON/model-config deserialization path.
    """
    import tensorflow as tf

    model_path = Path(model_path)

    if model_path.suffix.lower() != ".h5":
        return tf.keras.models.load_model(str(model_path), compile=False)

    # First try the normal loader. This keeps compatibility if the user's
    # local copy is already readable by the installed Keras version.
    try:
        return tf.keras.models.load_model(str(model_path), compile=False)
    except Exception:
        pass

    # Exact architecture recorded in the legacy Tamil verification_model.h5:
    # Input(64x64x1) -> Conv32+BN+Pool -> Conv64+BN+Pool ->
    # Conv128+BN+Pool -> Flatten -> Dense128 -> Dropout(0.4) -> Dense7.
    # Build it explicitly so Keras never has to deserialize the old config.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(64, 64, 1),
            name="input_layer",
        ),
        tf.keras.layers.Conv2D(
            32, (3, 3), padding="same", activation="relu",
            name="conv2d",
        ),
        tf.keras.layers.BatchNormalization(name="batch_normalization"),
        tf.keras.layers.MaxPooling2D(name="max_pooling2d"),
        tf.keras.layers.Conv2D(
            64, (3, 3), padding="same", activation="relu",
            name="conv2d_1",
        ),
        tf.keras.layers.BatchNormalization(name="batch_normalization_1"),
        tf.keras.layers.MaxPooling2D(name="max_pooling2d_1"),
        tf.keras.layers.Conv2D(
            128, (3, 3), padding="same", activation="relu",
            name="conv2d_2",
        ),
        tf.keras.layers.BatchNormalization(name="batch_normalization_2"),
        tf.keras.layers.MaxPooling2D(name="max_pooling2d_2"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(128, activation="relu", name="dense"),
        tf.keras.layers.Dropout(0.4, name="dropout"),
        tf.keras.layers.Dense(7, activation="softmax", name="dense_1"),
    ], name="sequential")

    # Build variables before loading the legacy H5 weights.
    model.build((None, 64, 64, 1))
    model.load_weights(str(model_path))
    return model


# The legacy Tamil verification_model.h5 was trained with exactly these
# seven folders.  Keras/flow_from_directory uses the folder names in
# alphabetical order for the class indices, so the model output indices are:
#   0 -> 030_ka
#   1 -> 126_pa
#   2 -> 130_pu
#   3 -> 174_la
#   4 -> 178_lu
#   5 -> 190_vu
#   6 -> 210_la_retro
#
# Do NOT use Mapping/model_index_map_247.json here.  That mapping contains
# 247 classes and belongs to a different Tamil model.
TAMIL_MODEL_CLASSES = [
    "030_ka",
    "126_pa",
    "130_pu",
    "174_la",
    "178_lu",
    "190_vu",
    "210_la_retro",
]


@st.cache_resource(show_spinner=False)
def load_tamil_resources():
    ensure_tamil_loaded()
    if not TAMIL_MODEL.exists():
        raise FileNotFoundError(
            f"Tamil model not found: {TAMIL_MODEL}"
        )

    model = load_h5_model_compat(TAMIL_MODEL)

    # Keep the class order exactly aligned with the 7-output Dense layer.
    classes = list(TAMIL_MODEL_CLASSES)

    output_count = int(model.output_shape[-1])
    if output_count != len(classes):
        raise ValueError(
            f"Tamil model/class mismatch: model outputs {output_count} classes, "
            f"but the configured Tamil mapping contains {len(classes)} classes."
        )

    return model, classes

def run_tamil_pipeline(image_bgr):
    ensure_tamil_loaded()
    model, classes = load_tamil_resources()

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blackhat, thresh, clean_mask, final_output = (
        _tamil["preprocess_morphology_stages"](gray)
    )

    with push_sys_path(TAMIL_DIR):
        from dline_segmentation_3_good_dup_2 import extract_line_strips
        line_crops, debug_img, _ = extract_line_strips(
            thresh, is_preprocessed=True
        )

    if not line_crops:
        raise ValueError("No Tamil inscription lines were detected.")

    all_raw, all_corrected, all_predictions = [], [], []

    # Save every automatically extracted Tamil character for this run.
    tamil_run_dir = UNIFIED_OUTPUT / "tamil_runs" / f"run_{int(time.time())}"
    tamil_extracted_dir = tamil_run_dir / "extracted"
    tamil_extracted_dir.mkdir(parents=True, exist_ok=True)

    for line_item in line_crops:
        line_idx = line_item["line_idx"]
        crop_image = line_item["image"]
        if crop_image is None or crop_image.size == 0:
            continue

        seg_img, character_crops, predictions = (
            _tamil["segment_and_predict_characters"](
                crop_image,
                cnn_model=model,
                classes=classes,
                selected_script="Tamil",
                line_idx=line_idx,
            )
        )

        raw_line = "".join(p[0] for p in predictions)
        corrected_line = _tamil["correct_linguistic_errors_normalized"](raw_line)
        all_raw.append(raw_line)
        all_corrected.append(corrected_line)
        # Persist the exact character crops returned by the original
        # Tamil character segmentation/recognition pipeline.
        for char_no, crop in enumerate(character_crops, start=1):
            cv2.imwrite(
                str(
                    tamil_extracted_dir
                    / f"line_{int(line_idx):02d}_char_{int(char_no):03d}.png"
                ),
                crop,
            )

        all_predictions.append({
            "line": line_idx,
            "segmented": seg_img,
            "crops": character_crops,
            "predictions": predictions,
        })

    with push_sys_path(TAMIL_DIR):
        from translation_utils import translate_batch_tamil_to_kannada
        kannada = translate_batch_tamil_to_kannada(all_corrected)

    return {
        "gray": gray,
        "blackhat": blackhat,
        "binary": thresh,
        "clean_mask": clean_mask,
        "final": final_output,
        "line_debug": debug_img,
        "line_count": len(line_crops),
        "lines": all_predictions,
        "line_crops": line_crops,
        "raw_text": "\n".join(all_raw),
        "corrected_text": "\n".join(all_corrected),
        "kannada_text": "\n".join(kannada),
        "run_dir": str(tamil_run_dir),
        "extracted_dir": str(tamil_extracted_dir),
    }

def render_tamil_manual_editor(result):
    """Manual Tamil segmentation with canvas when available, native editor otherwise."""
    ensure_tamil_loaded()
    model, classes = load_tamil_resources()

    with push_sys_path(TAMIL_DIR):
        from dline_segmentation_3_good_dup_2 import extract_line_strips
        line_crops, _, _ = extract_line_strips(
            result["binary"], is_preprocessed=True
        )

    st.subheader("4. Manual Segmentation Correction")
    st.write(
        "Draw one rectangle around each Tamil character. "
        "If the canvas package is unavailable, edit X/Y/Width/Height below."
    )

    manual_outputs = []
    for line_item in line_crops:
        line_idx = int(line_item["line_idx"])
        crop_image = line_item["image"]
        if crop_image is None or crop_image.size == 0:
            continue

        st.markdown(f"### Manual Segmentation — Line {line_idx}")
        h, w = crop_image.shape[:2]
        preview = (
            cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
            if crop_image.ndim == 3
            else cv2.cvtColor(crop_image, cv2.COLOR_GRAY2RGB)
        )

        boxes = []
        if DRAWABLE_CANVAS_AVAILABLE and st_canvas is not None:
            display_width = min(900, max(450, w))
            display_height = max(120, int(round(h * display_width / max(w, 1))))
            bg = Image.fromarray(preview).resize(
                (display_width, display_height),
                Image.Resampling.NEAREST
            )
            canvas = st_canvas(
                fill_color="rgba(0,255,0,0.12)",
                stroke_color="#00ff00",
                stroke_width=2,
                background_image=bg,
                update_streamlit=True,
                height=display_height,
                width=display_width,
                drawing_mode="rect",
                        key=f"unified_tamil_canvas_{line_idx}",
            )
            if canvas.json_data:
                sx = w / float(display_width)
                sy = h / float(display_height)
                for obj in canvas.json_data.get("objects", []):
                    if obj.get("type") != "rect":
                        continue
                    x = max(0, int(round(float(obj.get("left", 0)) * sx)))
                    y = max(0, int(round(float(obj.get("top", 0)) * sy)))
                    bw = max(1, int(round(float(obj.get("width", 0)) * sx)))
                    bh = max(1, int(round(float(obj.get("height", 0)) * sy)))
                    if x < w and y < h:
                        boxes.append((x, y, min(bw, w-x), min(bh, h-y)))
        else:
            st.image(preview, caption=f"Tamil line {line_idx}", use_container_width=True)
            key = f"unified_tamil_boxes_{line_idx}"
            if key not in st.session_state:
                st.session_state[key] = pd.DataFrame([
                    {"x": 0, "y": 0, "w": max(1, min(50, w)), "h": max(1, h)}
                ])
            edited = st.data_editor(
                st.session_state[key],
                num_rows="dynamic",
                hide_index=True,
                use_container_width=True,
                column_config={
                    "x": st.column_config.NumberColumn("X", min_value=0, step=1),
                    "y": st.column_config.NumberColumn("Y", min_value=0, step=1),
                    "w": st.column_config.NumberColumn("Width", min_value=1, step=1),
                    "h": st.column_config.NumberColumn("Height", min_value=1, step=1),
                },
                key=f"unified_tamil_editor_{line_idx}",
            )
            st.session_state[key] = edited
            for _, row in edited.iterrows():
                try:
                    x, y, bw, bh = map(int, (row["x"], row["y"], row["w"], row["h"]))
                except Exception:
                    continue
                if 0 <= x < w and 0 <= y < h:
                    boxes.append((x, y, min(max(1,bw), w-x), min(max(1,bh), h-y)))

        overlay = preview.copy()
        for i, (x, y, bw, bh) in enumerate(boxes, start=1):
            cv2.rectangle(overlay, (x, y), (x+bw-1, y+bh-1), (0,255,0), 2)
            cv2.putText(overlay, str(i), (x, max(15,y-3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        st.image(overlay, caption=f"Current manual boxes — Line {line_idx}",
                 use_container_width=True)

        if st.button(
            f"Apply Corrections — Tamil Line {line_idx}",
            type="primary",
            key=f"unified_apply_tamil_{line_idx}",
            use_container_width=True,
        ):
            if not boxes:
                st.error("Draw/add at least one character box.")
                continue
            seg_img, crops, preds = _tamil["process_manual_boxes_and_predict"](
                crop_image, boxes, cnn_model=model, classes=classes,
                selected_script="Tamil", line_idx=line_idx
            )

            manual_run_dir = Path(result.get("run_dir") or (
                UNIFIED_OUTPUT / "tamil_runs" / f"manual_{int(time.time())}"
            ))
            manual_extracted_dir = manual_run_dir / "manual_extracted"
            manual_extracted_dir.mkdir(parents=True, exist_ok=True)
            for char_no, crop in enumerate(crops, start=1):
                cv2.imwrite(
                    str(
                        manual_extracted_dir
                        / f"line_{int(line_idx):02d}_char_{int(char_no):03d}.png"
                    ),
                    crop,
                )

            raw = "".join(p[0] for p in preds)
            corrected = _tamil["correct_linguistic_errors_normalized"](raw)
            manual_outputs.append((line_idx, seg_img, crops, preds, raw, corrected))

    if manual_outputs:
        all_corrected = []
        for line_idx, seg_img, crops, preds, raw, corrected in manual_outputs:
            st.image(
                cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
                if seg_img.ndim == 3 else seg_img,
                caption=f"Applied segmentation — Line {line_idx}",
                use_container_width=True,
            )
            cols = st.columns(min(6, max(1, len(crops))))
            for col, crop, pred in zip(cols, crops, preds):
                with col:
                    st.image(crop, use_container_width=True)
                    st.markdown(f"**{pred[0]}**")
                    st.caption(f"{pred[4]} | {pred[1]*100:.1f}%")
            all_corrected.append(corrected)

        result["raw_text"] = "\n".join(x[4] for x in manual_outputs)
        result["corrected_text"] = "\n".join(all_corrected)
        with push_sys_path(TAMIL_DIR):
            from translation_utils import translate_batch_tamil_to_kannada
            result["kannada_text"] = "\n".join(
                translate_batch_tamil_to_kannada(all_corrected)
            )

# ============================================================
# Devanagari / FastOCR pipeline
# ============================================================
_dev = None
DEV_SOURCE = None

# IMPORTANT:
# The working Devanagari project stores its environment file as:
#     Devanagari/env
# not as the root .env file.
#
# app_final28 was reading os.getenv() before loading that file,
# which can make FastOCR fail even though the Devanagari pipeline
# works correctly in its standalone application.
DEV_ENV_FILE = DEV_DIR / "env"

try:
    if DEV_ENV_FILE.exists():
        load_dotenv(dotenv_path=DEV_ENV_FILE, override=False)
except Exception:
    pass

DEV_API_KEY = os.getenv("FASTOCR_API_KEY", "").strip()
DEV_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()

def ensure_dev_loaded():
    global _dev, DEV_SOURCE
    if _dev is not None:
        return
    candidates = [
        DEV_DIR / "neolipi2.py",
        DEV_DIR / "neolipi2(1).py",
        DEV_DIR / "app.py",
    ]
    if DEV_DIR.exists():
        candidates += sorted(DEV_DIR.glob("neolipi2*.py"), key=lambda p: p.stat().st_mtime, reverse=True)
    DEV_SOURCE = next((p for p in candidates if p.exists()), None)
    if DEV_SOURCE is None:
        raise FileNotFoundError(
            "Devanagari source file not found. Expected the working "
            "Devanagari app inside the Devanagari folder."
        )
    _dev = load_functions_only(DEV_SOURCE, "neolipi_devanagari_pipeline")

    # Pass the explicitly loaded key into the extracted original
    # Devanagari functions. This keeps the original FastOCR workflow:
    # normalized image -> PDF -> FastOCR document -> OCR text.
    _dev["FASTOCR_API_KEY"] = DEV_API_KEY
    _dev["FASTOCR_BASE_URL"] = "https://api.fastocr.org/v1"

def translate_sanskrit_to_kannada_secure(text):
    if not text or not text.strip():
        return "Translation error: No Sanskrit text provided."
    if not DEV_GEMINI_KEY:
        return "Translation unavailable: GEMINI_API_KEY is not configured."
    try:
        from google import genai
        client = genai.Client(api_key=DEV_GEMINI_KEY)
        cleaned_text = "\n".join(
            " ".join(line.strip().split())
            for line in text.splitlines() if line.strip()
        )
        system_instruction = """
You are the Sanskrit-to-Kannada translation engine of the NeoLipi
ancient inscription recognition system. Translate Sanskrit OCR into
meaningful modern Kannada. Preserve meaning, historical context,
names, places and terminology. Correct only obvious OCR mistakes.
Do not invent information. Return ONLY Kannada.
"""
        prompt = f"Translate this Sanskrit OCR text into modern Kannada:\n\n{cleaned_text}"
        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            system_instruction=system_instruction,
            input=prompt,
            generation_config={
                "seed": 12345,
                "thinking_level": "low",
                "max_output_tokens": 300,
            },
        )
        return interaction.output_text.strip() or "Translation error: Empty response."
    except Exception as exc:
        return f"Translation error: {exc}"

def run_devanagari_pipeline(image_bgr):
    ensure_dev_loaded()
    if not DEV_API_KEY:
        raise RuntimeError(
            "FASTOCR_API_KEY is missing. The Devanagari pipeline expects "
            f"the key in: {DEV_ENV_FILE}"
        )

    processed = _dev["preprocess_image"](image_bgr, blur_size=71)
    temp_dir = Path(tempfile.mkdtemp(prefix="neolipi_dev_"))
    image_path = temp_dir / "neolipi_preprocessed.png"
    pdf_path = temp_dir / "neolipi_preprocessed.pdf"
    cv2.imwrite(str(image_path), processed["normalized"])
    _dev["image_to_pdf"](str(image_path), str(pdf_path))
    text, raw_result = _dev["run_fastocr"](str(pdf_path))

    return {
        "gray": processed["gray"],
        "denoised": processed["denoised"],
        "enhanced": processed["enhanced"],
        "normalized": processed["normalized"],
        "ocr_text": text,
        "raw_fastocr": raw_result,
        "kannada_text": translate_sanskrit_to_kannada_secure(text),
        "manual_crops": [],
        "manual_run_dir": None,
    }

def render_devanagari_manual_editor(result):
    st.subheader("4. Manual Segmentation Correction")
    st.write(
        "The original Devanagari pipeline performs OCR on the complete "
        "normalized image. This editor additionally lets you manually "
        "extract character/region crops for inspection and saving."
    )

    image = result["normalized"]
    if image is None or image.size == 0:
        st.warning("No normalized Devanagari image is available.")
        return

    h, w = image.shape[:2]
    preview = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    boxes = []

    if DRAWABLE_CANVAS_AVAILABLE and st_canvas is not None:
        display_width = min(1000, max(500, w))
        display_height = max(160, int(round(h * display_width / max(w, 1))))
        bg = Image.fromarray(preview).resize(
            (display_width, display_height), Image.Resampling.NEAREST
        )
        canvas = st_canvas(
            fill_color="rgba(0,255,0,0.12)",
            stroke_color="#00ff00",
            stroke_width=2,
            background_image=bg,
            update_streamlit=True,
            height=display_height,
            width=display_width,
            drawing_mode="rect",
                key="unified_dev_canvas",
        )
        if canvas.json_data:
            sx = w / float(display_width)
            sy = h / float(display_height)
            for obj in canvas.json_data.get("objects", []):
                if obj.get("type") != "rect":
                    continue
                x = max(0, int(round(float(obj.get("left",0))*sx)))
                y = max(0, int(round(float(obj.get("top",0))*sy)))
                bw = max(1, int(round(float(obj.get("width",0))*sx)))
                bh = max(1, int(round(float(obj.get("height",0))*sy)))
                if x < w and y < h:
                    boxes.append((x, y, min(bw,w-x), min(bh,h-y)))
    else:
        st.image(preview, caption="Normalized Devanagari image",
                 use_container_width=True)
        key = "unified_dev_boxes"
        if key not in st.session_state:
            st.session_state[key] = pd.DataFrame([
                {"x": 0, "y": 0, "w": max(1, min(80,w)), "h": max(1,h)}
            ])
        edited = st.data_editor(
            st.session_state[key],
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            column_config={
                "x": st.column_config.NumberColumn("X", min_value=0, step=1),
                "y": st.column_config.NumberColumn("Y", min_value=0, step=1),
                "w": st.column_config.NumberColumn("Width", min_value=1, step=1),
                "h": st.column_config.NumberColumn("Height", min_value=1, step=1),
            },
            key="unified_dev_editor",
        )
        st.session_state[key] = edited
        for _, row in edited.iterrows():
            try:
                x,y,bw,bh = map(int,(row["x"],row["y"],row["w"],row["h"]))
            except Exception:
                continue
            if 0 <= x < w and 0 <= y < h:
                boxes.append((x,y,min(max(1,bw),w-x),min(max(1,bh),h-y)))

    overlay = preview.copy()
    for i,(x,y,bw,bh) in enumerate(boxes,1):
        cv2.rectangle(overlay,(x,y),(x+bw-1,y+bh-1),(0,255,0),2)
        cv2.putText(overlay,str(i),(x,max(15,y-3)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,cv2.LINE_AA)
    st.image(overlay, caption="Current manual boxes", use_container_width=True)

    if st.button("Apply Corrections — Devanagari",
                 type="primary", use_container_width=True,
                 key="unified_apply_dev"):
        if not boxes:
            st.error("Add at least one character/region box.")
            return
        run_dir = UNIFIED_OUTPUT / "devanagari_runs" / f"run_{int(time.time())}"
        extracted_dir = run_dir / "extracted"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for i,(x,y,bw,bh) in enumerate(boxes,1):
            crop = image[y:y+bh, x:x+bw].copy()
            path = extracted_dir / f"char_{i:03d}.png"
            cv2.imwrite(str(path), crop)
            saved.append((path,crop))
        result["manual_crops"] = saved
        result["manual_run_dir"] = str(run_dir)
        st.success(f"Saved {len(saved)} Devanagari manual character crops.")
        st.rerun()

def render_tamil_results(result, segmentation_mode):
    st.header("Tamil Results")
    p1,p2,p3,p4 = st.columns(4)
    p1.image(result["gray"], caption="Grayscale", use_container_width=True)
    p2.image(result["blackhat"], caption="Blackhat", use_container_width=True)
    p3.image(result["binary"], caption="Binary Mask", use_container_width=True)
    p4.image(result["final"], caption="Clean Inverted Mask", use_container_width=True)

    st.subheader("3. Character Segmentation")
    st.image(
        cv2.cvtColor(result["line_debug"], cv2.COLOR_BGR2RGB),
        caption=f"Detected {result['line_count']} Tamil line(s) — green character boxes",
        use_container_width=True,
    )

    st.subheader("4. Character Extraction")
    if result.get("extracted_dir"):
        st.caption(
            f"Automatically extracted Tamil characters are saved in: "
            f"{result['extracted_dir']}"
        )
    for line in result["lines"]:
        st.markdown(f"**Line {line['line']}**")
        cols = st.columns(min(6, max(1, len(line["crops"]))))
        for col,crop,pred in zip(cols,line["crops"],line["predictions"]):
            with col:
                st.image(crop, use_container_width=True)
                st.markdown(f"**{pred[0]}**")
                st.caption(f"{pred[4]} | {pred[1]*100:.1f}%")

    if segmentation_mode == "Manual (Tamil)":
        render_tamil_manual_editor(result)

    st.subheader("5. Raw Tamil OCR")
    st.text_area("Raw character output", result["raw_text"], height=140)
    st.subheader("6. Corrected Tamil")
    st.text_area("Corrected text", result["corrected_text"], height=140)
    st.subheader("7. Kannada Translation")
    st.text_area("Kannada", result["kannada_text"], height=160)

def render_devanagari_results(result, segmentation_mode):
    st.header("Devanagari Results")
    p1,p2,p3,p4 = st.columns(4)
    p1.image(result["gray"], caption="Grayscale", use_container_width=True)
    p2.image(result["denoised"], caption="Bilateral Denoising", use_container_width=True)
    p3.image(result["enhanced"], caption="CLAHE", use_container_width=True)
    p4.image(result["normalized"], caption="Background Normalization", use_container_width=True)

    st.subheader("3. Character Segmentation")
    st.info(
        "The original Devanagari pipeline does not perform automatic "
        "character segmentation. Manual segmentation below is available "
        "for selecting and saving character/region crops."
    )

    if segmentation_mode == "Manual (Devanagari)":
        render_devanagari_manual_editor(result)
    else:
        st.image(
            result["normalized"],
            caption="Normalized image sent to FastOCR",
            use_container_width=True,
        )

    st.subheader("4. Character Extraction")
    crops = result.get("manual_crops", [])
    if crops:
        cols = st.columns(min(6, len(crops)))
        for col,(path,crop) in zip(cols,crops):
            with col:
                st.image(crop, caption=path.name, use_container_width=True)
        st.caption(f"Saved in: {result.get('manual_run_dir','')}")
    else:
        st.info("No manual Devanagari character crops have been saved yet.")

    st.subheader("5. Sanskrit / Devanagari OCR")
    st.text_area("FastOCR result", result["ocr_text"], height=220)

    st.subheader("6. Kannada Translation")
    st.text_area("Kannada", result["kannada_text"], height=220)


def normalize_line_boxes(detected_lines, shape):
    """Normalize and sort line boxes into standard (x1, y1, x2, y2) format within image boundaries."""
    H, W = shape[:2]
    normalized = []
    if not detected_lines:
        return normalized

    for line in detected_lines:
        try:
            if len(line) >= 4:
                x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])
            elif len(line) == 2:
                (x1, y1), (x2, y2) = line
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            else:
                continue
        except Exception:
            continue

        # Ensure proper bounds and ordering
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(x1 + 1, min(W, x2))
        y2 = max(y1 + 1, min(H, y2))

        normalized.append((x1, y1, x2, y2))

    # Sort lines top-to-bottom by their vertical position (y1)
    normalized.sort(key=lambda b: b[1])
    return normalized
# ============================================================
# Unified routing
# ============================================================

def run_brahmi_pipeline_exact(uploaded_path, script_result):
    """NeoLipi pipeline using the working all-class CNN mapping model."""
    # This function is the exact Brahmi pipeline from the uploaded Brahmi app.
    # 2. BINARIZATION - unchanged.
    BRAHMI_DIR = ROOT / "Brahmi"
    if not BRAHMI_DIR.exists():
        raise FileNotFoundError(f"Brahmi folder was not found: {BRAHMI_DIR}")
    bin_candidates = [BRAHMI_DIR / "binarization.py"] + sorted(
        BRAHMI_DIR.glob("binarization*.py"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    bin_path = next((p for p in bin_candidates if p.exists()), None)
    if bin_path is None:
        raise FileNotFoundError(
            f"No Brahmi binarization*.py file was found in {BRAHMI_DIR}."
        )
    bin_module = load_module(bin_path, "neolipi_brahmi_binarization")
    if not callable(getattr(bin_module, "process", None)):
        raise AttributeError(f"Selected Brahmi binarization file has no process(): {bin_path}")
    with working_directory(BRAHMI_DIR):
        bin_module.process(str(uploaded_path))

    binary_path = BRAHMI_DIR / "outputs" / "08_FINAL_BINARY.png"
    if not binary_path.exists():
        raise FileNotFoundError(
            "Binarization completed but outputs/08_FINAL_BINARY.png "
            "was not created."
        )

    binary = load_binary(binary_path)

    result = {
        "script": script_result,
        "binary": binary,
        "line_boxes": [],
        "records": [],
        "mapping_source": None,
        "run_dir": None,
        "extracted_dir": None,
        "preprocessed_dir": None,
    }

    # 3. SEGMENTATION - existing working segmentation module unchanged.
    seg_candidates = [ROOT / "Brahmi" / "character_segmentation_1dup.py"] + sorted(
        (ROOT / "Brahmi").glob("character_segmentation*.py"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    segmentation_path = None
    segmentation = None
    for candidate in seg_candidates:
        if not candidate.exists():
            continue
        try:
            candidate_module = load_module(candidate, "neolipi_segmentation_probe")
            if all(hasattr(candidate_module, n) for n in ("load_binary", "detect_lines", "segment_line", "extract_character")):
                segmentation_path = candidate
                segmentation = candidate_module
                break
        except Exception:
            continue
    if segmentation is None:
        raise FileNotFoundError(
            "No compatible Brahmi character_segmentation*.py was found. "
            "Required API: load_binary(), detect_lines(), segment_line(), extract_character()."
        )

    detected = segmentation.detect_lines(binary)

    if isinstance(detected, tuple) and len(detected) == 3:
        detected_lines = detected[0]
    else:
        detected_lines = detected

    result["line_boxes"] = normalize_line_boxes(
        detected_lines,
        binary.shape,
    )

    if not result["line_boxes"]:
        raise ValueError("No inscription lines were detected.")

    # 4. Exact working all-class CNN + class file.
    model, class_list = load_all_class_cnn()
    mapping_path, mapping = load_brahmi_mapping_csv()
    result["mapping_source"] = str(mapping_path)

    # 5. Create this inscription's extraction/preprocessing folders.
    run_dir, extracted_dir, preprocessed_dir = create_character_run_dir()
    result["run_dir"] = str(run_dir)
    result["extracted_dir"] = str(extracted_dir)
    result["preprocessed_dir"] = str(preprocessed_dir)

    # ------------------------------------------------------------
    # STEP A: EXISTING SEGMENTATION -> SAVE ALL RAW CHARACTERS
    # ------------------------------------------------------------
    components_by_line = {}

    for line_no, line_box in enumerate(
        result["line_boxes"],
        start=1,
    ):
        lx1, ly1, lx2, ly2 = line_box
        line = binary[ly1:ly2, lx1:lx2].copy()

        segmented = segmentation.segment_line(line)
        if isinstance(segmented, tuple) and len(segmented) == 2:
            components = segmented[1]
        else:
            components = segmented

        if components is None:
            components = []

        components_by_line[line_no] = []

        for char_no, component in enumerate(components, start=1):
            crop = segmentation.extract_character(
                binary,
                line_box,
                component,
            )

            extracted_path = save_extracted_character(
                run_dir,
                line_no,
                char_no,
                crop,
            )

            x, y, w, h, _ = component
            components_by_line[line_no].append({
                "char": char_no,
                "component": component,
                "crop": crop,
                "extracted_path": extracted_path,
                "x": int(lx1 + x),
                "y": int(ly1 + y),
                "w": int(w),
                "h": int(h),
            })

    # ------------------------------------------------------------
    # STEP B: PREPROCESS THE WHOLE EXTRACTED FOLDER
    # ------------------------------------------------------------
    preprocess_extracted_folder(run_dir)

    # ------------------------------------------------------------
    # STEP C: MAPPING MUST USE THE PREPROCESSED FOLDER
    # ------------------------------------------------------------
    predictions = predict_preprocessed_folder(run_dir)

    global_index = 1

    for line_no in sorted(components_by_line):
        for item in components_by_line[line_no]:
            char_no = item["char"]
            key = _prediction_key(line_no, char_no)
            prediction = predictions.get(key)

            if prediction is None:
                continue

            final_label = prediction["label"]
            mapping_item = map_class_to_glyph(
                final_label,
                mapping,
            )

            preprocessed_path = Path(
                prediction["preprocessed_path"]
            )
            cnn_input = cv2.imread(
                str(preprocessed_path),
                cv2.IMREAD_GRAYSCALE,
            )

            result["records"].append({
                "global_index": global_index,
                "line": line_no,
                "char": char_no,
                "x": item["x"],
                "y": item["y"],
                "w": item["w"],
                "h": item["h"],
                "label": final_label,
                "cnn_label": final_label,
                "confidence": prediction["confidence"],
                "recognition_status": recognition_status(
                    prediction["confidence"]
                ),
                "dataset_label": None,
                "dataset_similarity": 0.0,
                "dataset_top": [],
                "decision_source": "ALL_CLASS_CNN",
                "brahmi_glyph": mapping_item.get("glyph", ""),
                "unicode_hex": mapping_item.get("unicode_hex", ""),
                "unicode_character_name": mapping_item.get(
                    "character_name", ""
                ),
                "raw": item["crop"].copy(),
                "cnn": cnn_input,
                "top5": prediction["top5"],
                "extracted_path": str(item["extracted_path"]),
                "preprocessed_path": str(preprocessed_path),
            })

            global_index += 1

    return result

def run_pipeline(uploaded):
    """
    One classifier -> exactly one script-specific pipeline.

    Brahmi: exact uploaded Brahmi pipeline is used.
    Tamil: original Tamil preprocessing/segmentation/CNN pipeline.
    Devanagari: original preprocessing/FastOCR pipeline.
    """
    uploaded_path = save_uploaded(uploaded)
    image = cv2.imread(str(uploaded_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read uploaded inscription image.")

    script_result = classify_script(uploaded_path)
    label = str(script_result["label"]).strip()

    if label.lower() == "brahmi":
        return run_brahmi_pipeline_exact(uploaded_path, script_result)

    if label.lower() == "tamil":
        result = run_tamil_pipeline(image)
        result["script"] = script_result
        result["run_dir"] = None
        result["extracted_dir"] = None
        result["preprocessed_dir"] = None
        return result

    if label.lower() == "devanagari":
        result = run_devanagari_pipeline(image)
        result["script"] = script_result
        result["run_dir"] = None
        result["extracted_dir"] = None
        result["preprocessed_dir"] = None
        return result

    raise ValueError(f"Unsupported detected script: {label}")


# ============================================================
# STREAMLIT UI
# ============================================================


# ============================================================
# NeoLipi Control Panel
# ============================================================
with st.sidebar:
    st.header("⚙️ NeoLipi Control Panel")
    segmentation_mode = st.radio(
        "Character segmentation mode",
        [
            "Automatic",
            "Manual (Brahmi)",
            "Manual (Tamil)",
            "Manual (Devanagari)",
        ],
        index=0,
        help=(
            "Choose Automatic for the original script pipeline. "
            "Manual modes provide script-specific character/region correction."
        ),
    )
    st.markdown("---")
    st.caption("Supported scripts")
    st.write("• Brahmi")
    st.write("• Tamil")
    st.write("• Devanagari")
    st.markdown("---")
    override_script = st.checkbox("Manually select script", value=False)
    if override_script:
        st.session_state["manual_script_selection"] = st.selectbox(
            "Select script",
            ["Brahmi", "Tamil", "Devanagari"],
            key="manual_script_selection_box",
        )

st.title(
    "NeoLipi — Ancient Inscription Recognition"
)

st.write(
    "AI-Based Ancient Stone Inscription Recognition System — Brahmi, Tamil and Devanagari"
)

uploaded = st.file_uploader(
    "Upload Stone Inscription Image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "webp",
        "tif",
        "tiff",
    ],
)

# Clear previous results and caches if a new image is uploaded
if "last_uploaded_name" not in st.session_state:
    st.session_state["last_uploaded_name"] = None

if uploaded is not None:
    if st.session_state["last_uploaded_name"] != uploaded.name:
        st.session_state["last_uploaded_name"] = uploaded.name
        if "result" in st.session_state:
            del st.session_state["result"]
        # Clear any leftover line-correction caches from previous runs
        for key in list(st.session_state.keys()):
            if "processed_boxes_line_" in key or "unified_tamil_" in key or "unified_dev_" in key:
                del st.session_state[key]
else:
    st.info(
        "Please upload an inscription image."
    )
    st.stop()

st.subheader(
    "Input Image"
)

st.image(
    uploaded,
    caption=uploaded.name,
    width='stretch',
)


if st.button(
    "Analyze",
    type="primary",
    width='stretch',
):
    try:
        with st.spinner(
            "Running NeoLipi..."
        ):
            if override_script:
                # Manual script override still uses the same script-specific
                # pipelines; it only changes the routing decision.
                manual_script = st.session_state.get(
                    "manual_script_selection",
                    "Brahmi"
                )
                uploaded_path = save_uploaded(uploaded)
                image_for_manual = cv2.imread(
                    str(uploaded_path), cv2.IMREAD_COLOR
                )
                classifier_result = classify_script(uploaded_path)

                if manual_script == "Brahmi":
                    st.session_state["result"] = run_brahmi_pipeline_exact(
                        uploaded_path, classifier_result
                    )
                elif manual_script == "Tamil":
                    rr = run_tamil_pipeline(image_for_manual)
                    rr["script"] = classifier_result
                    rr["script"]["label"] = "Tamil"
                    st.session_state["result"] = rr
                else:
                    rr = run_devanagari_pipeline(image_for_manual)
                    rr["script"] = classifier_result
                    rr["script"]["label"] = "Devanagari"
                    st.session_state["result"] = rr
            else:
                st.session_state["result"] = run_pipeline(uploaded)

    except Exception as exc:
        st.error(
            "Analysis failed."
        )
        st.exception(exc)
        st.stop()


if "result" not in st.session_state:
    st.stop()

result = st.session_state["result"]

# Track segmentation mode changes to clear automatic records when switching to manual mode
if "last_seg_mode" not in st.session_state:
    st.session_state["last_seg_mode"] = segmentation_mode

if st.session_state["last_seg_mode"] != segmentation_mode:
    st.session_state["last_seg_mode"] = segmentation_mode
    if segmentation_mode == "Manual (Brahmi)":
        result["records"] = []
        # Clear canvas caches for a fresh start
        for key in list(st.session_state.keys()):
            if "manual_canvas_line_" in key or "processed_boxes_line_" in key:
                del st.session_state[key]
        st.session_state["result"] = result
        st.rerun()
    elif segmentation_mode == "Automatic":
        st.rerun()

# The current run owns its extracted/preprocessed folders.
# Point the manual editor at that run so corrections replace only
# the selected line inside the same inscription run.
if result.get("run_dir"):
    MANUAL_CHARACTERS_DIR = Path(result["run_dir"]) / "extracted"
    MANUAL_CHARACTERS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. SCRIPT CLASSIFICATION
# ============================================================

st.subheader(
    "1. Script Classification"
)

c1, c2 = st.columns(2)

with c1:
    st.success(
        f"Detected Script: "
        f"{result['script']['label']}"
    )

with c2:
    st.metric(
        "Confidence",
        f"{result['script']['confidence'] * 100:.2f}%"
    )


# ============================================================
# 2. BINARY
# ============================================================

# Non-Brahmi scripts leave the Brahmi UI here and use their own
# matching result layout. The Brahmi UI below remains unchanged.
selected_script_for_result = result["script"]["label"]

if selected_script_for_result.lower() == "tamil":
    render_tamil_results(result, segmentation_mode)
    st.stop()

if selected_script_for_result.lower() == "devanagari":
    render_devanagari_results(result, segmentation_mode)
    st.stop()

# ============================================================
# 2. BINARY
# ============================================================
st.subheader("2. Binary Image")
st.image(
    result["binary"],
    caption="Existing Brahmi binarization output",
    width='stretch',
)

# ============================================================
# 3. SEGMENTATION
# ============================================================

st.subheader(
    "3. Character Segmentation"
)

st.write(
    f"Detected lines: "
    f"**{len(result['line_boxes'])}**"
)

line_image = cv2.cvtColor(
    result["binary"],
    cv2.COLOR_GRAY2RGB
)

for i, (
    x1,
    y1,
    x2,
    y2
) in enumerate(
    result["line_boxes"],
    start=1
):
    cv2.rectangle(
        line_image,
        (x1, y1),
        (x2 - 1, y2 - 1),
        (255, 255, 0),
        2,
    )

    cv2.putText(
        line_image,
        f"Line {i}",
        (x1, max(15, y1 - 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )

st.image(
    line_image,
    caption="Detected Inscription Lines",
    width='stretch',
)

# ============================================================
# 4. MANUAL SEGMENTATION CORRECTION
# ============================================================

if segmentation_mode == "Manual (Brahmi)":
    render_manual_segmentation_editor(result)
else:
    st.caption(
        "Manual Brahmi correction is available from the sidebar. "
        "Automatic mode keeps the original Brahmi segmentation unchanged."
    )

# Only show downstream character extraction if we are in Automatic mode OR if manual boxes have been drawn/processed.
if segmentation_mode == "Manual (Brahmi)" and not result.get("records"):
    st.info("👆 Select a line and draw green rectangles on the canvas above to extract characters and view recognition results.")
else:
    # ============================================================
    # 5. CHARACTER EXTRACTION
    # ============================================================
    
    st.subheader(
        "5. Character Extraction"
    )

    box_image = cv2.cvtColor(
        result["binary"],
        cv2.COLOR_GRAY2RGB
    )

    for record in result["records"]:

        x1 = int(record["x"])
        y1 = int(record["y"])

        x2 = (
            x1 +
            int(record["w"])
        )

        y2 = (
            y1 +
            int(record["h"])
        )

        number = int(
            record["global_index"]
        )

        # Character bounding box
        cv2.rectangle(
            box_image,
            (x1, y1),
            (max(x1 + 1, x2 - 1),
             max(y1 + 1, y2 - 1)),
            (0, 255, 0),
            1,
        )

        # Character number
        cv2.putText(
            box_image,
            str(number),
            (
                x1,
                max(12, y1 - 2)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    st.image(
        box_image,
        caption="Character Bounding Boxes",
        width="stretch",
    )

# ============================================================
# CHARACTER QUALITY DIAGNOSTICS
# ============================================================

def character_quality(record):
    """
    Analyze the extracted character crop.

    This does NOT modify the crop.
    It is only a diagnostic tool to identify
    suspicious segmentation results.
    """

    crop = record["raw"]

    if crop is None or crop.size == 0:
        return {
            "status": "INVALID",
            "aspect_ratio": 0.0,
            "foreground_ratio": 0.0,
            "width": 0,
            "height": 0,
        }

    h, w = crop.shape[:2]

    foreground = (
        np.count_nonzero(crop > 127)
    )

    foreground_ratio = (
        foreground / float(crop.size)
    )

    aspect_ratio = (
        w / float(h)
        if h > 0
        else 0.0
    )

    # Conservative diagnostic rules.
    #
    # These DO NOT remove characters.
    # They only mark suspicious crops.

    status = "GOOD"

    if w < 3 or h < 5:
        status = "TOO SMALL"

    elif foreground_ratio < 0.015:
        status = "MOSTLY NOISE"

    elif foreground_ratio > 0.70:
        status = "TOO DENSE"

    elif aspect_ratio > 4.5:
        status = "VERY WIDE"

    elif aspect_ratio < 0.15:
        status = "VERY NARROW"

    return {
        "status": status,
        "aspect_ratio": aspect_ratio,
        "foreground_ratio": foreground_ratio,
        "width": w,
        "height": h,
    }
# ============================================================
# CHARACTER QUALITY DIAGNOSTICS
# ============================================================

def character_quality(record):
    """
    Analyze the extracted character crop.

    This does NOT modify the crop.
    It is only a diagnostic tool to identify
    suspicious segmentation results.
    """

    crop = record["raw"]

    if crop is None or crop.size == 0:
        return {
            "status": "INVALID",
            "aspect_ratio": 0.0,
            "foreground_ratio": 0.0,
            "width": 0,
            "height": 0,
        }

    h, w = crop.shape[:2]

    foreground = (
        np.count_nonzero(crop > 127)
    )

    foreground_ratio = (
        foreground / float(crop.size)
    )

    aspect_ratio = (
        w / float(h)
        if h > 0
        else 0.0
    )

    # Conservative diagnostic rules.
    #
    # These DO NOT remove characters.
    # They only mark suspicious crops.

    status = "GOOD"

    if w < 3 or h < 5:
        status = "TOO SMALL"

    elif foreground_ratio < 0.015:
        status = "MOSTLY NOISE"

    elif foreground_ratio > 0.70:
        status = "TOO DENSE"

    elif aspect_ratio > 4.5:
        status = "VERY WIDE"

    elif aspect_ratio < 0.15:
        status = "VERY NARROW"

    return {
        "status": status,
        "aspect_ratio": aspect_ratio,
        "foreground_ratio": foreground_ratio,
        "width": w,
        "height": h,
    }

# # ============================================================
# # 6. CHARACTER RECOGNITION + BRAHMI UNICODE
# # ============================================================

# st.subheader(
#     "6. Character Recognition + Brahmi Unicode Mapping"
# )

# table_rows = []

# for r in result["records"]:

#     quality = character_quality(r)

#     table_rows.append({

#         "#":
#             r["global_index"],

#         "Line":
#             r["line"],

#         "Char":
#             r["char"],

#         "Predicted Class":
#             r["label"],

#         "Brahmi Unicode":
#             r["brahmi_glyph"]
#             or "N/A",

#         "Unicode Name":
#             r.get(
#                 "unicode_character_name",
#                 ""
#             ),

#         "Confidence":
#             f"{r['confidence'] * 100:.2f}%",

#         "Recognition":
#             r.get(
#                 "recognition_status",
#                 recognition_status(
#                     r["confidence"]
#                 )
#             ),

#         "Segmentation":
#             quality["status"],

#         "W":
#             r["w"],

#         "H":
#             r["h"],
#     })

# table = pd.DataFrame(
#     table_rows
# )

# st.dataframe(
#     table,
#     hide_index=True,
#     width="stretch",
# )

# st.caption(
#     "The Unicode value comes directly from "
#     "brahmi_mapping.csv."
# )
# ============================================================
# 7. CHARACTER INSPECTOR
# ============================================================
if result["records"]:

    st.subheader(
        "7. Character Inspector"
    )

    st.write(
        "Inspect each extracted character before judging "
        "the CNN and Unicode mapping."
    )
    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    quality_results = []

    for record in result["records"]:

        q = character_quality(record)

        quality_results.append(q)

    good_count = sum(
        q["status"] == "GOOD"
        for q in quality_results
    )

    suspicious_count = (
        len(quality_results) -
        good_count
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Characters",
            len(result["records"])
        )

    with c2:
        st.metric(
            "Looks Reasonable",
            good_count
        )

    with c3:
        st.metric(
            "Suspicious Crops",
            suspicious_count
        )

    # --------------------------------------------------------
    # ALL EXTRACTED CHARACTERS
    # --------------------------------------------------------

    st.markdown(
        "### All Extracted Characters"
    )

    st.caption(
        "This is the most important debugging view. "
        "Each tile shows the actual crop sent toward recognition."
    )

    # Six characters per row.
    GRID_COLUMNS = 6

    records = result["records"]

    for start in range(
        0,
        len(records),
        GRID_COLUMNS
    ):

        row_records = records[
            start:start + GRID_COLUMNS
        ]

        columns = st.columns(
            GRID_COLUMNS
        )

        for local_index, record in enumerate(
            row_records
        ):

            global_number = (
                start +
                local_index +
                1
            )

            quality = character_quality(
                record
            )

            with columns[local_index]:

                # ------------------------------------------------
                # Character crop
                # ------------------------------------------------

                # ------------------------------------------------
                # ACTUAL SAVED EXTRACTED CHARACTER
                # ------------------------------------------------
                extracted_image = cv2.imread(
                    str(record["extracted_path"]),
                    cv2.IMREAD_GRAYSCALE
                )

                if extracted_image is not None:
                    st.image(
                        extracted_image,
                        caption=(
                            f"#{global_number}  "
                            f"L{record['line']}-"
                            f"C{record['char']}"
                        ),
                        width=110,
                    )
                else:
                    st.warning(
                        f"Could not load extracted image: "
                        f"{record['extracted_path']}"
                    )

                # ------------------------------------------------
                # Prediction
                # ------------------------------------------------

                st.caption(
                    f"Class: {record['label']}"
                )

                st.caption(
                    f"CNN: {record.get('cnn_label', record['label'])} "
                    f"({record['confidence'] * 100:.1f}%)"
                )

                st.caption(
                    f"Selected by: {record.get('decision_source', 'ALL_CLASS_CNN')}"
                )

                st.caption(
                    f"Recognition: "
                    f"{record.get('recognition_status', 'UNKNOWN')}"
                )

                # ------------------------------------------------
                # Unicode
                # ------------------------------------------------

                glyph = (
                    record["brahmi_glyph"]
                    or "N/A"
                )

                st.markdown(
                    f"**Unicode:** {glyph}"
                )

                # ------------------------------------------------
                # Diagnostic status
                # ------------------------------------------------

                st.caption(
                    f"{quality['status']} | "
                    f"{quality['width']}×"
                    f"{quality['height']}"
                )

# ============================================================
# 8. DIGITAL BRAHMI UNICODE
# ============================================================

st.subheader(
    "8. Digital Brahmi Unicode"
)

digital_lines = []

for line_no in sorted(
    set(
        r["line"]
        for r in result["records"]
    )
):

    current = [
        r
        for r in result["records"]
        if r["line"] == line_no
    ]

    # Left-to-right ordering
    current.sort(
        key=lambda r: (
            int(r.get("x", 0)),
            int(r.get("char", 0))
        )
    )

    line_text = "".join(
        r.get("brahmi_glyph", "")
        for r in current
        if r.get("brahmi_glyph", "")
    )

    digital_lines.append(
        line_text
    )

digital_text = "\n".join(
    digital_lines
)

st.text_area(
    "Digital Brahmi Unicode",
    value=digital_text,
    height=220,
)

if st.button(
    "Copy-ready Unicode Text",
    use_container_width=True
):

    st.code(
        digital_text,
        language=None
    )


# ============================================================
# 9. READABLE / TRANSLITERATED TEXT
# ============================================================

st.subheader(
    "9. Readable / Transliterated Text"
)

readable_text = build_readable_text(
    result["records"]
)

if readable_text:

    st.text_area(
        "Recognized Class Sequence",
        value=readable_text,
        height=180,
    )

else:

    st.warning(
        "No readable text was generated."
    )

# ============================================================
# 10. KANNADA TRANSLATION - CSV MAPPING
# ============================================================

st.subheader(
    "10. Kannada Translation"
)

if readable_text:
    try:
        _, mapping = load_brahmi_mapping_csv()
        kannada_translation, missing_kannada = (
            build_kannada_translation(
                result["records"],
                mapping,
            )
        )

        if missing_kannada:
            st.warning(
                "Kannada mapping is missing for: "
                + ", ".join(missing_kannada)
            )

        st.text_area(
            "Kannada Translation",
            value=kannada_translation,
            height=180,
        )

        st.download_button(
            "Download Kannada Translation",
            data=kannada_translation,
            file_name="kannada_translation.txt",
            mime="text/plain",
            use_container_width=True,
        )

    except Exception as exc:
        st.error(
            "Could not generate Kannada translation from "
            f"brahmi_mapping.csv: {exc}"
        )
else:
    st.info(
        "Kannada translation will be available after character "
        "recognition generates the class sequence."
    )