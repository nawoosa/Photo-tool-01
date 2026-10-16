import os, json, time, uuid, shutil, threading, queue, zipfile, secrets
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

DATA = Path(os.getenv("DATA_DIR", "/data"))
JOBS = DATA / "jobs"
JOBS.mkdir(parents=True, exist_ok=True)
PASSWORD = os.getenv("APP_PASSWORD", "")
MODELS = {"birefnet": "birefnet-general", "massive": "birefnet-massive", "dis": "birefnet-dis",
          "hrsod": "birefnet-hrsod", "bria": "bria-rmbg"}
HQ_SIDE = int(os.getenv("HQ_SIDE", "1500"))

security = HTTPBasic(auto_error=False)


def auth(c: HTTPBasicCredentials = Depends(security)):
    if not PASSWORD or not c or not secrets.compare_digest(c.password, PASSWORD):
        raise HTTPException(401, "Unauthorized", headers={"WWW-Authenticate": "Basic"})


app = FastAPI(dependencies=[Depends(auth)])
Q = queue.Queue()
LOCK = threading.Lock()
_lama = None
_sessions = {}


_inp = {}
SD_NAME = "Uminosachi/realisticVisionV51_v51VAE-inpainting"
SDXL_NAME = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
_hr = None


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def hr_mask(img):
    """BiRefNet HR بدقة 2048"""
    global _hr
    import torch
    from torchvision import transforms
    dev = device()
    if _hr is None:
        from transformers import AutoModelForImageSegmentation
        m = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet_HR", trust_remote_code=True)
        m.to(dev).eval()
        if dev == "cuda":
            m.half()
        _hr = m
    tf = transforms.Compose([transforms.Resize((2048, 2048)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    x = tf(img).unsqueeze(0).to(dev)
    if dev == "cuda":
        x = x.half()
    with torch.no_grad():
        p = _hr(x)[-1].sigmoid().float().cpu()[0].squeeze()
    return transforms.ToPILImage()(p).resize(img.size, Image.LANCZOS)


def get_inpainter(name):
    if name not in _inp:
        from iopaint.model_manager import ModelManager
        _inp[name] = ModelManager(name=name, device=device())
    return _inp[name]


def get_session(key):
    if key not in _sessions:
        from rembg import new_session
        _sessions[key] = new_session(MODELS[key])
    return _sessions[key]


def save(job):
    with LOCK:
        (JOBS / job["id"] / "job.json").write_text(json.dumps(job, ensure_ascii=False))


def load(jid):
    return json.loads((JOBS / jid / "job.json").read_text())


def build_mask(img, rects, precise):
    w, h = img.size
    mask = np.zeros((h, w), np.uint8)
    gray = np.array(img.convert("L"))
    for x, y, rw, rh in rects:
        x0, y0 = max(int(x * w), 0), max(int(y * h), 0)
        x1, y1 = min(int((x + rw) * w), w), min(int((y + rh) * h), h)
        box = np.full((y1 - y0, x1 - x0), 255, np.uint8)
        if precise:
            crop = gray[y0:y1, x0:x1]
            k = 31 if min(crop.shape) > 40 else 5
            diff = cv2.absdiff(crop, cv2.medianBlur(crop, k))
            _, t = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            t = cv2.dilate(t, np.ones((5, 5), np.uint8), iterations=2)
            if 0.003 < t.mean() / 255 < 0.6:
                box = t
        mask[y0:y1, x0:x1] = box
    return cv2.dilate(mask, np.ones((5, 5), np.uint8))


def remove_watermark(img, rects, precise=True, engine="lama"):
    from iopaint.schema import InpaintRequest
    mask = build_mask(img, rects, precise)
    if engine in ("sd", "sdxl"):
        mgr = get_inpainter(SD_NAME if engine == "sd" else SDXL_NAME)
        req = InpaintRequest(prompt="clean plain product photo, no text, no watermark",
                             negative_prompt="text, watermark, logo, letters, signature",
                             sd_steps=30)
    else:
        mgr = get_inpainter("lama")
        req = InpaintRequest()
    res = mgr(np.array(img), mask, req)
    return Image.fromarray(cv2.cvtColor(res.astype(np.uint8), cv2.COLOR_BGR2RGB))


def hq_cutout(img, key):
    """قناع النموذج -> trimap -> closed-form matting -> إزالة هالة الخلفية"""
    from rembg import remove
    try:
        from pymatting import estimate_alpha_cf, estimate_foreground_ml
        orig = img.size
        mask = hr_mask(img) if key == "hr" else remove(img, session=get_session(key), only_mask=True)
        sc = min(1.0, HQ_SIDE / max(orig))
        sz = (int(orig[0] * sc), int(orig[1] * sc))
        im_s, m_s = img.resize(sz, Image.LANCZOS), mask.resize(sz, Image.LANCZOS)
        arr = np.asarray(im_s, np.float64) / 255.0
        m = (np.asarray(m_s) > 127).astype(np.uint8)
        k = max(5, int(max(sz) * 0.005)) | 1
        ker = np.ones((k, k), np.uint8)
        trimap = np.full(m.shape, 0.5)
        trimap[cv2.erode(m, ker) == 1] = 1.0
        trimap[cv2.dilate(m, ker) == 0] = 0.0
        alpha = np.clip(estimate_alpha_cf(arr, trimap), 0, 1)
        fg = estimate_foreground_ml(arr, alpha)
        rgba = np.dstack([fg, alpha])
        out = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA")
        if sc < 1.0:
            out = out.resize(orig, Image.LANCZOS)
        return out
    except Exception as e:
        print("hq failed, fallback:", e, flush=True)
        return simple_cutout(img, key)


def simple_cutout(img, key):
    from rembg import remove
    if key == "hr":
        c = img.convert("RGBA")
        c.putalpha(hr_mask(img))
        return c
    return remove(img, session=get_session(key))


def studio(cut, size, shadow):
    a = cut.getchannel("A")
    bbox = a.point(lambda v: 255 if v > 8 else 0).getbbox()
    if bbox:
        cut = cut.crop(bbox)
    scale = size * 0.84 / max(cut.size)
    cut = cut.resize((max(1, int(cut.width * scale)), max(1, int(cut.height * scale))), Image.LANCZOS)
    px, py = (size - cut.width) // 2, (size - cut.height) // 2
    canvas = Image.new("RGB", (size, size), "white")
    if shadow > 0:
        sh = Image.new("L", (size, size), 0)
        sh.paste(cut.getchannel("A"), (px, py + int(size * 0.015)))
        sh = sh.filter(ImageFilter.GaussianBlur(size * 0.02))
        sh = sh.point(lambda v: int(v * 0.5 * shadow / 100))
        canvas.paste(Image.new("RGB", (size, size), (0, 0, 0)), mask=sh)
    canvas.paste(cut, (px, py), cut)
    return canvas


def photoroom(src, dst_dir, opts):
    """PhotoRoom Image Editing API: إزالة خلفية + استديو + ظل + إزالة علامة نصية"""
    import io
    import requests
    key = os.getenv("PHOTOROOM_API_KEY", "")
    if not key:
        raise RuntimeError("PHOTOROOM_API_KEY غير مضبوط")
    d = {"removeBackground": "true" if opts.get("bg") else "false"}
    if opts.get("wm"):
        d["textRemoval.mode"] = opts.get("textmode", "ai.artificial")
    is_studio = bool(opts.get("bg") and opts.get("studio"))
    if is_studio:
        size = int(opts.get("size", 1200))
        d.update({"background.color": "FFFFFF", "outputSize": f"{size}x{size}", "padding": "0.08"})
        if int(opts.get("shadow", 40)) > 0:
            d["shadow.mode"] = "ai.soft"
    r = None
    for attempt in range(3):
        with open(src, "rb") as f:
            r = requests.post("https://image-api.photoroom.com/v2/edit", headers={"x-api-key": key},
                              files={"imageFile": f}, data=d, timeout=180)
        if r.status_code == 200:
            break
        if r.status_code not in (429, 500, 502, 503, 504):
            break
        time.sleep(3 * (attempt + 1))
    if r.status_code != 200:
        raise RuntimeError(f"PhotoRoom {r.status_code}: {r.text[:150]}")
    im = Image.open(io.BytesIO(r.content))
    stem = Path(src).stem
    if is_studio:
        name = stem + ".jpg"
        im.convert("RGB").save(dst_dir / name, quality=95, subsampling=0)
    else:
        name = stem + ".png"
        im.save(dst_dir / name)
    return name


def process(src, dst_dir, opts):
    if opts.get("engine") == "api":
        return photoroom(src, dst_dir, opts)
    from rembg import remove
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    if opts.get("wm") and opts.get("rects"):
        img = remove_watermark(img, opts["rects"], opts.get("precise", True), opts.get("wmengine", "lama"))
    stem = Path(src).stem
    if opts.get("bg"):
        key = opts.get("model", "birefnet")
        cut = hq_cutout(img, key) if opts.get("hq", True) else simple_cutout(img, key)
        if opts.get("studio"):
            out = studio(cut, int(opts.get("size", 1200)), int(opts.get("shadow", 40)))
            name = stem + ".jpg"
            out.save(dst_dir / name, quality=95, subsampling=0)
            return name
        name = stem + ".png"
        cut.save(dst_dir / name)
        return name
    name = stem + ".png"
    img.save(dst_dir / name)
    return name


def worker():
    while True:
        jid = Q.get()
        try:
            job = load(jid)
            job["status"] = "running"
            save(job)
            out_dir = JOBS / jid / "out"
            out_dir.mkdir(exist_ok=True)
            for f in job["files"]:
                if f["status"] == "done":
                    continue
                try:
                    f["out"] = process(JOBS / jid / "in" / f["in"], out_dir, job["opts"])
                    f["status"] = "done"
                except Exception as e:
                    f["status"] = "error"
                    f["error"] = str(e)[:200]
                job["done"] = sum(1 for x in job["files"] if x["status"] in ("done", "error"))
                save(job)
            job["status"] = "finished"
            save(job)
        except Exception as e:
            print("job failed", jid, e, flush=True)


@app.on_event("startup")
def startup():
    threading.Thread(target=worker, daemon=True).start()
    for d in sorted(JOBS.iterdir()):
        try:
            j = json.loads((d / "job.json").read_text())
            if j["status"] in ("queued", "running"):
                Q.put(j["id"])
        except Exception:
            pass


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@app.post("/api/jobs")
def create(opts: str = Form(...)):
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    (JOBS / jid / "in").mkdir(parents=True)
    job = {"id": jid, "created": time.time(), "status": "uploading", "total": 0,
           "done": 0, "opts": json.loads(opts), "files": []}
    save(job)
    return {"id": jid}


@app.post("/api/jobs/{jid}/files")
async def add_file(jid: str, file: UploadFile = File(...)):
    jid = Path(jid).name
    job = load(jid)
    i = len(job["files"])
    orig = Path(file.filename or f"img{i}").name
    stored = f"{i:04d}_{Path(orig).stem}{Path(orig).suffix.lower()}"
    with open(JOBS / jid / "in" / stored, "wb") as fh:
        shutil.copyfileobj(file.file, fh)
    job["files"].append({"name": orig, "in": stored, "out": None, "status": "queued"})
    job["total"] = len(job["files"])
    save(job)
    return {"n": job["total"]}


@app.post("/api/jobs/{jid}/start")
def start(jid: str):
    jid = Path(jid).name
    job = load(jid)
    job["status"] = "queued"
    save(job)
    Q.put(jid)
    return {"ok": True}


@app.get("/api/jobs")
def list_jobs():
    out = []
    for d in sorted(JOBS.iterdir(), reverse=True)[:20]:
        try:
            out.append(json.loads((d / "job.json").read_text()))
        except Exception:
            pass
    return out


@app.get("/api/jobs/{jid}/file/{name}")
def get_file(jid: str, name: str):
    p = JOBS / Path(jid).name / "out" / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/api/jobs/{jid}/zip")
def get_zip(jid: str):
    d = JOBS / Path(jid).name
    z = d / "result.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
        for p in sorted((d / "out").glob("*")):
            zf.write(p, p.name)
    return FileResponse(z, filename=f"{jid}.zip")


@app.delete("/api/jobs/{jid}")
def delete(jid: str):
    shutil.rmtree(JOBS / Path(jid).name, ignore_errors=True)
    return {"ok": True}
