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
MODELS = {"birefnet": "birefnet-general", "bria": "bria-rmbg"}

security = HTTPBasic(auto_error=False)


def auth(c: HTTPBasicCredentials = Depends(security)):
    if not PASSWORD or not c or not secrets.compare_digest(c.password, PASSWORD):
        raise HTTPException(401, "Unauthorized", headers={"WWW-Authenticate": "Basic"})


app = FastAPI(dependencies=[Depends(auth)])
Q = queue.Queue()
LOCK = threading.Lock()
_lama = None
_sessions = {}


def get_lama():
    global _lama
    if _lama is None:
        from iopaint.model_manager import ModelManager
        _lama = ModelManager(name="lama", device="cpu")
    return _lama


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


def remove_watermark(img, rects):
    from iopaint.schema import InpaintRequest
    w, h = img.size
    mask = np.zeros((h, w), np.uint8)
    for x, y, rw, rh in rects:
        x0, y0 = int(x * w), int(y * h)
        x1, y1 = int((x + rw) * w), int((y + rh) * h)
        mask[max(y0, 0):y1, max(x0, 0):x1] = 255
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    res = get_lama()(np.array(img), mask, InpaintRequest())
    return Image.fromarray(cv2.cvtColor(res.astype(np.uint8), cv2.COLOR_BGR2RGB))


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


def process(src, dst_dir, opts):
    from rembg import remove
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    if opts.get("wm") and opts.get("rects"):
        img = remove_watermark(img, opts["rects"])
    stem = Path(src).stem
    if opts.get("bg"):
        cut = remove(img, session=get_session(opts.get("model", "birefnet")), post_process_mask=True)
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
