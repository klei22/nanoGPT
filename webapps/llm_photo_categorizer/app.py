import os
import json
from flask import Flask, render_template, request, send_file, url_for, abort

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "heic"}
CATEGORIES = [
    "Model Architecture",
    "Training Techniques",
    "Datasets",
    "Evaluation",
    "Applications",
]

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
CATEGORIZED_DIR = os.path.join(STATIC_DIR, "categorized")
os.makedirs(CATEGORIZED_DIR, exist_ok=True)

TAGS_FILE = os.path.join(CATEGORIZED_DIR, "tags.json")
if os.path.exists(TAGS_FILE):
    with open(TAGS_FILE, "r") as f:
        TAGS = json.load(f)
else:
    TAGS = {}

app = Flask(__name__)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    folder = request.values.get("folder")
    index = int(request.values.get("index", 0))
    message = None

    if request.method == "POST" and folder:
        filename = request.form.get("filename")
        selected = request.form.getlist("category")
        new_cat_input = request.form.get("new_category", "")
        new_cats = [c.strip() for c in new_cat_input.split(",") if c.strip()]
        for cat in new_cats:
            if cat not in CATEGORIES:
                CATEGORIES.append(cat)
            selected.append(cat)
        if filename:
            src = os.path.abspath(os.path.join(folder, filename))
            if os.path.isfile(src) and allowed_file(filename):
                for cat in selected:
                    cat_dir = os.path.join(CATEGORIZED_DIR, cat)
                    os.makedirs(cat_dir, exist_ok=True)
                    link_path = os.path.join(cat_dir, filename)
                    if not os.path.exists(link_path):
                        os.symlink(src, link_path)
                TAGS[src] = list(sorted(set(selected)))
                with open(TAGS_FILE, "w") as f:
                    json.dump(TAGS, f, indent=2)
                message = f"Saved categories for {filename}"
                index += 1

    files = []
    if folder and os.path.isdir(folder):
        all_files = [f for f in os.listdir(folder) if allowed_file(f)]
        abs_paths = [os.path.abspath(os.path.join(folder, f)) for f in all_files]
        files = [f for f, p in zip(all_files, abs_paths) if p not in TAGS]
    filename = files[index] if folder and index < len(files) else None
    image_url = url_for("image", folder=folder, filename=filename) if filename else None

    return render_template(
        "index.html",
        categories=CATEGORIES,
        folder=folder,
        filename=filename,
        index=index,
        image_url=image_url,
        message=message,
    )


@app.route("/image")
def image():
    folder = request.args.get("folder")
    filename = request.args.get("filename")
    if not folder or not filename:
        abort(404)
    path = os.path.abspath(os.path.join(folder, filename))
    if not os.path.isfile(path):
        abort(404)
    return send_file(path)


@app.route("/category/<cat>")
def show_category(cat):
    cat_dir = os.path.join(CATEGORIZED_DIR, cat)
    if not os.path.isdir(cat_dir):
        abort(404)
    files = sorted(os.listdir(cat_dir))
    images = [url_for("static", filename=f"categorized/{cat}/{f}") for f in files]
    return render_template("category.html", category=cat, images=images)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
