from flask import Flask, render_template, request
from inference import translate_sentence

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    english_text = ""
    french_text = ""

    if request.method == "POST":
        english_text = request.form.get("english_text", "")
        if english_text.strip():
            french_text = translate_sentence(english_text)

    return render_template(
        "index.html",
        english_text=english_text,
        french_text=french_text
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
