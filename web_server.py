from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path
import os


app = Flask(__name__, template_folder=".")

app.config['UPLOAD_FOLDER'] = 'upload/'


@app.route("/")
def idnex():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file_obj = request.files.get('file')
    if not file_obj:
        return jsonify(code=-1, msg='no file'), 400

    file_name = secure_filename(file_obj.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    file_obj.save(file_path)
    ret = verify_image_from_tf(file_path)
    if not ret:
        return jsonify(code=-2, msg='verify_image failed'), 400

    return jsonify(code=0, ret=ret)


def verify_image_from_tf(file_path):
    from tf_client import TFClient
    file_path = Path.cwd().joinpath(file_path)
    # return TFClient.verify_image(str(file_path))
    try:
        return TFClient.verify_image(str(file_path))
    except Exception as e:
        print("TFClient error: %s" % e)

    return None

# app.run(debug=True, host="0.0.0.0")
