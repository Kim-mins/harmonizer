from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from harmonizer import harmonize
import os

app = Flask(__name__)

@app.route('/upload')
def load_file():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(f.filename)
      os.system("python3 test.py --audio_dir . --save_dir . --voca False")
      harmonize(f.filename)
      return send_from_directory('./', f.filename.split(".")[0]+"_t.wav", as_attachment=True)
		
if __name__ == '__main__':
   app.run(debug = True)