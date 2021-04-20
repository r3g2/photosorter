from flask import Flask,flash,request,redirect,url_for,render_template,send_from_directory
import os
import shutil
from datetime import datetime
from werkzeug.utils import secure_filename
from PhotoSorter import Photo, PhotoSorter
from utils import *
import logging

logging.basicConfig(level=logging.DEBUG, format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)),'uploads')


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['TEMP_FOLDER'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),'temp')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['COLORED_DIRECTORY'] = os.path.join(UPLOAD_FOLDER,"colored_images")
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

photosorter = PhotoSorter(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=["GET","POST"])
def index():

    files = os.listdir(app.config['UPLOAD_FOLDER'])
    if files and len(photosorter.photos) == 0:
        photosorter.load_images()
    return render_template('loaded_options.html',files=files)

@app.route("/upload", methods=["POST"])
def upload_files():
    if request.method == 'POST':
        if 'images' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        uploaded_files = request.files.getlist("images")

        #TODO Check how to get foldername and where to save
        save_path = app.config['UPLOAD_FOLDER']
        color_save_path = os.path.join(app.config['COLORED_DIRECTORY'])
        if not os.path.exists(color_save_path):
            os.mkdir(color_save_path)
        for file in uploaded_files:
            # If user doesn't select files then it will have an empty filename
            if file != '':
                
                filename = secure_filename(file.filename)
                file.save(os.path.join(save_path,filename))
        # Load images
        photosorter.load_images()
        for p in photosorter.photos:
            if p.color:
                shutil.copy(os.path.join(p.cur_dir,p.fname),os.path.join(color_save_path,p.fname))
        # TODO decide what to do once the user uploads files
        return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/uploads/temp/<filename>')
def temp_uploaded(filename):
    return send_from_directory(app.config['TEMP_FOLDER'], filename)

@app.route('/duplicates', methods=["POST"])
def find_duplicates():
    dup_fnames,_ = photosorter.find_duplicate_images()
    return render_template('duplicates.html',groups=dup_fnames)
    
@app.route('/remove', methods=["POST"])
def remove_duplicates():
    duplicates = request.form['remove_button']
    app.logger.info(duplicates)
    return 201

@app.route('/similarimages', methods=["POST"])
def find_similar_images():
    if request.method == 'POST':
        if 'query' not in request.files:
            flash('No query image')
            return redirect(request.url)
        file = request.files['query']
        if file != '':    
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["TEMP_FOLDER"],filename)
            file.save(filepath)
        query_img = cv2.imread(filepath)
        top_sims = photosorter.query_similar(query_img)
        #fnames = [t[0].fname for t in top_sims]
        return render_template('similar.html',similarities=top_sims,query_image=filename)

@app.route('/remove_upload', methods=["POST"])
def remove_upload():
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
        try:
            if os.path.isfile(path):
                os.unlink(path)
        except Exception as e:
            print("Couldn't delete %s because %s" % (path,e))
    return redirect(url_for('index'))

@app.route('/group/<numberofgroups>')
def group_images(numberofgroups):
    pass

if __name__ == '__main__':
    app.run(debug=True)