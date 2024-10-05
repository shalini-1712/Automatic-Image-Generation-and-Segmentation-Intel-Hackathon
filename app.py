from flask import Flask, render_template, request
from flask_wtf import CSRFProtect
from inference import generate_image, predict_image, segement_image

app = Flask(__name__, static_folder='generated_images')
app.secret_key = b'_53oi3uriq9pifpff;apl'
csrf = CSRFProtect(app)


@app.route('/')
def open_main():
    return render_template('index.html')

@app.route('/generation', methods=['POST', 'GET'])
def generation():
    if request.method == 'POST':
        dataroot = request.form['model']
        try:
            num = int(request.form['num_to_gen'])
            ls = generate_image(dataroot=dataroot, num=num)
            return render_template('generation.html', images=ls)
        except:
            img_data = request.files['file']
            img_data.save('generated_images/mask.png')
            img = predict_image(dataroot=dataroot, img_path='generated_images/mask.png')
            return render_template('generation.html', images=img)
            
    return render_template('generation.html')

@app.route('/segmentation', methods=['POST', 'GET'])
def segmentation():
    if request.method == 'POST':
        dataroot = request.form['model']
        img_data = request.files['file']
        img_data.save('generated_images/seg_input.png')
        img = segement_image(dataroot=dataroot, img_path='generated_images/seg_input.png')
        return render_template('segmentation.html', images=img)
    return render_template('segmentation.html')

if __name__ == '__main__':
    app.run(debug=True)