from flask import Flask , render_template, request, send_from_directory
from tensorflow.keras.preprocessing.image import load_img
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './images/'

classification_label = [line.rstrip() for line in tf.io.gfile.GFile('logs/train.txt')]
with tf.io.gfile.GFile('logs/train.pb', 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
    graph = tf.compat.v1.get_default_graph()
    sess = tf.compat.v1.Session()
    softmax_tensor = graph.get_tensor_by_name('final_result:0')

@app.route('/', methods = ['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST']) 
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    frame = load_img(image_path)
    image_data = tf.io.gfile.GFile(image_path, 'rb').read()
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    n_id = top_k[0]
    labell = classification_label[n_id]
    for node_id in top_k:
        label = classification_label[node_id]
        score = predictions[0][node_id]
        if score*100 >= 50:
            classification = '%s (score = %.5f)' % (label, score*100)
    return render_template('index.html', uploaded_image= imagefile.filename, prediction = classification)

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
