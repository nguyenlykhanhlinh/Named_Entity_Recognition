from flask import Flask, render_template, url_for, request


#NLP Spacy
import spacy
from spacy import displacy

nlp=spacy.load("vi_core_news_lg")
from flaskext.markdown import Markdown

#Init App
app = Flask(__name__)
Markdown(app)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract', methods=['POST','GET'])
def extract():
    entities = []
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        doc = nlp(rawtext)
        for entity in doc.ents:
            entities.append({"text": entity.text, "label": entity.label_})
        return render_template('results.html', rawtext=rawtext, entities=entities)
        


if __name__ == '__main__':
    app.run(debug=True)