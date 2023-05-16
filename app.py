from flask import Flask, request, render_template, jsonify
from model import SentimentRecommenderModel

# Create flask app
app = Flask(__name__)

sentiment_model = SentimentRecommenderModel()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get user from the html form
    user = request.form['userName'].lower()
    top_5_recommendations = sentiment_model.get_top5_recommendations(user)

    if(not(top_5_recommendations is None)):
        # print(f"retrieving items....{len(top_5_recommendations)}")
        print(top_5_recommendations)
        # data=[items.to_html(classes="table-striped table-hover", header="true",index=False)
        return render_template("index.html", column_names=top_5_recommendations.columns.values, row_data=list(top_5_recommendations.values.tolist()), zip=zip)
    else:
        return render_template("index.html", message="Username does not exist. Please try again.")

if __name__ == '__main__':
    app.run()