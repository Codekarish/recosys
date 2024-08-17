from flask import Flask, render_template, request
import recommendation  # Import your recommendation module
from waitress import serve

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    query = ""
    recommendations_list = []
    if request.method == 'POST':
        query = request.form['query']
        recommendations_list = recommendation.main(query)  # Pass the query to the main function

    return render_template('index.html', query=query, recommendations=recommendations_list)

@app.route('/property/<int:property_id>')
def property_detail(property_id):
    url = "https://sapi.hauzisha.co.ke/api/properties/search"
    params = {"id": property_id}
    property_data = recommendation.fetch_data(url, params)
    if property_data.empty:
        return "Property not found", 404
    property_detail = property_data.iloc[0].to_dict()
    return render_template('property_detail.html', property=property_detail)

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)


