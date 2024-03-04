from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/api/background-subtraction', methods=['POST'])
def background_subtraction():
    # Extract the frame from the request
    data = request.json
    encoded_frame = data['data']
    # Here, you would add your background subtraction logic
    # For simplicity, we'll just return a success message
    return jsonify({"status": "success", "task": "background-subtraction", "message": "Processed"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)