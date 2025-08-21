from flask import Flask, request, jsonify
import os
# ... copy all same classes and logic from above ...

app = Flask(__name__)

@app.route('/process-document', methods=['POST'])
def process_document():
    # Same logic but adapted for Flask
    pass

if __name__ == '__main__':
    app.run(debug=True)
