

import requests

app = 'http://localhost:8000/predict/?ticker=SBIN'

req = requests.get(app)

req = req.json()

print(req)