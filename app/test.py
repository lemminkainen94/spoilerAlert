import requests

html_doc = ""

with open('The Name of the Wind by Patrick Rothfuss.html') as f:
	html_doc = f.read()

url = 'http://localhost:9696/spoilerAlert'
response = requests.post(url, data=html_doc.encode('utf-8'))
with open('test2.html', 'w') as f:
	f.write(str(response.content.decode('utf-8')))
