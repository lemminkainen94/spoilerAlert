let req = new XMLHttpRequest();
req.open('POST', 'http://localhost:9696/spoilerAlert');
req.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');    req.onload = function() {
    let response = req.responseText;
    document.documentElement.innerHTML = response;
    console.log(response.result);
}
req.send(document.documentElement.innerHTML.toString());
