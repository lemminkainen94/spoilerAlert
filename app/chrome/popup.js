chrome.tabs.executeScript({
    code: "window.getSelection().toString();"
}, function(selection) {
    let req = new XMLHttpRequest();
    req.open('POST', 'https://YOUR-API-ADDRESS’);
    req.setRequestHeader('Content-Type', 'application/json');    req.onload = function() {
        let response = JSON.parse(req.responseText);
        document.getElementById('output').innerHTML = response;
    }    let sendData = JSON.stringify({
        "review": selection[0]
    });    req.send(sendData);
});