let constraintObj = { 
    audio: false, 
    video: true 
}; 


if (navigator.mediaDevices === undefined) {
    navigator.mediaDevices = {};
    navigator.mediaDevices.getUserMedia = function(constraintObj) {
        let getUserMedia = navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
        if (!getUserMedia) {
            return Promise.reject(new Error('getUserMedia is not implemented in this browser'));
        }
        return new Promise(function(resolve, reject) {
            getUserMedia.call(navigator, constraintObj, resolve, reject);
        });
    }
}else{
    navigator.mediaDevices.enumerateDevices()
    .then(devices => {
        devices.forEach(device=>{
            console.log(device.kind.toUpperCase(), device.label);

        })
    })
    .catch(err=>{
        console.log(err.name, err.message);
    })
}

navigator.mediaDevices.getUserMedia(constraintObj)
.then(function(mediaStreamObj) {

    let video = document.querySelector('video');
    if ("srcObject" in video) {
        video.srcObject = mediaStreamObj;
    } else {

        video.src = window.URL.createObjectURL(mediaStreamObj);
    }
    
    video.onloadedmetadata = function(ev) {

        video.play();
    };

    let start = document.getElementById("startbtn");
    let stop = document.getElementById("stopbtn");
    let vidSave = document.getElementById("vid");
    vidSave.style.display = "none"
    let save = document.getElementById("saveVid")
    let mediaRecorder = new MediaRecorder(mediaStreamObj);
    let chunks = [];
    
    start.addEventListener('click', (ev) =>{
        mediaRecorder.start();
        var y = document.getElementById("vid1");
        var x = document.getElementById("vid");
        if (y.style.display === "none") {
            y.style.display = "block";
            x.style.display = "block";
        }
        else {
            x.style.display = "block";
            y.style.display = "block";
        }
        console.log(mediaRecorder.state);
    })
    stop.addEventListener('click', (ev) =>{
        mediaRecorder.stop();
        var x = document.getElementById("vid1");
        var y = document.getElementById("vid");
        if (x.style.display === "none") {
            x.style.display = "block";
            y.style.display = "none";
        }
        else {
            x.style.display = "none";
            y.style.display = "block";
        }
        console.log(mediaRecorder.state);
    });
    mediaRecorder.ondataavailable = function(ev) {
        chunks.push(ev.data);
    }
    mediaRecorder.onstop = (ev) =>{
        let blob = new Blob(chunks, { 'type' : 'video/mp4;' });
        chunks = [];
        let videoURL = window.URL.createObjectURL(blob);
        vidSave.src = videoURL;
        let formData = new FormData();
        formData.append('video', blob, "video.mp4")

        save.addEventListener('click', (ev) =>{
            // var file = new File([blob], "video.mp4",{type: "video/mp4"});
            // getSignedRequest(file);

            $.ajax ({
                url: "/upload",
                type: "POST",
                data: formData,
                cache: false,
                processData: false,
                contentType: false
            }).done(function(response){
                console.log(response);
            })
        })
    }})


.catch(function(err) { 
    console.log(err.name, err.message); 
});
