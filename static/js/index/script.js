let mediaRecorder;
let audioChunks = [];
const micBtn = document.getElementById('mic-btn');

async function startRecording(){
try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = (event) => {
    audioChunks.push(event.data);
    };

    mediaRecorder.start();
    console.log("Recording started...");
} catch (err) {
    alert("Mic Error: " + err.name);
    console.error("Mic access denied or error:", err);
}
}

async function stopRecording(){
if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    console.log("Recording stopped.");

    mediaRecorder.onstop = async () => {
    const path = window.location.pathname;
    const parts = path.split("/").filter(part => part.length > 0);
    const [sex, word] = parts;
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const formData = new FormData();
    formData.append('recording', audioBlob, 'recording.wav');
    formData.append("sex", sex)
    formData.append("word", word)

    try {
        const response = await fetch('/post_audio', {
        method: 'POST',
        body: formData
        });
        if (response.ok){
        const data = await response.json();
        console.log(data);
        document.getElementById('user_pronounciation').innerText = data.output;
        } else {
        document.getElementById('user_pronounciation').innerText = "server error :(";
        }
        console.log("Sent to server successfully");
    } catch (err) {
        console.error("Error sending to server:", err);
    }
    };
}
}

micBtn.addEventListener('mousedown', startRecording);
micBtn.addEventListener('touchstart', startRecording);

micBtn.addEventListener('mouseup', stopRecording);
micBtn.addEventListener('touchend', stopRecording);


function playAudioUK() {
document.getElementById("audioUK").play();
}
function playAudioUS() {
document.getElementById("audioUS").play();
}
