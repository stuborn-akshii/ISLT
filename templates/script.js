const dropZone = document.getElementById('drop-zone');
const input = document.querySelector('input[type="file"]');
const successIcon = document.querySelector('.success-icon');
const successText = document.querySelector('.success-text');

dropZone.addEventListener('dragover', (event) => {
  event.preventDefault();
  dropZone.classList.add('dragged');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragged');
});

dropZone.addEventListener('drop', (event) => {
  event.preventDefault();
  dropZone.classList.remove('dragged');
  successIcon.classList.add('up');
  successIcon.style.display = 'block';
  successText.style.display = 'block';
  const files = event.dataTransfer.files;
  handleFiles(files);
});

input.addEventListener('change', () => {
  const files = input.files;
  handleFiles(files);
});

function handleFiles(files) {
  if (files.length > 0) {
    const fileTypes = ['video/mp4', 'video/webm', 'video/ogg'];
    const file = files[0];
    if (fileTypes.indexOf(file.type) === -1) {
      alert('Invalid file type. Please upload a video file.');
    } else {
      // Handle the video file here
      console.log(file);
    }
  }
}
