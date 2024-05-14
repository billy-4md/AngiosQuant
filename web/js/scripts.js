function launchNapari(imageUrl) {
    fetch(`http://localhost:8000/launch_napari?image_url=${encodeURIComponent(imageUrl)}`)
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log("Napari launched with image");
        } else {
            console.error('Error launching Napari:', data.message);
        }
    })
    .catch(error => console.error('Error:', error));
}


