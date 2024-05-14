let selectedImages = [];
let selectedExcel = [];
let currentProject;

document.addEventListener('DOMContentLoaded', function () { 
    currentProject = "";
    loadProject(); 
    loadImageList(currentProject);
})

function loadProject(){
    fetch('http://localhost:8000/get_projects')
    .then(response => response.json())
    .then(data => {
        const projectMenu = document.getElementById('projectMenu');
        const projectDropdownTitle = document.getElementById('ProjectDropdown');

        while (projectMenu.firstChild) {
            projectMenu.removeChild(projectMenu.firstChild);
        }

        projectDropdownTitle.textContent = data.length ? 'No project selected' : 'Project: ';

        data.forEach(project => {
            const projectElement = document.createElement('a');
            projectElement.href = '#';
            projectElement.classList.add('dropdown-item');
            projectElement.textContent = project;

            projectElement.addEventListener('click', function() {
                projectDropdownTitle.textContent = `Project: ${project}`;
                currentProject = project;
                fetch('http://localhost:8000/save_project', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ projectName: project }),
                })
                .then(response => response.json())
                .then(data => {
                    console.log('Success:', data);
                })
                .catch((error) => {
                    console.error('Error:', error);
                });
                loadImageList(project);
            });

            projectMenu.appendChild(projectElement);
        });
    })
    .catch(error => console.error('Error:', error));
}

function loadImageList(projectName) {
    const gallery = document.querySelector('.image-gallery');
    const projectInfo = document.getElementById('projectInfo');

    if (currentProject === "") {
        gallery.innerHTML = '<div class="alert alert-warning" role="alert">No Project selected</div>';
        projectInfo.innerHTML = '';
    } else {
        fetch(`http://localhost:8000/images_list/${projectName}`)
        .then(response => response.json())
        .then(data => {
            gallery.innerHTML = '';
            projectInfo.innerHTML = '';

            data.images.sort((a, b) => a.localeCompare(b));

            const table = document.createElement('table');
            table.classList.add('table');

            const thead = document.createElement('thead');
            const trHead = document.createElement('tr');
            ['Name', 'Segmentations', 'Actions'].forEach(text => {
                const th = document.createElement('th');
                th.textContent = text;
                trHead.appendChild(th);
            });
            thead.appendChild(trHead);
            table.appendChild(thead);

            const tbody = document.createElement('tbody');

            data.images.forEach(image => {
                const tr = document.createElement('tr');
                tr.classList.add('image-row');

                const tdName = document.createElement('td');
                const imageName = String(image).split('/').pop();
                tdName.textContent = imageName;
                tr.appendChild(tdName);

                const tdSegmentation = document.createElement('td');
                const segmentationDropdownDiv = document.createElement('div');
                segmentationDropdownDiv.classList.add('dropdown');

                const dropdownToggle = document.createElement('a');
                dropdownToggle.classList.add('nav-link', 'dropdown-toggle');
                dropdownToggle.href = '#';
                dropdownToggle.role = 'button';
                dropdownToggle.id = `segmentationDropdown-${imageName}`;
                dropdownToggle.dataset.bsToggle = 'dropdown';
                dropdownToggle.setAttribute('aria-haspopup', 'true');
                dropdownToggle.setAttribute('aria-expanded', 'false');

                const dropdownMenuDiv = document.createElement('div');
                dropdownMenuDiv.classList.add('dropdown-menu');
                dropdownMenuDiv.setAttribute('aria-labelledby', `segmentationDropdown-${imageName}`);

                segmentationDropdownDiv.appendChild(dropdownToggle);
                segmentationDropdownDiv.appendChild(dropdownMenuDiv);

                getSegmentationMask(imageName, projectName).then(segmentation_masks => {
                    segmentation_masks.forEach(mask => {
                        const dropdownItem = document.createElement('a');
                        dropdownItem.classList.add('dropdown-item');
                        dropdownItem.href = '#';
                        dropdownItem.textContent = mask;
                        dropdownItem.setAttribute('data-fullname', `${mask}_${imageName}_seg`); 
                        dropdownItem.addEventListener('click', function(e) {
                            e.stopPropagation(); // Empêcher l'événement de remonter au tr
                            const fullMaskName = this.getAttribute('data-fullname'); 
                            const index = selectedImages.indexOf(fullMaskName);

                            if (index === -1) {
                                selectedImages.push(fullMaskName);
                                this.classList.add('active');
                            } else {
                                selectedImages.splice(index, 1);
                                this.classList.remove('active');
                            }
                        });
                        dropdownMenuDiv.appendChild(dropdownItem);
                    });
                }).catch(error => {
                    console.error('Error fetching segmentation masks:', error);
                });

                tdSegmentation.appendChild(segmentationDropdownDiv);
                tr.appendChild(tdSegmentation);

                const tdActions = document.createElement('td');
                const deleteButton = document.createElement('button');
                deleteButton.classList.add('btn', 'btn-danger', 'btn-sm');
                deleteButton.textContent = 'Delete';
                deleteButton.onclick = function() {
                    deleteImage(imageName, projectName);
                };
                tdActions.appendChild(deleteButton);
                tr.appendChild(tdActions);

                tr.addEventListener('click', function() {
                    this.classList.toggle('selected');
                    const fileName = imageName;  
                    const fileExtension = fileName.slice(-5).toLowerCase();
                
                    if (fileExtension === '.xlsx') {
                        const excelIndex = selectedExcel.indexOf(fileName);
                        if (this.classList.contains('selected')) {
                            if (excelIndex === -1) {
                                selectedExcel.push(fileName);
                            }
                        } else {
                            if (excelIndex !== -1) {
                                selectedExcel.splice(excelIndex, 1);
                            }
                        }
                    } else {
                        const imageIndex = selectedImages.indexOf(fileName);
                        if (this.classList.contains('selected')) {
                            if (imageIndex === -1) {
                                selectedImages.push(fileName);
                            }
                        } else {
                            if (imageIndex !== -1) {
                                selectedImages.splice(imageIndex, 1);
                            }
                        }
                    }
                });

                tbody.appendChild(tr);
            });

            table.appendChild(tbody);
            gallery.appendChild(table);

            if (data.project_info) {
                const sortedKeys = Object.keys(data.project_info).sort((a, b) => a.localeCompare(b));
            
                sortedKeys.forEach(key => {
                    const value = data.project_info[key];
                    const infoPara = document.createElement('div');
                    infoPara.className = 'info-para';
            
                    const label = document.createElement('label');
                    label.textContent = `${key.charAt(0).toUpperCase() + key.slice(1).replace('_', ' ')}: `;
                    label.htmlFor = `input-${key}`;
            
                    const input = document.createElement('input');
                    input.id = `input-${key}`;
                    input.type = 'text';
                    input.value = Array.isArray(value) ? value.join(', ') : value;
                    input.disabled = true;
            
                    infoPara.appendChild(label);
                    infoPara.appendChild(input);
                    projectInfo.appendChild(infoPara);
                });
            
                //toggleEdit(false);
                const editButton = document.createElement('button');
                editButton.textContent = 'Modify';
                editButton.className = 'btn btn-primary';
                editButton.onclick = () => toggleEdit(true);
            
                const saveButton = document.createElement('button');
                saveButton.textContent = 'Save';
                saveButton.className = 'btn btn-success';
                saveButton.style.display = 'none';  // Initially hidden
                saveButton.onclick = () => saveProjectInfo(projectName);
            
                projectInfo.appendChild(editButton);
                projectInfo.appendChild(saveButton);
            }
        })
        .catch(error => console.error('Error:', error));
    }
}

function toggleCardVisibility(elementId) {
    const element = document.getElementById(elementId);
    if (element.style.display === 'none') {
        element.style.display = 'block'; 
    } else {
        element.style.display = 'none';   
    }
}

function toggleEdit(enable) {
    const inputs = document.querySelectorAll('#projectInfo input');
    inputs.forEach(input => {
        input.disabled = !enable;  
    });
    document.querySelector('#projectInfo .btn-primary').style.display = enable ? 'none' : 'block';
    document.querySelector('#projectInfo .btn-success').style.display = enable ? 'block' : 'none';
}

function saveProjectInfo(projectName) {
    const inputs = document.querySelectorAll('#projectInfo input');
    const updatedInfo = {};
    inputs.forEach(input => {
        const key = input.id.replace('input-', '');
        updatedInfo[key] = input.value.split(',').map(item => item.trim());  
    });

    fetch(`http://localhost:8000/update_project_info/${projectName}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(updatedInfo)
    })
    .then(response => response.json())
    .then(data => {
        console.log('Success:', data);
        toggleEdit(false);
    })
    .catch(error => console.error('Error updating project info:', error));
}





function getSegmentationMask(imageName) {
    return fetch(`http://localhost:8000/get_segmentation/${currentProject}?image_name=${encodeURIComponent(imageName)}`)
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('Get segmentation mask succeeded!');
            return data.segmentation_masks; 
        } else {
            console.error('Error:', data.message);
            throw new Error(data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        throw error;
    });
}


// Fonction pour gérer la suppression de l'image
function deleteImage(imageName, projectName) {
    const confirmation = confirm(`Are you sure you want to delete this image: ${imageName}?`);
    if (confirmation) {
        const url = `http://localhost:8000/delete_image/${projectName}?image_name=${encodeURIComponent(imageName)}`;

        fetch(url, {
            method: 'DELETE'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log('Image deleted successfully');
                loadImageList(projectName); 
            } else {
                console.error('Error:', data.message);
            }
        })
        .catch(error => console.error('Error:', error));
    }
}


// Fonction pour afficher un message temporaire
function showPopup(message, timeout = 3000) {
    // Trouver ou créer le conteneur du message
    let popupContainer = document.getElementById('popup-container');
    if (!popupContainer) {
        popupContainer = document.createElement('div');
        popupContainer.id = 'popup-container';
        popupContainer.style.position = 'fixed';
        popupContainer.style.bottom = '20px';
        popupContainer.style.right = '20px';
        popupContainer.style.padding = '10px';
        popupContainer.style.backgroundColor = 'rgba(0, 128, 0, 0.8)'; // Couleur de fond avec transparence
        popupContainer.style.color = 'white';
        popupContainer.style.borderRadius = '5px';
        popupContainer.style.zIndex = 10000;
        popupContainer.style.boxShadow = '0px 0px 10px rgba(0, 0, 0, 0.5)';
        document.body.appendChild(popupContainer);
    }

    // Mettre à jour le texte du message
    popupContainer.textContent = message;

    // Rendre le message visible
    popupContainer.style.display = 'block';

    // Cacher le message après le délai défini
    setTimeout(() => {
        popupContainer.style.display = 'none';
    }, timeout);
}



function uploadImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    fetch(`http://localhost:8000/upload_image/${currentProject}`, {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            loadImageList(currentProject);
        } else {
            console.error('Error:', data.message);
        }
    })
    .catch(error => console.error('Error:', error));
}


function uploadFunction(){
    if (currentProject == "") {
        return;
    }
    document.getElementById('file-input').click();
    document.getElementById('file-input').addEventListener('change', function(event) {
        const files = event.target.files;
        uploadImage(files[0]);
    });
}

function deselectAllImages() {
    document.querySelectorAll('.image-row.selected').forEach(row => {
        row.classList.remove('selected');
    });
    selectedImages = []; 
}

function deselectAllExcelFiles() {
    document.querySelectorAll('.image-row.selected').forEach(row => {
        row.classList.remove('selected');
    });
    selectedExcel = []; 
}

function visualizeFunction() {
    if (selectedImages.length === 0 && selectedExcel.length === 0) {
        showPopup("Nothing is selected to visualize");
        return;
    } else {
        if (selectedImages.length > 0) {
            const imageNamesString = selectedImages.join(',');

            // Utilisez URLSearchParams pour un encodage correct
            const params = new URLSearchParams({
                image_names: imageNamesString, // Pas besoin d'encodeURIComponent ici
                control_points: false
            });

            fetch(`http://localhost:8000/launch_napari/${currentProject}?${params.toString()}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        console.log('Napari launched for images:', imageNamesString);
                    } else {
                        console.error('Error:', data.message);
                    }
                })
                .catch(error => console.error('Error:', error));

            deselectAllImages();
        }

        if (selectedExcel.length > 0) {
            const excelFilesString = selectedExcel.join(',');

            fetch(`http://localhost:8000/open_excel_files/${currentProject}?excel_files=${encodeURIComponent(excelFilesString)}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        console.log('Excel files opened:', excelFilesString);
                    } else {
                        console.error('Error:', data.message);
                    }
                })
                .catch(error => console.error('Error:', error));

            deselectAllExcelFiles();
        }
    }
}



function SetControlPoints() {
    if (selectedImages.length === 0 && selectedExcel.length === 0) {
        showPopup("Nothing is selected to set the control points");
        return;
    } else {
        if (selectedImages.length > 0) {
            const imageNamesString = selectedImages.join(',');

            // Utilisez URLSearchParams pour un encodage correct
            const params = new URLSearchParams({
                image_names: imageNamesString, // Pas besoin d'encodeURIComponent ici
                control_points: true
            });

            fetch(`http://localhost:8000/launch_napari/${currentProject}?${params.toString()}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        console.log('Napari launched for images:', imageNamesString);
                    } else {
                        console.error('Error:', data.message);
                    }
                })
                .catch(error => console.error('Error:', error));

            deselectAllImages();
        }

        if (selectedExcel.length > 0) {
            showPopup("Don't select an Excel File to set the control points");
            deselectAllExcelFiles();
        }
    }
}




function deleteFunction() {
    const confirmation = confirm("Are you sure you want to delete this project?");
        if (confirmation) {
            fetch(`http://localhost:8000/delete_project/${currentProject}`, {
                method: 'DELETE'
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    console.log('Project deleted successfully');
                    currentProject = "";
                    loadProject(); 
                    loadImageList(currentProject);
                } else {
                    console.error('Error:', data.message);
                }
            })
            .catch(error => console.error('Error:', error));
        }
}

document.getElementById('newProject').addEventListener('click', function() {
    var myModal = new bootstrap.Modal(document.getElementById('newProjectModal'), {
        keyboard: false
    });
    myModal.show();
});


document.getElementById('newProjectForm').addEventListener('submit', function(event) {
    event.preventDefault();
    var projectName = document.getElementById('projectNameInput').value;
    if (projectName) {
        var data = { projectName: projectName };

        fetch('http://localhost:8000/create_project', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        })
        .then(response => response.json())
        .then(data => {
            console.log('Success:', data);
            loadProject();
            document.getElementById('projectNameInput').value = "";
        })
        .catch((error) => {
            console.error('Error:', error);
        });

        var myModalEl = document.getElementById('newProjectModal');
        var modal = bootstrap.Modal.getInstance(myModalEl);
        modal.hide();
        loadProject();

    } else {
        alert("Please enter a project name.");
    }
});



