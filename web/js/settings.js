document.addEventListener("DOMContentLoaded", function() {
    fetch('http://localhost:8000/get_settings')
    .then(response => response.json())
    .then(data => {
        console.log(data)
        fillFormWithData(data);
    })
    .catch(error => console.error('Error:', error));
});

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



// document.getElementById('startSegmentation').addEventListener('click', function() {
//     save_settings(); 
    
//     fetch('http://localhost:8000/start_segmentation')
//     .then(response => response.json())
//     .then(data => console.log('Success:', data))
//     .catch(error => console.error('Error:', error));

// });

document.getElementById('saveSettings').addEventListener('click', function() {
    save_settings();    
    showPopup("Settings have been saved!")
});

function save_settings(){
    const image1Path = document.getElementById('image1PathInput').value.trim();
    const image2Path = document.getElementById('image2PathInput').value.trim();
    const saveDirectory = document.getElementById('saveDirectoryInput').value.trim(); 
    const tagsImage1 = collectTags('1');
    const tagsImage2 = collectTags('2');
    

    const upperLayersToRemove1 = parseInt(document.getElementById('upperLayersToRemove1').value.trim() || "0", 10);
    const lowerLayersToRemove1 = parseInt(document.getElementById('lowerLayersToRemove1').value.trim() || "0", 10);
    const upperLayersToRemove2 = parseInt(document.getElementById('upperLayersToRemove2').value.trim() || "0", 10); 
    const lowerLayersToRemove2 = parseInt(document.getElementById('lowerLayersToRemove2').value.trim() || "0", 10);

    const croppingvalue = parseInt(document.getElementById('croppedvalue').value.trim() || "0", 10);

    const saveAllImages = document.getElementById('saveAllImages').checked;
    const automaticMerging = document.getElementById('automaticMerging').checked;
    const useTagCenter = document.getElementById('useTagCenter').checked;

    const phalloidinTag = document.getElementById('phalloidinTagDropdown').value;

    // const populations = collectPopulations();
    // const hasDuplicateTags = populations.some(population => {
    //     const uniqueTags = new Set(population);
    //     return uniqueTags.size !== population.length;
    // });
    // if (hasDuplicateTags) {
    //     alert("Error: One of the populations has duplicate tags. Please correct it.");
    //     return;
    // }
    // const populations_dico = collectPopulationsDico();


    const settings = {
        image1: { 
            path: image1Path, 
            tags: tagsImage1,
            upperLayersToRemove: upperLayersToRemove1,
            lowerLayersToRemove: lowerLayersToRemove1 
        },
        image2: { 
            path: image2Path, 
            tags: tagsImage2,
            upperLayersToRemove: upperLayersToRemove2, 
            lowerLayersToRemove: lowerLayersToRemove2 
        },
        configuration: {
            saveAllImages: saveAllImages,
            automaticMerging: automaticMerging,
            useTagCenter: useTagCenter
        },
        //populations: populations_dico,
        croppingvalue: croppingvalue,
        saveDirectory: saveDirectory,
        phaloTag: phalloidinTag
    };

    fetch('http://localhost:8000/save_settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings),
    })
    .then(response => response.json())
    .then(data => console.log('Success:', data))
    .catch(error => console.error('Error:', error));

}

function collectTags(imageId) {
    const tagsContainer = document.getElementById(`tagsImage${imageId}`);
    const tagEntries = tagsContainer.querySelectorAll('.tag-entry');
    const tags = {};

    tagEntries.forEach(entry => {
        const tagNameInput = entry.querySelector('.labelname').textContent.slice(0, -1);
        const tagWavelengthInput = entry.querySelector('.wavelength').value;
        if (tagNameInput && tagWavelengthInput && tagWavelengthInput.trim() !== '') {
            tags[tagNameInput.trim()] = tagWavelengthInput.trim();
            // tags.push({
            //     "name": tagNameInput.trim(),
            //     "wavelength": tagWavelengthInput.trim()
            // });
        }
    });
    return tags;
}



function collectPopulations() {
    const populationsContainer = document.getElementById('populationsContainer');
    const populationEntries = populationsContainer.getElementsByClassName('population-entry');
    const populations = Array.from(populationEntries).map(population => {
        const selectElements = population.querySelectorAll('select');
        return Array.from(selectElements).map(select => select.value);
    });
    return populations;
}

function collectPopulationsDico() {
    const populationsContainer = document.getElementById('populationsContainer');
    const populationEntries = populationsContainer.getElementsByClassName('population-entry');
    const populations = {};

    Array.from(populationEntries).forEach((population, index) => {
        const selectElements = population.querySelectorAll('select');
        const tagsForPopulation = Array.from(selectElements).map(select => select.value);
        populations[`pop${index + 1}`] = tagsForPopulation;
    });

    return populations;
}


let tagCounterImage1 = 1;
let tagCounterImage2 = 0;
let tagsAddedToImage2 = false; 

function addTag(imageId) {
    // Si l'imageId est '1' et que des tags ont été ajoutés à l'image 2, bloquer l'ajout de nouveaux tags
    if (imageId === '1' && tagsAddedToImage2) {
        alert("You cannot add more TAGs in image 1 because you have some TAGs on image 2");
        return; 
    }

    const tagsContainer = document.getElementById(`tagsImage${imageId}`);
    var dropdown = document.getElementById('phalloidinTagDropdown');
    var option = document.createElement('option');
    let newTagId;

    if (imageId === '1') {
        newTagId = tagCounterImage1++;
    } else {
        tagsAddedToImage2 = true; 
        newTagId = tagCounterImage1 + tagCounterImage2; 
        tagCounterImage2++;
    }

    option.value = newTagId;
    option.textContent = "TAG " + newTagId;
    dropdown.appendChild(option);

    const tagLabel = `TAG${newTagId}`;
    const newTagDiv = document.createElement('div');
    newTagDiv.classList.add('tag-entry', 'd-flex', 'align-items-center', 'mb-2');
    newTagDiv.innerHTML = `
        <label class="labelname me-2">${tagLabel}:</label>
        <input type="text" class="form-control form-control-sm me-2 wavelength" placeholder="Wavelength" name="${tagLabel}" />
        <button onclick="removeTag(this, '${imageId}')" class="btn btn-danger btn-sm">Remove</button>
    `;
    tagsContainer.appendChild(newTagDiv);
}

function updateTagLabels(imageId) {
    const tagsContainer = document.getElementById(`tagsImage${imageId}`);
    const allTags = tagsContainer.getElementsByClassName('tag-entry');
    let counter = imageId === '1' ? 1 : tagCounterImage1; // Commencez à compter à partir de 1 pour l'image 1 ou à partir du dernier TAG de l'image 1 pour l'image 2
    Array.from(allTags).forEach(tagDiv => {
        const tagLabel = `TAG${counter}`;
        tagDiv.querySelector('label').textContent = `${tagLabel}:`;
        tagDiv.querySelector('input').setAttribute('name', tagLabel);
        counter++;
    });
    // Ajuster les compteurs en fonction de imageId
    if (imageId === '1') {
        tagCounterImage1 = counter;
    } else {
        tagCounterImage2 = counter - tagCounterImage1;
    }
}

function removeTag(button, imageId) {
    button.parentElement.remove();
    // Décrémentez le compteur en fonction de imageId pour conserver la numérotation correcte des TAGs
    if (imageId === '1') {
        tagCounterImage1--;
    } else {
        tagCounterImage2--;
        if (tagCounterImage2 == 0){
            tagsAddedToImage2 = false;
        }
    }
    // Mettez à jour tous les labels de TAG et les noms d'input pour refléter le nouvel ordre
    updateTagLabels(imageId);
}


let populationCounter = 1;

function getTotalTagsCount() {
    return tagCounterImage1 + tagCounterImage2 -1;
}

function addPopulation() {
    const populationsContainer = document.getElementById('populationsContainer');
    const newPopulationDiv = document.createElement('div');
    newPopulationDiv.classList.add('population-entry', 'mb-3'); // Ajout d'une marge en bas
    newPopulationDiv.setAttribute('id', `population${populationCounter}`);

    // Générer les sélecteurs de TAG pour cette population
    let selectorsHtml = '';
    const totalTags = getTotalTagsCount();
    for (let i = 1; i <= 3; i++) {
        selectorsHtml += `<select name="tagSelector${i}" id="tagSelector${populationCounter}-${i}" class="form-select form-select-sm mx-1">`;
        selectorsHtml += `<option value="">Select TAG</option>`;
        for (let tagNum = 1; tagNum <= totalTags; tagNum++) {
            selectorsHtml += `<option value="TAG${tagNum}">TAG${tagNum}</option>`;
        }
        selectorsHtml += `</select>`;
    }

    newPopulationDiv.innerHTML = `
        <label for="population${populationCounter}" class="form-label">Population ${populationCounter}: </label>
        <div class="d-flex align-items-center">
            ${selectorsHtml}
            <button onclick="removePopulation(${populationCounter})" class="btn btn-danger btn-sm ms-2">Remove</button>
        </div>
    `;

    populationsContainer.appendChild(newPopulationDiv);
    populationCounter++;
}

function removePopulation(populationId) {
    // Sélectionnez et supprimez le div de la population spécifique
    const populationDiv = document.getElementById(`population${populationId}`);
    if (populationDiv) {
        populationDiv.remove();
    }

    // Mettre à jour les labels et les identifiants des populations restantes pour maintenir l'ordre
    updatePopulationLabels();
}

function updatePopulationLabels() {
    // Sélectionnez tous les divs de population
    const populationsContainer = document.getElementById('populationsContainer');
    const allPopulations = populationsContainer.getElementsByClassName('population-entry');
    
    // Réinitialiser le compteur de population pour le réaligner avec le nombre actuel de populations
    populationCounter = 1;

    // Parcourez chaque div de population et mettez à jour les labels et les identifiants
    Array.from(allPopulations).forEach((populationDiv, index) => {
        // Mettre à jour l'identifiant du div de la population
        populationDiv.setAttribute('id', `population${index + 1}`);

        // Mettre à jour le label de la population
        const label = populationDiv.querySelector('label');
        if (label) {
            label.textContent = `Population ${index + 1}: `;
            label.setAttribute('for', `population${index + 1}`);
        }

        // Mettre à jour l'attribut onclick du bouton Remove pour refléter le nouvel index
        const removeButton = populationDiv.querySelector('button');
        if (removeButton) {
            removeButton.setAttribute('onclick', `removePopulation(${index + 1})`);
        }

        // Mettre à jour le compteur de population
        populationCounter++;
    });
}


function fillFormWithData(data) {
    if (data.process) {
        document.getElementById('processStatus').textContent = data.process;
    }
    if (data.image1 && data.image1.path) {
        document.getElementById('image1PathInput').value = data.image1.path;
    }
    if (data.image1 && data.image1.upperLayersToRemove) {
        document.getElementById('upperLayersToRemove1').value = data.image1.upperLayersToRemove;
    }
    if (data.image1 && data.image1.lowerLayersToRemove) {
        document.getElementById('lowerLayersToRemove1').value = data.image1.lowerLayersToRemove;
    }
    if (data.image2 && data.image2.path) {
        document.getElementById('image2PathInput').value = data.image2.path;
    }
    if (data.image2 && data.image2.upperLayersToRemove) {
        document.getElementById('upperLayersToRemove2').value = data.image2.upperLayersToRemove;
    }
    if (data.image2 && data.image2.lowerLayersToRemove) {
        document.getElementById('lowerLayersToRemove2').value = data.image2.lowerLayersToRemove;
    }
    if (data.configuration && data.configuration.saveAllImages) {
        if (document.getElementById('saveAllImages').value){
            document.getElementById('saveAllImages').checked = true;
        }
    }
    if (data.configuration && data.configuration.automaticMerging) {
        if (document.getElementById('automaticMerging').value){
            document.getElementById('automaticMerging').checked = true;
        }
    }
    if (data.configuration && data.configuration.useTagCenter) {
        if (document.getElementById('useTagCenter').value){
            document.getElementById('useTagCenter').checked = true;
        }
    }
    if (data.croppingValue) {
        document.getElementById('croppedvalue').value = data.croppingValue;
    }
    if (data.project_name) {
        document.getElementById('saveDirectoryInput').value = data.project_name;
    }
   
    // Tags Image 1
    if (data.image1 && data.image1.tags) {
        Object.entries(data.image1.tags).forEach(([tag, wavelength], index) => {
            addTag('1'); 
            const tagContainers = document.querySelectorAll('#tagsImage1 .tag-entry');
            const lastTagContainer = tagContainers[tagContainers.length - 1];
            lastTagContainer.querySelector('.wavelength').value = wavelength;
        });
    }
    // Tags Image 2
    if (data.image2 && data.image2.tags) {
        Object.entries(data.image2.tags).forEach(([tag, wavelength], index) => {
            addTag('2'); 
            const tagContainers = document.querySelectorAll('#tagsImage2 .tag-entry');
            const lastTagContainer = tagContainers[tagContainers.length - 1];
            lastTagContainer.querySelector('.wavelength').value = wavelength;
        });
    }
    // Populations
    if (data.populations) {
        Object.entries(data.populations).forEach((pop, index) => {
            addPopulation(); // Ajouter des champs supplémentaires si nécessaire
            const populationContainers = document.querySelectorAll('#populationsContainer .population-entry');
            const lastPopulationContainer = populationContainers[populationContainers.length - 1];
            pop[1].forEach((tag, tagIndex) => {
                const selectElement = lastPopulationContainer.querySelectorAll('select')[tagIndex];
                selectElement.value = tag;
            });
        });
    }

    if (data.phaloTag) {
        document.getElementById('phalloidinTagDropdown').value = data.phaloTag;
    }
}
