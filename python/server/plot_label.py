import numpy as np
import matplotlib.pyplot as plt
import imageio

def plot_label_areas(labels_3d):
    depth = labels_3d.shape[0]
    mid_layer = depth // 2
    start_layer = max(0, mid_layer - 10)
    end_layer = min(depth, mid_layer + 10)  # +1 pour inclure la couche end_layer

    # Calcul du nombre de couches à afficher
    display_depth = end_layer - start_layer
    
    rows = int(np.ceil(np.sqrt(display_depth)))
    cols = int(np.ceil(display_depth / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    axes_flat = axes.flatten()

    # Itération seulement sur les couches spécifiées
    for idx, z in enumerate(range(start_layer, end_layer)):
        layer = labels_3d[z, :, :]
        unique_labels, counts = np.unique(layer, return_counts=True)

        valid_indices = unique_labels != 0
        unique_labels = unique_labels[valid_indices]
        counts = counts[valid_indices]

        # Tri des labels et des aires
        sorted_areas_indices = np.argsort(counts)
        sorted_labels = unique_labels[sorted_areas_indices]
        sorted_areas = counts[sorted_areas_indices]

        # Calcul de la valeur moyenne de l'aire
        mean_area = np.mean(sorted_areas)

        # Création du graphique en bâtonnets pour chaque couche avec les labels triés
        axes_flat[idx].bar(range(len(sorted_labels)), sorted_areas, color='skyblue', tick_label=sorted_labels)
        axes_flat[idx].set_title(f'Layer {z + 1}')
        axes_flat[idx].set_xlabel('Sorted Label Number')
        axes_flat[idx].set_ylabel('Area (pixels)')

        # Ajout d'une ligne rouge à la valeur moyenne et affichage de la valeur
        axes_flat[idx].axhline(y=mean_area, color='red', linestyle='-', label=f'Mean Area: {mean_area:.2f}')
        axes_flat[idx].legend()

        # Afficher la valeur moyenne sur le graphique
        axes_flat[idx].text(0.5, mean_area, f'{mean_area:.2f}', color='red', verticalalignment='bottom', horizontalalignment='center')

        # Rotation des étiquettes sur l'axe x pour une meilleure lisibilité
        for label in axes_flat[idx].get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')

    # Masquer les axes inutilisés
    for ax in axes_flat[idx + 1:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()






label =  imageio.volread("D:\\Users\\MFE\\Xavier\\Result_Rapport2_new\\label_image1_processed_non_processed.tif")   
plot_label_areas(label)
