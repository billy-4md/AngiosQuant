import os
import sys
import subprocess
import socket


def get_local_ip_address():
    # Crée un socket pour se connecter à un serveur Internet public
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            # Utilise Google Public DNS server pour trouver l'adresse IP locale
            s.connect(("8.8.8.8", 80))
            ip_address = s.getsockname()[0]
        except Exception:
            ip_address = "127.0.0.1"  # Retourne localhost si une erreur survient
    return ip_address



def check_and_load_docker_image(image_name, tar_path):
    try:
        result = subprocess.run(["docker", "image", "inspect", image_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode == 0:
            print(f"Docker image '{image_name}' is already installed")
        else:
            print(f"Docker image '{image_name}' not found. Starting loading...")
            load_result = subprocess.run(["docker", "load", "-i", tar_path], stdout=subprocess.PIPE, text=True)
            if load_result.returncode == 0:
                print(f"Docker image '{image_name}' has been successful installed")
            else:
                print(f"Error while loading the docker image '{tar_path}'.")
    except Exception as e:
        print(f"Error while checking the docker image: {e}")



def load_and_run_docker_image_napari(image_path):
    print(image_path)
    docker_image_file = "napari-docker-image-git.tar" 
    docker_image_name = "ghcr.io/napari/napari:sha-4f4c063"
    rel_image_path = f"server/docker_images/{docker_image_file}"
    abs_image_path = os.path.abspath(rel_image_path)   

    #Load the docker image
    check_and_load_docker_image(docker_image_name, abs_image_path)

    #Launch container and get ID
    if sys.platform == "darwin":
        display_var = f"{get_local_ip_address()}:0"
    elif sys.platform == "win32":
        display_var = "host.docker.internal:0.0"
    else:
        display_var = ":0"
    
    try:

        container_result = subprocess.run( 
            ["docker", "run", "-it",
            "-ipc", "host",
            "-m", "1g",
             "-v", f"{image_path}:/app/image",
             docker_image_name,
             "bash"],
            capture_output=True, text=True, check=True
        )
        container_id = container_result.stdout.strip()

        subprocess.run(
            ["docker", "exec", container_id,
             "bash", "-c", "napari /app/image/billes 20X-1.czi && exit"],
            check=True
        )

    except subprocess.CalledProcessError as e:
        print(f"Error while executing Docker: {e}")

    finally:
        #Stop everything
        if 'container_id' in locals():
            subprocess.run(["docker", "stop", container_id])
            subprocess.run(["docker", "rm", container_id])






labels_to_remove = []
for label_num in unique_labels:
    if label_num == 0: 
        continue

    instance_mask = labels == label_num
    #circle_mask = masque_redimensionne_3d[z].astype(bool)
    circle_mask = create_mask_from_gray_image(normalize_and_convert_to_uint8(img_slice)).astype(bool)

    intersection = np.logical_and(instance_mask, circle_mask)
    proportion_in_circle = np.sum(intersection) / np.sum(instance_mask)

    if proportion_in_circle >= 0.5:
        labels[instance_mask] = 0 
        labels_to_remove.append(label_num)


# start_centers, start_radii = read_circle_data('/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/code/Ch1-T4_start.csv')
# mid_centers, mid_radii = read_circle_data('/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/code/Ch1-T4_middle.csv')
# end_centers, end_radii = read_circle_data('/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/code/Ch1-T4_end.csv')

# indices_first = match_circles(mid_centers, start_centers)
# indices_second = match_circles(mid_centers, end_centers)

# centers_global = find_general_center(start_centers, mid_centers, end_centers, indices_first, indices_second)
# interpolated_radii = interpolate_circles(start_radii[indices_first], mid_radii, end_radii[indices_second], n_slices=14)


# image_shape = (1024, 1024)
# img_3d_circles = create_3d_image(centers_global, interpolated_radii, n_slices=14, image_shape=image_shape)

# imageio.volwrite('3d_image.tif', img_3d_circles)


# Préparer le masque redimensionné
masque_redimensionne_3d = np.zeros((dic_dim['Z'], dic_dim['Y'], dic_dim['X']), dtype=bool)
for z in range(dic_dim['Z']):
    masque_redimensionne_3d[z] = resize(img_3d_circles[z], (dic_dim['Y'], dic_dim['X']), preserve_range=True).astype(bool)

imageio.volwrite('/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/hough_transform/3d_mask.tif', masque_redimensionne_3d)

# # Appliquer le masque à chaque tranche Z de l'image CZI
# # for z in range(dic_dim['Z']):
# #     image_czi_reduite[z, :, :][masque_redimensionne_3d[z]] = 0 









def icp(common_centers, centers_TAG7, max_iterations=100, tolerance=0.001):
    """
    Perform Iterative Closest Point (ICP) matching between common centers and TAG7 centers.
    """
    # Convertir les centres en numpy arrays pour faciliter les calculs
    src_points = np.array(list(common_centers.values()))
    dst_points = np.array(list(centers_TAG7.values()))

    prev_error = 0

    for i in range(max_iterations):
        # Trouver les correspondances les plus proches dans dst_points pour chaque point de src_points
        kdtree = KDTree(dst_points)
        distances, indices = kdtree.query(src_points)

        # Sélectionner les points correspondants
        matched_points = dst_points[indices]

        # Estimer la transformation
        T, _, _ = best_fit_transform(src_points, matched_points)

        # Transformer les points source
        src_points = (np.dot(T[:3, :3], src_points.T).T + T[:3, 3]).reshape(src_points.shape)

        # Calculer l'erreur
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # Calculer la transformation finale
    T, _, _ = best_fit_transform(np.array(list(common_centers.values())), src_points)
    
    return T

def best_fit_transform(A, B):
    """
    Calcule la meilleure transformation rigide (rotation + translation) qui aligne les points A sur les points B
    en minimisant la distance quadratique moyenne entre les points correspondants.
    Retourne la matrice de transformation, la rotation et la translation.
    """
    # Centrer les nuages de points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # Matrice de covariance
    H = np.dot(AA.T, BB)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Calculer la rotation
    R = np.dot(Vt.T, U.T)

    # S'assurer que la matrice de rotation est une rotation propre
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # Matrice de transformation
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T, R, t

def apply_transformation_3d(image_3d, transformation_matrix):
    # Assurez-vous que la matrice de transformation est de type float32
    transformation_matrix = transformation_matrix.astype(np.float32)

    # Extraire la sous-matrice 2x3 si nécessaire
    if transformation_matrix.shape == (4, 4):
        transformation_matrix = transformation_matrix[:2, :3]
    
    transformed_image_3d = np.zeros_like(image_3d)
    for z in range(image_3d.shape[0]):
        transformed_image_3d[z] = cv2.warpAffine(image_3d[z], transformation_matrix, (image_3d.shape[2], image_3d.shape[1]))
    
    return transformed_image_3d

def combine_images(image_3d_1, image_3d_2_transformed):
    combined_image_3d = np.zeros_like(image_3d_1)

    for z in range(image_3d_1.shape[0]):
        slice_1 = image_3d_1[z]
        slice_2_transformed = image_3d_2_transformed[z]

        # Assurez-vous que les valeurs combinées ne dépassent pas le maximum autorisé pour le type de données
        combined_slice = np.clip(slice_1 + slice_2_transformed, 0, np.iinfo(image_3d_1.dtype).max)

        combined_image_3d[z] = combined_slice
    
    return combined_image_3d

def plot_3d_point_fit(src_points, dst_points, transformed_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Points sources
    ax.scatter(src_points[:, 0], src_points[:, 1], src_points[:, 2], c='r', label='Source Points')

    # Points cibles
    ax.scatter(dst_points[:, 0], dst_points[:, 1], dst_points[:, 2], c='g', label='Target Points')

    # Points sources transformés
    ax.scatter(transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2], c='b', marker='^', label='Transformed Source Points')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def find_common_centers(centers_TAG1, centers_TAG2, distance_threshold=0.1, merge_threshold=0.05):
    common_centers = []
    centers_list_TAG1 = list(centers_TAG1.values())
    centers_list_TAG2 = list(centers_TAG2.values())

    # Identifier les paires de centres en dessous du seuil de distance
    for center1 in centers_list_TAG1:
        for center2 in centers_list_TAG2:
            # Convertir les tuples en tableaux NumPy avant de soustraire
            dist = np.linalg.norm(np.array(center1) - np.array(center2))
            if dist <= distance_threshold:
                common_center = np.mean([center1, center2], axis=0)
                common_centers.append(common_center)

    # Fusionner les centres proches
    merged_common_centers = []
    while common_centers:
        center = common_centers.pop(0)
        to_merge = [center]

        # Indices des éléments à supprimer
        to_remove = []

        # Trouver d'autres centres à fusionner
        for i, other_center in enumerate(common_centers):
            if np.linalg.norm(center - other_center) <= merge_threshold:
                to_merge.append(other_center)
                to_remove.append(i)

        # Supprimer les éléments à fusionner de la liste
        for i in sorted(to_remove, reverse=True):
            del common_centers[i]

        # Ajouter la moyenne des centres à fusionner à la liste finale
        merged_center = np.mean(to_merge, axis=0)
        merged_common_centers.append(merged_center)

    # Convertir la liste des centres fusionnés en dictionnaire
    merged_common_centers_dict = {i + 1: center for i, center in enumerate(merged_common_centers)}

    return merged_common_centers_dict

def associate_nuclei(common_centers, centers_TAG1, centers_TAG2, centers_TAG3, labels_TAG1, labels_TAG2, labels_TAG3, distance_threshold=0.05):
    associated_labels = {}
    label_id = 1

    # Initialiser labels_to_keep avec une liste vide pour chaque TAG
    labels_to_keep = {'TAG1': [], 'TAG2': [], 'TAG3': []}

    # Parcourir chaque centre commun
    for common_center in common_centers.values():
        # Initialiser le dictionnaire pour les centres sélectionnés de chaque TAG
        selected_centers = {'TAG1': None, 'TAG2': None, 'TAG3': None}

        # Parcourir chaque ensemble de TAGs
        for tag, centers_dict in [('TAG1', centers_TAG1), ('TAG2', centers_TAG2), ('TAG3', centers_TAG3)]:
            distances = cdist([common_center], list(centers_dict.values()))[0]
            nearest_index = np.argmin(distances)
            nearest_distance = distances[nearest_index]

            # Vérifier si la distance est en dessous du seuil
            if nearest_distance <= distance_threshold:
                nearest_label = list(centers_dict.keys())[nearest_index]
                selected_centers[tag] = centers_dict[nearest_label]

                # Ajouter le label à conserver pour ce TAG
                labels_to_keep[tag].append(nearest_label)

        # Si tous les TAGs ont des centres sélectionnés, ajouter à associated_labels
        if all(selected_centers.values()):
            associated_labels[label_id] = selected_centers
            label_id += 1

    # Mettre à jour les images de labels pour chaque TAG en utilisant labels_to_keep
    updated_labels_TAG1 = update_labels_image(labels_TAG1, labels_to_keep['TAG1'])
    updated_labels_TAG2 = update_labels_image(labels_TAG2, labels_to_keep['TAG2'])
    updated_labels_TAG3 = update_labels_image(labels_TAG3, labels_to_keep['TAG3'])

    return associated_labels, updated_labels_TAG1, updated_labels_TAG2, updated_labels_TAG3

def combine_and_find_shared_population(centers_TAG1, centers_TAG2, centers_TAG3, distance_threshold=0.05):
    # Combiner les nuages de points et créer un tableau d'étiquettes
    combined_points = np.vstack((centers_TAG1, centers_TAG2, centers_TAG3))
    labels = np.array(['TAG1'] * len(centers_TAG1) + ['TAG2'] * len(centers_TAG2) + ['TAG3'] * len(centers_TAG3))

    # Calculer la matrice de distance
    distance_matrix = cdist(combined_points, combined_points)

    shared_population = []

    # Parcourir chaque point du premier TAG
    for i, (point, label) in enumerate(zip(combined_points, labels)):
        if label != 'TAG1':
            continue  # On commence par les points du premier TAG

        # Trouver les points les plus proches pour les deux autres TAGs
        nearest_TAG2_idx = nearest_TAG3_idx = None
        nearest_TAG2_dist = nearest_TAG3_dist = np.inf

        for j, (other_point, other_label) in enumerate(zip(combined_points, labels)):
            if i == j:
                continue  # Ignorer la distance avec soi-même
            dist = distance_matrix[i, j]
            if other_label == 'TAG2' and dist < nearest_TAG2_dist:
                nearest_TAG2_idx, nearest_TAG2_dist = j, dist
            elif other_label == 'TAG3' and dist < nearest_TAG3_dist:
                nearest_TAG3_idx, nearest_TAG3_dist = j, dist

        # Vérifier si les points les plus proches sont en dessous du seuil de distance
        if nearest_TAG2_dist <= distance_threshold and nearest_TAG3_dist <= distance_threshold:
            shared_population.append({
                'TAG1_point': point,
                'TAG2_point': combined_points[nearest_TAG2_idx],
                'TAG3_point': combined_points[nearest_TAG3_idx]
            })

    return shared_population

def update_labels_with_shared_centers(common_centers, centers_TAG, labels_image_TAG, distance_threshold=0.05):
    updated_labels_image = np.zeros_like(labels_image_TAG)

    # Convertir les coordonnées des centres TAG en un tableau NumPy
    centers_coordinates = np.array(list(centers_TAG.values()), dtype=float)

    for shared_center in common_centers.values():
        # Convertir le centre partagé (un tableau NumPy) directement en un tableau 2D pour cdist
        shared_center_2d = shared_center.reshape(1, -1)

        # Calculer la distance entre le centre partagé et tous les centres du TAG
        distances = cdist(shared_center_2d, centers_coordinates)

        nearest_center_idx = np.argmin(distances)
        if distances[0, nearest_center_idx] <= distance_threshold:
            # Trouver le label du centre le plus proche dans l'image de labels
            nearest_center_label = list(centers_TAG.keys())[nearest_center_idx]
            # Conserver uniquement les pixels avec le label identifié
            updated_labels_image[labels_image_TAG == nearest_center_label] = nearest_center_label

    return updated_labels_image




def keep_best_matches(centers_TAG1, centers_TAG2, centers_TAG3, distance_threshold=0.1, weights=(1, 1, 1)):
    # Appliquer les poids aux coordonnées des centres
    def apply_weights(coordinates, weights):
        return np.multiply(coordinates, weights)

    # Appliquer les poids
    coordinates_TAG1_weighted = apply_weights(np.array(list(centers_TAG1.values())), weights)
    coordinates_TAG2_weighted = apply_weights(np.array(list(centers_TAG2.values())), weights)
    coordinates_TAG3_weighted = apply_weights(np.array(list(centers_TAG3.values())), weights)

    matches = []

    # Calculer les distances entre tous les points pondérés des trois groupes de tags
    for label1, point1_weighted in zip(centers_TAG1.keys(), coordinates_TAG1_weighted):
        for label2, point2_weighted in zip(centers_TAG2.keys(), coordinates_TAG2_weighted):
            for label3, point3_weighted in zip(centers_TAG3.keys(), coordinates_TAG3_weighted):
                # Calculer la distance moyenne entre les trois points
                distance = np.mean(cdist([point1_weighted], [point2_weighted]) +
                                   cdist([point1_weighted], [point3_weighted]) +
                                   cdist([point2_weighted], [point3_weighted]))
                
                # Ajouter la correspondance et la distance à la liste si elle est inférieure au seuil
                if distance <= distance_threshold:
                    matches.append(((label1, label2, label3), distance))

    # Trier les correspondances par distance (qualité de la correspondance)
    matches.sort(key=lambda x: x[1])

    # Conserver uniquement les deux meilleures correspondances
    best_matches = matches[:2]

    # Préparer les dictionnaires mis à jour pour chaque groupe de tags
    updated_centers_TAG1 = {match[0][0]: centers_TAG1[match[0][0]] for match in best_matches}
    updated_centers_TAG2 = {match[0][1]: centers_TAG2[match[0][1]] for match in best_matches}
    updated_centers_TAG3 = {match[0][2]: centers_TAG3[match[0][2]] for match in best_matches}

    return updated_centers_TAG1, updated_centers_TAG2, updated_centers_TAG3

def keep_regions_with_all_tags(centers_TAG1, centers_TAG2, centers_TAG3, distance_threshold=0.1, weights=(1, 1, 0.75)):
    updated_centers_TAG1 = {}
    updated_centers_TAG2 = {}
    updated_centers_TAG3 = {}

    # Appliquer les poids aux coordonnées des centres
    def apply_weights(coordinates, weights):
        return np.multiply(coordinates, weights)

    coordinates_TAG1_weighted = apply_weights(np.array(list(centers_TAG1.values())), weights)
    coordinates_TAG2_weighted = apply_weights(np.array(list(centers_TAG2.values())), weights)
    coordinates_TAG3_weighted = apply_weights(np.array(list(centers_TAG3.values())), weights)

    for tag_dict, tag_coords_weighted, other_coords1_weighted, other_coords2_weighted, updated_dict in [
        (centers_TAG1, coordinates_TAG1_weighted, coordinates_TAG2_weighted, coordinates_TAG3_weighted, updated_centers_TAG1),
        (centers_TAG2, coordinates_TAG2_weighted, coordinates_TAG1_weighted, coordinates_TAG3_weighted, updated_centers_TAG2),
        (centers_TAG3, coordinates_TAG3_weighted, coordinates_TAG1_weighted, coordinates_TAG2_weighted, updated_centers_TAG3)]:

        for label, point in tag_dict.items():
            point_weighted = np.multiply(point, weights)

            distances_to_other1 = cdist([point_weighted], other_coords1_weighted)
            distances_to_other2 = cdist([point_weighted], other_coords2_weighted)

            nearest_other1_idx = np.argmin(distances_to_other1) if np.any(distances_to_other1 <= distance_threshold) else None
            nearest_other2_idx = np.argmin(distances_to_other2) if np.any(distances_to_other2 <= distance_threshold) else None

            if nearest_other1_idx is not None and nearest_other2_idx is not None:
                updated_dict[label] = point
                if tag_dict is centers_TAG1:
                    updated_centers_TAG2[list(centers_TAG2.keys())[nearest_other1_idx]] = other_coords1_weighted[nearest_other1_idx] / weights
                    updated_centers_TAG3[list(centers_TAG3.keys())[nearest_other2_idx]] = other_coords2_weighted[nearest_other2_idx] / weights
                elif tag_dict is centers_TAG2:
                    updated_centers_TAG1[list(centers_TAG1.keys())[nearest_other1_idx]] = other_coords1_weighted[nearest_other1_idx] / weights
                    updated_centers_TAG3[list(centers_TAG3.keys())[nearest_other2_idx]] = other_coords2_weighted[nearest_other2_idx] / weights
                else:  # tag_dict is centers_TAG3
                    updated_centers_TAG1[list(centers_TAG1.keys())[nearest_other1_idx]] = other_coords1_weighted[nearest_other1_idx] / weights
                    updated_centers_TAG2[list(centers_TAG2.keys())[nearest_other2_idx]] = other_coords2_weighted[nearest_other2_idx] / weights

    return updated_centers_TAG1, updated_centers_TAG2, updated_centers_TAG3

def keep_matching_regions_across_tags_vol(centers_TAG1, centers_TAG2, centers_TAG3, vol_TAG1, vol_TAG2, vol_TAG3, distance_threshold=0.125, volume_threshold=0.1, weights=(1, 1, 0)):
    updated_centers_TAG1, updated_centers_TAG2, updated_centers_TAG3 = {}, {}, {}
    unified_label_counter = 1

    def find_volume_matches(vol1, vol2, vol3, volume_threshold):
        volume_matches = []
        for label1, volume1 in vol1.items():
            for label2, volume2 in vol2.items():
                if abs(volume1 - volume2) > volume_threshold * volume1:
                    continue
                for label3, volume3 in vol3.items():
                    if abs(volume1 - volume3) <= volume_threshold * volume1 and abs(volume2 - volume3) <= volume_threshold * volume2:
                        volume_matches.append((label1, label2, label3))

        return volume_matches

    def apply_weights(coordinates, weights):
        return np.multiply(coordinates, weights)

    def validate_spatial_matches(volume_matches, centers1, centers2, centers3, distance_threshold, weights, updated_centers1, updated_centers2, updated_centers3, unified_label_counter):
        score_label = {}
        for labels in volume_matches:
            point1 = apply_weights(centers1[int(labels[0].split(" ")[-1])], weights)
            point2 = apply_weights(centers2[int(labels[1].split(" ")[-1])], weights)
            point3 = apply_weights(centers3[int(labels[2].split(" ")[-1])], weights)

            dist12 = np.linalg.norm(point1 - point2) #Permet de faire des calculs de distance x,y et z
            dist13 = np.linalg.norm(point1 - point3)
            dist23 = np.linalg.norm(point2 - point3)

            score_tot = dist12 * dist13 * dist23
            score_label[labels] = score_tot
            
            # if dist12 <= distance_threshold and dist13 <= distance_threshold and dist23 <= distance_threshold:
            #     unified_label = f"Unified {unified_label_counter}"
            #     unified_label_counter += 1
            #     updated_centers1[labels[0]] = {'point': centers1[int(labels[0].split(" ")[-1])], 'unified_label': unified_label}
            #     updated_centers2[labels[1]] = {'point': centers2[int(labels[1].split(" ")[-1])], 'unified_label': unified_label}
            #     updated_centers3[labels[2]] = {'point': centers3[int(labels[2].split(" ")[-1])], 'unified_label': unified_label}

        sorted_matches = sorted(score_label.items(), key=lambda item: item[1])

        # Conserver uniquement les X meilleurs triplets
        best_matches = sorted_matches[:2]
        for label, score in best_matches:
            print(score_label[label])

        unified_label_counter = 1  # Initialiser le compteur pour les labels unifiés

        for labels, score in best_matches:
            # Extraire les points correspondants pour chaque label
            point1 = centers1[int(labels[0].split(" ")[-1])]
            point2 = centers2[int(labels[1].split(" ")[-1])]
            point3 = centers3[int(labels[2].split(" ")[-1])]

            # Attribuer un label unifié aux points conservés
            unified_label = f"Unified {unified_label_counter}"
            updated_centers1[labels[0]] = {'point': point1, 'unified_label': unified_label}
            updated_centers2[labels[1]] = {'point': point2, 'unified_label': unified_label}
            updated_centers3[labels[2]] = {'point': point3, 'unified_label': unified_label}

            unified_label_counter += 1  # Incrémenter le compteur de labels unifiés pour le prochain match

        return updated_centers1, updated_centers2, updated_centers3

    volume_matches = find_volume_matches(vol_TAG1, vol_TAG2, vol_TAG3, volume_threshold)
    print(volume_matches)
    unified_label_counter = validate_spatial_matches(volume_matches, centers_TAG1, centers_TAG2, centers_TAG3, distance_threshold, weights, updated_centers_TAG1, updated_centers_TAG2, updated_centers_TAG3, unified_label_counter)

    return updated_centers_TAG1, updated_centers_TAG2, updated_centers_TAG3

def keep_regions_with_all_tags_volumes(centers_TAG1, centers_TAG2, centers_TAG3, volumes_TAG1, volumes_TAG2, volumes_TAG3, distance_threshold=0.25, volume_threshold=1, weights=(1, 1, 0.75)):
    updated_centers_TAG1 = {}
    updated_centers_TAG2 = {}
    updated_centers_TAG3 = {}

    # Appliquer les poids aux coordonnées des centres
    def apply_weights(coordinates, weights):
        return np.multiply(coordinates, weights)

    coordinates_TAG1_weighted = apply_weights(np.array(list(centers_TAG1.values())), weights)
    coordinates_TAG2_weighted = apply_weights(np.array(list(centers_TAG2.values())), weights)
    coordinates_TAG3_weighted = apply_weights(np.array(list(centers_TAG3.values())), weights)

    for tag_dict, tag_coords_weighted, other_coords1_weighted, other_coords2_weighted, other_volumes1, other_volumes2, updated_dict in [
    (centers_TAG1, coordinates_TAG1_weighted, coordinates_TAG2_weighted, coordinates_TAG3_weighted, volumes_TAG2, volumes_TAG3, updated_centers_TAG1),
    (centers_TAG2, coordinates_TAG2_weighted, coordinates_TAG1_weighted, coordinates_TAG3_weighted, volumes_TAG1, volumes_TAG3, updated_centers_TAG2),
    (centers_TAG3, coordinates_TAG3_weighted, coordinates_TAG1_weighted, coordinates_TAG2_weighted, volumes_TAG1, volumes_TAG2, updated_centers_TAG3)]:

        tag2_keys = list(centers_TAG2.keys())
        tag3_keys = list(centers_TAG3.keys())

        for label, point in centers_TAG1.items():
            point_weighted = np.multiply(point, weights)

            # Calculer les distances aux autres points
            distances_to_tag2 = cdist([point_weighted], coordinates_TAG2_weighted)
            distances_to_tag3 = cdist([point_weighted], coordinates_TAG3_weighted)

            # Trouver les points les plus proches dans TAG2 et TAG3
            nearest_tag2_idx = np.argmin(distances_to_tag2) if np.any(distances_to_tag2 <= distance_threshold) else None
            nearest_tag3_idx = np.argmin(distances_to_tag3) if np.any(distances_to_tag3 <= distance_threshold) else None

            # Utiliser les clés directement pour vérifier la similarité des volumes
            if nearest_tag2_idx is not None and nearest_tag3_idx is not None:
                nearest_tag2_key = tag2_keys[nearest_tag2_idx]
                nearest_tag3_key = tag3_keys[nearest_tag3_idx]

                try:
                    volume_similarity1 = abs(volumes_TAG1[label] - volumes_TAG2[nearest_tag2_key]) / volumes_TAG1[label] < volume_threshold
                except KeyError:
                    continue
                try:
                    volume_similarity2 = abs(volumes_TAG1[label] - volumes_TAG3[nearest_tag3_key]) / volumes_TAG1[label] < volume_threshold
                except KeyError:
                    continue  

                if volume_similarity1 and volume_similarity2:
                    # Mettre à jour les dictionnaires des centres en fonction de la similarité des volumes
                    updated_centers_TAG1[label] = centers_TAG1[label]
                    updated_centers_TAG2[nearest_tag2_key] = centers_TAG2[nearest_tag2_key]
                    updated_centers_TAG3[nearest_tag3_key] = centers_TAG3[nearest_tag3_key]

    return updated_centers_TAG1, updated_centers_TAG2, updated_centers_TAG3









# #TAG1
# labels_all_slices, data = do_segmentation(model, dic_dim_1, TAG1_img, 600, 75)
# labels_all_slices = np.stack(labels_all_slices, axis=0)
# df = pd.DataFrame(data)
# #plot_label_areas(labels_all_slices)
# labels_TAG_1 = reassign_labels(labels_all_slices, df)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_{IMAGE_NAME}_TAG1.tif', labels_TAG_1)

# centers_TAG1 = find_centers_of_labels_in_3d(labels_TAG_1)
# centers_TAG1_norm = normalize_centers(labels_TAG_1)



# #TAG2
# labels_all_slices, data = do_segmentation(model, dic_dim_1, TAG2_img, 650, 25)
# labels_all_slices = np.stack(labels_all_slices, axis=0)
# df = pd.DataFrame(data)
# labels_TAG_2 = reassign_labels(labels_all_slices, df)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_{IMAGE_NAME}_TAG2.tif', labels_TAG_2)

# centers_TAG2 = find_centers_of_labels_in_3d(labels_TAG_2)
# centers_TAG2_norm = normalize_centers(labels_TAG_2)


# #TAG3
# labels_all_slices, data = do_segmentation(model, dic_dim_1, TAG3_img, 650, 25)
# labels_all_slices = np.stack(labels_all_slices, axis=0)
# df = pd.DataFrame(data)
# labels_TAG_3 = reassign_labels(labels_all_slices, df)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_{IMAGE_NAME}_TAG3.tif', labels_TAG_3)

# centers_TAG3 = find_centers_of_labels_in_3d(labels_TAG_3)
# centers_TAG3_norm = normalize_centers(labels_TAG_3)

# #TAG4
# labels_all_slices, data = do_segmentation(model, dic_dim_2, TAG4_img, 650, 25)
# labels_all_slices = np.stack(labels_all_slices, axis=0)
# df = pd.DataFrame(data)
# labels_TAG_4 = reassign_labels(labels_all_slices, df)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_{IMAGE_NAME_2}_TAG4.tif', labels_TAG_4)

# centers_TAG4 = find_centers_of_labels_in_3d(labels_TAG_7)
# centers_TAG4_norm = normalize_centers(labels_TAG_7)

# #TAG7
# labels_all_slices, data = do_segmentation(model, dic_dim_2, TAG7_img, 650, 75)
# labels_all_slices = np.stack(labels_all_slices, axis=0)
# df = pd.DataFrame(data)
# labels_TAG_7 = reassign_labels(labels_all_slices, df)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_{IMAGE_NAME_2}_TAG7.tif', labels_TAG_7)

# centers_TAG7 = find_centers_of_labels_in_3d(labels_TAG_7)
# centers_TAG7_norm = normalize_centers(labels_TAG_7)



# #FIND POPULATION
# update_center1, update_center2, update_center7  = keep_regions_with_all_tags(centers_TAG1_norm, centers_TAG2_norm, centers_TAG7_norm)

# print(update_center1)
# print(update_center2)
# print(update_center7)


# updated_labels_images = update_labels_image(labels_TAG_1, update_center1)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_{IMAGE_NAME}_TAG1_update_pop127.tif', updated_labels_images)
# updated_labels_images = update_labels_image(labels_TAG_2, update_center2)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_{IMAGE_NAME}_TAG2_update_pop127.tif', updated_labels_images)
# updated_labels_images = update_labels_image(labels_TAG_7, update_center7)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_{IMAGE_NAME_2}_TAG7_update_pop127.tif', updated_labels_images)



# update_center1, update_center2, update_center3  = keep_regions_with_all_tags(centers_TAG1_norm, centers_TAG2_norm, centers_TAG3_norm)

# print(update_center1)
# print(update_center2)
# print(update_center3)


# updated_labels_images = update_labels_image(labels_TAG_1, update_center1)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_{IMAGE_NAME}_TAG1_update_pop123.tif', updated_labels_images)
# updated_labels_images = update_labels_image(labels_TAG_2, update_center2)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_{IMAGE_NAME}_TAG2_update_pop123.tif', updated_labels_images)
# updated_labels_images = update_labels_image(labels_TAG_3, update_center3)
# imageio.volwrite(f'/Users/xavierdekeme/Desktop/ULB-MA2/Memoire/Outil2.0/server/project/New/label_{IMAGE_NAME}_TAG3_update_pop123.tif', updated_labels_images)




