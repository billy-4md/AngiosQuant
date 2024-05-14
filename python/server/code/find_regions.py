import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull

# Étape 1 : Charger les données
df = pd.read_csv('/Users/xavierdekeme/Desktop/segmentation_details_light.csv')

# Étape 2 : Regrouper les points par Instance et obtenir les contours
segments = {}
for instance, group in df.groupby('Instance'):
    if group['Probabilité'].values > 0.7:
        points = group[['Point X', 'Point Y']].values
        segments[instance] = points

# Étape 3 : Calculer le centre de chaque région
centers = {}
for instance, points in segments.items():
    if len(points) > 2:  # Nécessaire pour calculer le ConvexHull
        hull = ConvexHull(points)
        center = np.mean(points[hull.vertices, :], axis=0)
    else:
        center = np.mean(points, axis=0)
    centers[instance] = center

# Étape 4 : Afficher les centres
print("Centres des régions :")
for instance, center in centers.items():
    print(f"Instance {instance}: Centre = {center}")

# Optionnel : Visualiser les résultats
import matplotlib.pyplot as plt

plt.figure()
for points in segments.values():
    plt.plot(points[:, 0], points[:, 1], 'o')
for center in centers.values():
    plt.plot(center[0], center[1], 'rx')  # 'rx' pour marquer les centres avec des croix rouges
plt.show()
