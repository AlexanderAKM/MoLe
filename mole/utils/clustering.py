"""
Structure-based molecular clustering using fingerprint similarity.

Provides two variants:
- clustering(): RDKit fingerprints with random reference subsampling
- clustering_hce(): AtomPair fingerprints (better for HCE-style diversity)
"""

import os
import random as rd

import numpy as np
import pandas as pd
import rdkit.Chem as rdc
import rdkit.DataStructs as rdd
import rdkit.Chem.AllChem as rdca
import rdkit.Chem.Draw as rdcd
import sklearn.decomposition as skd
import sklearn.cluster as skc
import sklearn.metrics as skm
import matplotlib.pyplot as plt

rd.seed(0)
np.random.seed(0)


def clustering(data, target_column, output_dir, dataset_name='dataset'):
    fingerprint_size = 4096
    number_of_molecules = len(data['smiles'].values)
    number_of_references = 1200
    fingerprint_generator = rdca.GetRDKitFPGenerator(fpSize=fingerprint_size)

    fingerprints = [fingerprint_generator.GetFingerprint(rdc.MolFromSmiles(si)) for si in list(data['smiles'].values)]

    fingerprints_references = rd.sample(fingerprints, min(number_of_references, number_of_molecules))
    n_ref = len(fingerprints_references)

    similarities = np.zeros((number_of_molecules, n_ref), dtype=np.float32)
    for ri in range(number_of_molecules):
        similarities[ri, :] = rdd.BulkTanimotoSimilarity(fingerprints[ri], fingerprints_references)

    subset_size = int(0.05 * number_of_molecules) if number_of_molecules > 20000 else min(number_of_molecules, 1000)
    pca = skd.IncrementalPCA(n_components=n_ref)
    pca.fit(similarities[np.random.choice(similarities.shape[0], size=subset_size, replace=False), :])

    total_variance_explained = 0.80
    number_of_components = len([np.sum(pca.explained_variance_ratio_[:ni]) for ni in range(number_of_molecules) if np.sum(pca.explained_variance_ratio_[:ni]) < total_variance_explained])
    number_of_components = max(1, min(number_of_components, n_ref))
    print(f'Number of dominant principal components: {number_of_components}')

    pca = skd.IncrementalPCA(n_components=number_of_components)
    transformed_similarities = pca.fit_transform(similarities)

    silhouette_averages = list()
    lower_cluster_limit = 3
    upper_cluster_limit = 25
    clusterization_range = range(lower_cluster_limit, upper_cluster_limit)

    for ni in clusterization_range:
        clustering = skc.MiniBatchKMeans(n_clusters=ni, random_state=0)
        cluster_labels = clustering.fit_predict(transformed_similarities)
        silhouette_averages.append(skm.silhouette_score(transformed_similarities, cluster_labels))
        print("done")

    number_of_clusters = silhouette_averages.index(max(silhouette_averages)) + lower_cluster_limit
    print(f'Best number of clusters: {number_of_clusters}')

    clustering = skc.MiniBatchKMeans(n_clusters=number_of_clusters, random_state=0)
    cluster_labels = clustering.fit_predict(transformed_similarities)
    silhouette_average = skm.silhouette_score(transformed_similarities, cluster_labels)

    plt.plot(clusterization_range, silhouette_averages)
    plt.plot(number_of_clusters, silhouette_average, marker='x', color='black')
    plt.xlabel('Number of clusters')
    plt.ylabel('Average silhouette score')
    plt.show()

    data['cluster'] = cluster_labels

    os.makedirs(output_dir, exist_ok=True)

    for ci in range(number_of_clusters):
        smiles_list = list(data.loc[data['cluster'] == ci, 'smiles'])
        print(f"Cluster {ci}: {len(smiles_list)} Molecules")
        molecules_list = [rdc.MolFromSmiles(si) for si in smiles_list]
        grid = rdcd.MolsToGridImage(molecules_list[:18], returnPNG=False)
        grid.save(f'{output_dir}/{ci}.png')
    
    data.to_csv(f"{output_dir}/{dataset_name}.csv", index=False)
    
    return data


def clustering_hce(data, target_column, output_dir, dataset_name='dataset'):
    fingerprint_size = 4096
    number_of_molecules = len(data['smiles'].values)
    number_of_references = 1200
    fingerprint_generator = rdc.rdFingerprintGenerator.GetAtomPairGenerator(fpSize=fingerprint_size)

    fingerprints = [fingerprint_generator.GetFingerprint(rdc.MolFromSmiles(si)) for si in list(data['smiles'].values)]

    fingerprints_references = rd.sample(fingerprints, min(number_of_references, number_of_molecules))
    n_ref = len(fingerprints_references)

    similarities = np.zeros((number_of_molecules, n_ref), dtype=np.float32)
    for ri in range(number_of_molecules):
        similarities[ri, :] = rdd.BulkTanimotoSimilarity(fingerprints[ri], fingerprints_references)

    subset_size = int(0.05 * number_of_molecules) if number_of_molecules > 20000 else min(number_of_molecules, 1000)
    pca = skd.IncrementalPCA(n_components=n_ref)
    pca.fit(similarities[np.random.choice(similarities.shape[0], size=subset_size, replace=False), :])

    total_variance_explained = 0.65
    number_of_components = len([np.sum(pca.explained_variance_ratio_[:ni]) for ni in range(number_of_molecules) if np.sum(pca.explained_variance_ratio_[:ni]) < total_variance_explained])
    number_of_components = max(1, min(number_of_components, n_ref))
    print(f'Number of dominant principal components: {number_of_components}')

    pca = skd.IncrementalPCA(n_components=number_of_components)
    transformed_similarities = pca.fit_transform(similarities)

    silhouette_averages = list()
    lower_cluster_limit = 3
    upper_cluster_limit = 25
    clusterization_range = range(lower_cluster_limit, upper_cluster_limit)

    for ni in clusterization_range:
        clustering = skc.MiniBatchKMeans(n_clusters=ni, random_state=0)
        cluster_labels = clustering.fit_predict(transformed_similarities)
        silhouette_averages.append(skm.silhouette_score(transformed_similarities, cluster_labels))
        print("done")

    number_of_clusters = silhouette_averages.index(max(silhouette_averages)) + lower_cluster_limit
    print(f'Best number of clusters: {number_of_clusters}')

    clustering = skc.MiniBatchKMeans(n_clusters=number_of_clusters, random_state=0)
    cluster_labels = clustering.fit_predict(transformed_similarities)
    silhouette_average = skm.silhouette_score(transformed_similarities, cluster_labels)

    plt.plot(clusterization_range, silhouette_averages)
    plt.plot(number_of_clusters, silhouette_average, marker='x', color='black')
    plt.xlabel('Number of clusters')
    plt.ylabel('Average silhouette score')
    plt.show()

    data['cluster'] = cluster_labels

    os.makedirs(output_dir, exist_ok=True)

    for ci in range(number_of_clusters):
        smiles_list = list(data.loc[data['cluster'] == ci, 'smiles'])
        print(f"Cluster {ci}: {len(smiles_list)} Molecules")
        molecules_list = [rdc.MolFromSmiles(si) for si in smiles_list]
        grid = rdcd.MolsToGridImage(molecules_list[:18], returnPNG=False)
        grid.save(f'{output_dir}/{ci}.png')
    
    data.to_csv(f"{output_dir}/{dataset_name}.csv", index=False)
    
    return data
