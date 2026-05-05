import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from datetime import datetime

def visualize_embeddings_bs1(model, dataloader, device, num_samples=30):
    model.eval()
    all_a, all_p, all_n = [], [], []
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"plot_{now}.png"

    with torch.no_grad():
        for i, (anchor, positive, negative) in enumerate(dataloader):
            if i >= num_samples:
                break

            emb_a = model(anchor.to(device)).cpu().numpy()
            emb_p = model(positive.to(device)).cpu().numpy()
            emb_n = model(negative.to(device)).cpu().numpy()
            all_a.append(emb_a)
            all_p.append(emb_p)
            all_n.append(emb_n)

    combined = np.vstack([np.vstack(all_a), np.vstack(all_p), np.vstack(all_n)])
    tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
    embeddings_2d = tsne.fit_transform(combined)

    a_2d = embeddings_2d[:num_samples]
    p_2d = embeddings_2d[num_samples:2 * num_samples]
    n_2d = embeddings_2d[2 * num_samples:]
    print(a_2d)
    plt.figure(figsize=(10, 8))

    plt.scatter(a_2d[:, 0], a_2d[:, 1], c='blue', label='Anchors', alpha=0.6)
    plt.scatter(p_2d[:, 0], p_2d[:, 1], c='green', label='Positives', alpha=0.6)
    plt.scatter(n_2d[:, 0], n_2d[:, 1], c='red', label='Negatives', alpha=0.6)

    # Draw lines for the first 5 triplets to see the "gap"
    for i in range(30):
        plt.plot([a_2d[i, 0], p_2d[i, 0]], [a_2d[i, 1], p_2d[i, 1]], 'g--', alpha=0.3)  # Line to Positive
        plt.plot([a_2d[i, 0], n_2d[i, 0]], [a_2d[i, 1], n_2d[i, 1]], 'r--', alpha=0.3)  # Line to Negative

    plt.legend()
    plt.title("2D Visualization of Embedding Space (t-SNE)")
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    plt.close()

