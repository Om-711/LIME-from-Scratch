
from sklearn.linear_model import LinearRegression
import seaborn as sns

def lime_explainer_astgcn(model, batch, kernel_width=2, num_perturbations=50):
    device = batch['features'].device

    x = batch['features']  # shape: (B, T, N, F)
    B, T, N, F = x.shape
    input_np = x.cpu().detach().numpy()
    flat_input = input_np.flatten()

    perturbation = np.random.normal(loc=flat_input, scale=0.01, size=(num_perturbations, B * T * N * F))
    perturbation_tensor = torch.tensor(perturbation, dtype=torch.float32).to(device)
    perturbation_tensor = perturbation_tensor.view(num_perturbations, B, T, N, F)

    def predict_fn(perturbed_features):
        model.eval()
        preds = []
        for i in range(num_perturbations):
            perturbed_batch = {
                'features': perturbed_features[i],
                'adj':      batch['adj'],
                'v':        batch['v'],
                'theta':    batch['theta'],
                'D_ji':     batch['D_ji'],
                'cluster_indices': batch['cluster_indices'],
                'pm25':     perturbed_features[i][:, :, 0]  
            }
            with torch.no_grad():
                out = model(perturbed_batch)  # shape: (B, T, N, 1)
                preds.append(out.mean().item())  
        return np.array(preds)

    model_prediction = predict_fn(perturbation_tensor)

    # Compute kernel weights based on distance from original input
    distance = np.linalg.norm(perturbation - flat_input, axis=1)
    weight = np.exp(-((distance) ** 2) / (2 * (kernel_width ** 2)))

    # Fit a linear model
    lime_model = LinearRegression()
    lime_model.fit(perturbation, model_prediction, sample_weight=weight)
    feature_importance = lime_model.coef_

    # Reshape to original feature dimensions
    importance_tensor = feature_importance.reshape(B, T, N, F)
    node_importance = importance_tensor.mean(axis=(0, 1, 3))  # shape: (N,)

    # Plot top 50 important nodes
    num_display_nodes = min(50, N)
    selected_indices = np.argsort(node_importance)[-num_display_nodes:]

    # Node-wise Importance
    plt.figure(figsize=(14, 5))
    sns.barplot(x=selected_indices, y=node_importance[selected_indices])
    plt.xlabel("Node Index")
    plt.ylabel("Average Feature Importance")
    plt.title("Node-wise LIME Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig('Node_wise_importance.png', dpi=300)
    plt.show()

    # Feature-wise Importance Averaged over Nodes
    feature_importance_tensor = feature_importance.reshape(B, T, N, F)
    feature_mean_importance = feature_importance_tensor.mean(axis=(0, 1, 2))  # shape: (F,)
    f = [
        'COMMERCIAL ZONE',
        'INDUSTRIAL ZONE',
        'PM2.5',
        'RESIDENTIAL ZONE',
        'Speed',
        'TEMP.',
        'TRASPORT & COMMUNICATION ZONE',
        'humidity',
        'winddir',
        'windspeed'
    ]

    plt.figure(figsize=(10, 4))
    sns.barplot(x=f, y=feature_mean_importance)
    plt.xlabel("Feature")
    plt.ylabel("Average LIME Importance")
    plt.title("Feature-wise LIME Importance (Averaged over All Nodes, Time, Batch)")
    plt.xticks(rotation=45, ha="right")
    plt.savefig('Feature_wise_importance.png', dpi=300)
    plt.tight_layout()
    plt.show()


    return feature_importance, node_importance
