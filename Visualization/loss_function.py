import matplotlib.pyplot as plt


def plot_loss(train_losses, test_losses):
    plt.figure(figsize=(10, 6))

    # Línea de entrenamiento
    plt.plot(train_losses, label='Training Loss', color='#1f77b4', linewidth=2)

    # Línea de validación (usualmente en naranja o rojo para contrastar)
    plt.plot(test_losses, label='Test Loss', color='#ff7f0e', linewidth=2, linestyle='--')

    # Configuración de etiquetas y título
    plt.title('Loss Curve: Training vs Test', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)

    # Estética
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.ylim(bottom=0)  # Mantiene el inicio en 0

    plt.show()


