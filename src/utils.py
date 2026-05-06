"""
Utility functions for visualization, metrics, and model evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_training_history(history, figsize=(12, 4)):
    """
    Plot training and validation accuracy/loss.
    
    Args:
        history: Keras training history object
        figsize: Figure size (width, height)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(8, 6)):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels (one-hot encoded or class indices)
        y_pred: Predicted labels (one-hot encoded or class indices)
        class_names: List of class names
        figsize: Figure size
    """
    # Convert one-hot to class indices if needed
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    return fig


def plot_classification_report(y_true, y_pred, class_names=None):
    """
    Print and visualize classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    # Convert one-hot to class indices if needed
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    )
    
    print("Classification Report:")
    print(report)
    
    return report


def plot_sample_predictions(images, true_labels, pred_labels, class_names, 
                            n_samples=9, figsize=(12, 10)):
    """
    Plot sample images with true and predicted labels.
    
    Args:
        images: Array of images
        true_labels: True labels (one-hot or class indices)
        pred_labels: Predicted labels (one-hot or class indices)
        class_names: List of class names
        n_samples: Number of samples to plot
        figsize: Figure size
    """
    # Convert one-hot to class indices if needed
    if true_labels.ndim > 1:
        true_labels = np.argmax(true_labels, axis=1)
    if pred_labels.ndim > 1:
        pred_labels = np.argmax(pred_labels, axis=1)
    
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    axes = axes.flatten()
    
    for idx in range(min(n_samples, len(images))):
        ax = axes[idx]
        
        # Plot image
        ax.imshow(images[idx])
        
        # Get labels
        true_label = class_names[true_labels[idx]]
        pred_label = class_names[pred_labels[idx]]
        
        # Color red if wrong, green if correct
        color = 'green' if true_labels[idx] == pred_labels[idx] else 'red'
        
        title = f"True: {true_label}\nPred: {pred_label}"
        ax.set_title(title, color=color, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def plot_roc_curves(y_true, y_pred_proba, class_names, figsize=(10, 8)):
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred_proba: Predicted probabilities
        class_names: List of class names
        figsize: Figure size
    """
    n_classes = len(class_names)
    
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], 
               label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    return fig, roc_auc


# Example usage
if __name__ == "__main__":
    print("Utility functions for model evaluation and visualization")
