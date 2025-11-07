"""
Model evaluation for Syngenta Crop Disease Classification

Performs comprehensive evaluation, generates reports, and visualizations.

"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
from pathlib import Path 

from . import config # <--- Change this to a relative import
from .model import get_last_conv_layer_name # For Grad-CAM
from .data_utils import load_class_indices, preprocess_image # For inference


# Set random seed for reproducibility
np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)

def evaluate_model_performance(model, test_generator, class_names):
    """
    Evaluates the model on the test set and generates a classification report.
    
    Args:
        model (keras.Model): The trained Keras model.
        test_generator (ImageDataGenerator): The test data generator.
        class_names (list): List of class names.
        
    Returns:
        tuple: (accuracy, predicted_classes, true_labels, report_df)
    """
    print("\n======================================================================")
    print("EVALUATING MODEL PERFORMANCE")
    print("======================================================================")
    
    test_generator.reset() # Ensure predictions start from the beginning
    
    # Get true labels
    true_labels = test_generator.classes
    
    # Generate predictions
    print("\nGenerating predictions on test set...")
    predictions = model.predict(test_generator, verbose=config.VERBOSE)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate overall accuracy
    accuracy = accuracy_score(true_labels, predicted_classes)
    print(f"\n======================================================================")
    print(f"OVERALL TEST ACCURACY: {accuracy*100:.2f}%")
    print(f"======================================================================")
    
    # Generate classification report
    report = classification_report(
        true_labels,
        predicted_classes,
        target_names=[name.replace('___', ' ') for name in class_names], # Clean names for report
        digits=3,
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    
    print("\nDETAILED CLASSIFICATION REPORT:")
    print(report_df)
    
    # Save classification report
    report_df.to_csv(config.CLASSIFICATION_REPORT_PATH)
    print(f"✓ Classification report saved to: {config.CLASSIFICATION_REPORT_PATH}")
    
    # Save evaluation metrics (accuracy)
    with open(config.EVALUATION_METRICS_PATH, 'w') as f:
        f.write(f"Overall Test Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"Number of classes: {len(class_names)}\n")
        f.write(f"Number of test samples: {test_generator.samples}\n")
    print(f"✓ Evaluation metrics saved to: {config.EVALUATION_METRICS_PATH}")
    
    return accuracy, predicted_classes, true_labels, report_df

def plot_confusion_matrix(true_labels, predicted_classes, class_names, save_path=None):
    """
    Plots a professional confusion matrix.
    
    Args:
        true_labels (np.array): True class labels.
        predicted_classes (np.array): Predicted class labels.
        class_names (list): List of class names.
        save_path (Path, optional): Path to save the plot. Defaults to None.
    """
    print("\n----------------------------------------------------------------------")
    print("PLOTTING CONFUSION MATRIX")
    print("----------------------------------------------------------------------")
    
    cm = confusion_matrix(true_labels, predicted_classes)
    
    plt.figure(figsize=config.CM_FIGSIZE)
    
    # Normalize confusion matrix to show percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=config.CM_ANNOT_FORMAT,
        cmap=config.CM_CMAP,
        xticklabels=[name.replace('___', ' ') for name in class_names],
        yticklabels=[name.replace('___', ' ') for name in class_names],
        cbar_kws={'label': 'Percentage (%)'}
    )
    
    plt.title('Confusion Matrix - Disease Classification', 
             fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"✓ Confusion matrix plot saved to: {save_path}")
    plt.show()

def visualize_sample_predictions(model, test_generator, class_names,
                                num_correct=5, num_incorrect=5, save_path=None):
    """
    Visualizes sample correct and incorrect predictions with their images.
    
    Args:
        model (keras.Model): The trained Keras model.
        test_generator (ImageDataGenerator): The test data generator.
        class_names (list): List of class names.
        num_correct (int): Number of correct predictions to show.
        num_incorrect (int): Number of incorrect predictions to show.
        save_path (Path, optional): Path to save the plot. Defaults to None.
    """
    print("\n----------------------------------------------------------------------")
    print("VISUALIZING SAMPLE PREDICTIONS")
    print("----------------------------------------------------------------------")
    
    test_generator.reset()
    all_images = []
    all_true_labels = []
    all_predicted_probs = []
    
    # Collect a few batches to ensure enough samples
    for i in range(min(len(test_generator), 5)): # Process up to 5 batches
        X_batch, y_batch = next(test_generator)
        y_pred_batch = model.predict(X_batch, verbose=0)
        
        all_images.append(X_batch)
        all_true_labels.append(np.argmax(y_batch, axis=1))
        all_predicted_probs.append(y_pred_batch)
        
    all_images = np.vstack(all_images)
    all_true_labels = np.concatenate(all_true_labels)
    all_predicted_probs = np.concatenate(all_predicted_probs)
    all_predicted_classes = np.argmax(all_predicted_probs, axis=1)

    correct_indices = np.where(all_true_labels == all_predicted_classes)[0]
    incorrect_indices = np.where(all_true_labels != all_predicted_classes)[0]

    # Ensure we don't request more samples than available
    num_correct = min(num_correct, len(correct_indices))
    num_incorrect = min(num_incorrect, len(incorrect_indices))

    # Randomly sample indices
    if num_correct > 0:
        sample_correct_indices = np.random.choice(correct_indices, num_correct, replace=False)
    else:
        sample_correct_indices = []
    
    if num_incorrect > 0:
        sample_incorrect_indices = np.random.choice(incorrect_indices, num_incorrect, replace=False)
    else:
        sample_incorrect_indices = []
    
    total_plots = num_correct + num_incorrect
    if total_plots == 0:
        print("No samples to visualize.")
        return

    fig, axes = plt.subplots(2, max(num_correct, num_incorrect), figsize=(15, 6))
    if max(num_correct, num_incorrect) == 0: return # Handle case with no plots

    # Plot correct predictions
    for i, idx in enumerate(sample_correct_indices):
        ax = axes[0, i] if max(num_correct, num_incorrect) > 1 else axes[0]
        ax.imshow(all_images[idx])
        true_label = class_names[all_true_labels[idx]]
        pred_label = class_names[all_predicted_classes[idx]]
        confidence = all_predicted_probs[idx, all_predicted_classes[idx]] * 100
        ax.set_title(f"True: {true_label.replace('___', ' ')}\nPred: {pred_label.replace('___', ' ')}\nConf: {confidence:.1f}%",
                     color='green', fontsize=9, fontweight='bold')
        ax.axis('off')
    # Hide empty subplots if num_correct < max
    for i in range(num_correct, max(num_correct, num_incorrect)):
        ax = axes[0, i] if max(num_correct, num_incorrect) > 1 else axes[0]
        ax.axis('off')

    # Plot incorrect predictions
    for i, idx in enumerate(sample_incorrect_indices):
        ax = axes[1, i] if max(num_correct, num_incorrect) > 1 else axes[1]
        ax.imshow(all_images[idx])
        true_label = class_names[all_true_labels[idx]]
        pred_label = class_names[all_predicted_classes[idx]]
        confidence = all_predicted_probs[idx, all_predicted_classes[idx]] * 100
        ax.set_title(f"True: {true_label.replace('___', ' ')}\nPred: {pred_label.replace('___', ' ')}\nConf: {confidence:.1f}%",
                     color='red', fontsize=9, fontweight='bold')
        ax.axis('off')
    # Hide empty subplots if num_incorrect < max
    for i in range(num_incorrect, max(num_correct, num_incorrect)):
        ax = axes[1, i] if max(num_correct, num_incorrect) > 1 else axes[1]
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    fig.suptitle('Sample Correct (Top) and Incorrect (Bottom) Predictions', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path)
        print(f"✓ Sample predictions plot saved to: {save_path}")
    plt.show()

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for a given image and model prediction.
    
    Args:
        img_array (np.array): Preprocessed image array (batch of 1).
        model (keras.Model): The trained Keras model.
        last_conv_layer_name (str): Name of the last convolutional layer.
        pred_index (int, optional): Index of the predicted class. Defaults to None (uses top prediction).
        
    Returns:
        np.array: The Grad-CAM heatmap.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization, clip values below 0 and normalize to [0,1]
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(model, image_path, class_names, save_path=None):
    """
    Loads an image, makes a prediction, and displays it with its Grad-CAM heatmap.
    
    Args:
        model (keras.Model): The trained Keras model.
        image_path (Path): Path to the image file.
        class_names (list): List of class names.
        save_path (Path, optional): Path to save the plot. Defaults to None.
    """
    print(f"\nVisualizing Grad-CAM for: {image_path.name}")
    
    # Preprocess image for model input
    img_array = preprocess_image(str(image_path), target_size=config.IMG_SIZE)
    
    # Original image for display
    original_img = keras.preprocessing.image.load_img(str(image_path), target_size=config.IMG_SIZE)
    original_img_array = keras.preprocessing.image.img_to_array(original_img)
    
    # Get prediction
    preds = model.predict(img_array, verbose=0)
    pred_class_idx = np.argmax(preds[0])
    pred_class_name = class_names[pred_class_idx]
    confidence = preds[0][pred_class_idx] * 100

    # Get last convolutional layer name from the base model
    last_conv_layer_name = get_last_conv_layer_name(model)
    if not last_conv_layer_name:
        print("Warning: Could not find a suitable convolutional layer for Grad-CAM.")
        print("Skipping Grad-CAM visualization.")
        return
    
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_class_idx)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    # Use jet colormap to colorize heatmap
    jet = plt.cm.get_cmap("jet")
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((original_img_array.shape[1], original_img_array.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * 0.4 + original_img_array
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_img)
    ax[0].set_title(f"Original Image\nTrue: {image_path.parent.name.replace('___', ' ')}", fontsize=10)
    ax[0].axis('off')

    ax[1].imshow(superimposed_img)
    ax[1].set_title(f"Predicted: {pred_class_name.replace('___', ' ')}\nConfidence: {confidence:.1f}%", fontsize=10)
    ax[1].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"✓ Grad-CAM plot saved to: {save_path}")
    plt.show()

def visualize_gradcam_for_samples(model, test_generator, class_names, num_samples=config.GRADCAM_NUM_SAMPLES, save_path=None):
    """
    Visualizes Grad-CAM heatmaps for a number of random samples from the test set.
    """
    print("\n======================================================================")
    print("GRAD-CAM VISUALIZATION")
    print("======================================================================")
    
    test_generator.reset()
    all_filepaths = test_generator.filepaths
    
    # Take random samples
    sample_indices = np.random.choice(len(all_filepaths), min(num_samples, len(all_filepaths)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 5 * num_samples))
    if num_samples == 1:
        axes = [axes] # Make it iterable for single sample
    
    last_conv_layer_name = get_last_conv_layer_name(model)
    if not last_conv_layer_name:
        print("Warning: Could not find a suitable convolutional layer for Grad-CAM. Skipping visualization.")
        plt.close(fig) # Close empty figure
        return

    for i, idx in enumerate(sample_indices):
        image_path = Path(all_filepaths[idx])
        
        # Preprocess image for model input
        img_array = preprocess_image(str(image_path), target_size=config.IMG_SIZE)
        
        # Original image for display
        original_img = keras.preprocessing.image.load_img(str(image_path), target_size=config.IMG_SIZE)
        original_img_array = keras.preprocessing.image.img_to_array(original_img)
        
        # Get prediction
        preds = model.predict(img_array, verbose=0)
        pred_class_idx = np.argmax(preds[0])
        pred_class_name = class_names[pred_class_idx]
        confidence = preds[0][pred_class_idx] * 100
        
        # Generate heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_class_idx)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)
        jet = plt.cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((original_img_array.shape[1], original_img_array.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        superimposed_img = jet_heatmap * 0.4 + original_img_array
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

        # Plot
        ax0 = axes[i, 0] if num_samples > 1 else axes[0]
        ax0.imshow(original_img)
        ax0.set_title(f"Original: {image_path.parent.name.replace('___', ' ')}", fontsize=10)
        ax0.axis('off')

        ax1 = axes[i, 1] if num_samples > 1 else axes[1]
        ax1.imshow(superimposed_img)
        ax1.set_title(f"Pred: {pred_class_name.replace('___', ' ')} (Conf: {confidence:.1f}%)", fontsize=10)
        ax1.axis('off')
        
    plt.tight_layout()
    plt.suptitle("Grad-CAM Visualizations for Sample Predictions", fontsize=16, fontweight='bold', y=1.02)
    if save_path:
        plt.savefig(save_path)
        print(f"✓ Grad-CAM samples plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    # Smoke test evaluation (requires a trained model and data_utils to run first)
    print("Testing Evaluation Pipeline (Smoke Test)...")
    
    # Ensure config is validated
    config.validate_config()

    # Create dummy class indices and a simple model for testing
    dummy_class_indices = {'ClassA': 0, 'ClassB': 1, 'ClassC': 2}
    class_names = list(dummy_class_indices.keys())
    
    # Create a dummy model (simplified, not full EfficientNet)
    dummy_model = keras.Sequential([
        keras.Input(shape=config.IMG_SIZE + (config.IMG_CHANNELS,)),
        layers.Conv2D(8, 3, activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(len(class_names), activation='softmax')
    ])
    dummy_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Create dummy test generator (need a directory to point to)
    # This requires data_utils to create processed/test directory
    from src.data_utils import DataPipeline
    data_pipeline = DataPipeline()
    data_pipeline.verify_dataset_structure() # To ensure classes are loaded
    _, _, test_gen, _ = data_pipeline.create_data_generators()

    # Use actual class names from the generator
    actual_class_names = list(test_gen.class_indices.keys())

    # --- Run Evaluation Functions ---
    print("\nRunning smoke evaluation functions...")
    acc, preds, trues, report = evaluate_model_performance(dummy_model, test_gen, actual_class_names)
    plot_confusion_matrix(trues, preds, actual_class_names, config.CONFUSION_MATRIX_FIGURE)
    visualize_sample_predictions(dummy_model, test_gen, actual_class_names, save_path=config.PREDICTIONS_FIGURE)
    visualize_gradcam_for_samples(dummy_model, test_gen, actual_class_names, num_samples=2, save_path=config.GRADCAM_FIGURE)
    
    print("\n✓ Evaluation pipeline smoke test completed successfully!")