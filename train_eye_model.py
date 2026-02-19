
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
import pickle

# Configuration
DATA_DIR = 'eyes data' # Ensure this points to correct folder relative to script
IMG_SIZE = (32, 32) # Smaller size for MLP
# Note: MLP is fully connected, so resizing to 32x32 is better than 64x64 unless very powerful machine.

def parse_label(filename):
    # Filename format example: s001_00123_0_0_0_0_0_01.png
    # Index 4 is eye state: 0 (close), 1 (open)
    parts = filename.split('_')
    try:
        # Check against expected length or format if needed
        return int(parts[4])
    except (IndexError, ValueError):
        return None

def load_data(data_dir):
    images = []
    labels = []
    
    # Walk through subject directories
    print(f"Scanning {data_dir}...")
    count = 0 
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.png'):
                label = parse_label(file)
                if label is not None and label in [0, 1]:
                    try:
                        img_path = os.path.join(root, file)
                        img = Image.open(img_path).convert('L') # Grayscale
                        img = img.resize(IMG_SIZE)
                        img_array = np.array(img).flatten() / 255.0 # Flatten for MLP
                        images.append(img_array)
                        labels.append(label)
                        count += 1
                        if count % 500 == 0:
                            print(f"Loaded {count} images...")
                    except Exception as e:
                        print(f"Error loading {file}: {e}")
                        
    return np.array(images), np.array(labels)

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} not found.")
        return

    print("Loading data...")
    X, y = load_data(DATA_DIR)
    
    if len(X) == 0:
        print("No images found or parsed correctly. Check your data directory structure.")
        return

    print(f"Data loaded: {len(X)} images")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training on {len(X_train)}, Testing on {len(X_test)}")
    
    # Create MLP Classifier
    # Hidden layers: (100, 50) roughly
    clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, alpha=1e-4,
                        solver='adam', verbose=10, random_state=1,
                        learning_rate_init=.001)
    
    print("Training MLP model...")
    clf.fit(X_train, y_train)
    
    print("\nEvaluating model...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=['Closed', 'Open'])
    print(report)
    
    # Save model
    with open('eye_state_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print("Model saved to eye_state_model.pkl")
    
    # Save results to text file for README
    with open('training_results.txt', 'w') as f:
        f.write(f"Test Accuracy: {accuracy*100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)

if __name__ == "__main__":
    main()
