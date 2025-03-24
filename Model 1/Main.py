import gc
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

file_path = ""

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Data Generators
data_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

# Train Generator
train_generator = data_gen.flow_from_directory(
    directory=file_path,
    target_size=(120, 120),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Test Generator
test_generator = data_gen.flow_from_directory(
    directory=file_path,
    target_size=(120, 120),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Callback for tracking accuracy
class AccuracyLoggerCallback(Callback):
    def __init__(self, test_data, checkpoint_epochs):
        super().__init__()
        self.test_data = test_data
        self.checkpoint_epochs = checkpoint_epochs
        self.accuracy_log = {}

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) in self.checkpoint_epochs:
            _, accuracy = self.model.evaluate(self.test_data, verbose=0)
            self.accuracy_log[epoch + 1] = accuracy
            print(f"Accuracy at epoch {epoch + 1}: {accuracy * 100:.2f}%")

# Train function
def train_cnn_model(base_model, model_name):
    base_model = base_model(weights='imagenet', include_top=False, input_shape=(120, 120, 3))

    # Unfreeze last 20 layers for fine-tuning
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.95), loss='categorical_crossentropy', metrics=['accuracy'])

    accuracy_callback = AccuracyLoggerCallback(test_data=test_generator, checkpoint_epochs=[5, 10, 15, 20])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(train_generator, epochs=20, batch_size=32, verbose=1, callbacks=[accuracy_callback, early_stopping])

    return model, accuracy_callback.accuracy_log

# Models to compare
models = {
    "ResNet50": ResNet50,
}

model_accuracies = {}
classification_reports = {}

# Directory for saving reports
os.makedirs("reports", exist_ok=True)

# Train and compare models
for model_name, base_model in models.items():
    print(f"Training {model_name}...")
    model, accuracy_log = train_cnn_model(base_model, model_name)
    model_accuracies[model_name] = accuracy_log

    # Generate predictions and classification report
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = test_generator.classes

    report = classification_report(y_true_classes, y_pred_classes, target_names=test_generator.class_indices.keys())
    classification_reports[model_name] = report

    # Save report to file
    with open(f"reports/{model_name}_classification_report.txt", "w") as file:
        file.write(report)

    # Clear session and memory
    del model
    tf.keras.backend.clear_session()
    gc.collect()

# Determine best model
best_model_name = max(model_accuracies, key=lambda k: model_accuracies[k].get(20, 0))
best_model_accuracy = model_accuracies[best_model_name].get(20, 0)
print(f"Best Model: {best_model_name} with Accuracy at 20 epochs: {best_model_accuracy * 100:.2f}%")

# Print classification reports
for model_name, report in classification_reports.items():
    print(f"\nClassification Report for {model_name}:")
    print(report)
