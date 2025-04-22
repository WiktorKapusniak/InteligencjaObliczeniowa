import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

filepaths = []
labels = []

for fname in os.listdir("dogs-cats-mini"):
    if fname.endswith(".jpg"):
        label = fname.split('.')[0]  # 'cat' albo 'dog'
        filepaths.append(os.path.join("dogs-cats-mini", fname))
        labels.append(label)

df = pd.DataFrame({
    'filename': filepaths,
    'category': labels
})

df['category'].value_counts().plot.bar()
plt.title("Liczba zdjęć w każdej kategorii")
plt.show()

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_gen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='category',
    target_size=(128, 128),
    class_mode='binary',
    batch_size=32
)

val_generator = val_gen.flow_from_dataframe(
    val_df,
    x_col='filename',
    y_col='category',
    target_size=(128, 128),
    class_mode='binary',
    batch_size=32,
    shuffle=False
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stop]
)

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title("Dokładność modelu")
plt.xlabel("Epoka")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title("Strata modelu")
plt.xlabel("Epoka")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.show()

import numpy as np

# pliki z walidacji i prawdziwe etykiety
val_generator.reset()
predictions = model.predict(val_generator, verbose=1)
predicted_classes = (predictions > 0.5).astype("int32").flatten()
true_classes = val_generator.classes
filenames = val_generator.filenames
labels = val_generator.class_indices  # {'cat': 0, 'dog': 1}
inv_labels = {v: k for k, v in labels.items()}

# błędnie sklasyfikowane
wrong = np.where(predicted_classes != true_classes)[0]
print(f"Liczba błędnych klasyfikacji: {len(wrong)}")

# kilka przykładów
plt.figure(figsize=(12, 12))
for i, idx in enumerate(wrong[:16]):  # pokaż max 16
    filepath = filenames[idx]
    true_label = inv_labels[true_classes[idx]]
    pred_label = inv_labels[predicted_classes[idx]]
    img = plt.imread(filepath)

    plt.subplot(4, 4, i + 1)
    plt.imshow(img)
    plt.title(f"Prawda: {true_label}\nPrzewidziano: {pred_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Macierz błędów
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[inv_labels[0], inv_labels[1]],
            yticklabels=[inv_labels[0], inv_labels[1]])
plt.xlabel('Przewidywana etykieta')
plt.ylabel('Prawdziwa etykieta')
plt.title('Macierz błędów')
plt.show()