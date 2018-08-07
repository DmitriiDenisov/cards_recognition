import pandas as pd
from keras.preprocessing import image

train_datagen = image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    directory=r"J:\Projects\CardsMobile\data\train",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())

df = pd.DataFrame(list(labels.items()), columns=['class_index', 'class_name'])
df.to_csv('lasels.csv', index=False)
