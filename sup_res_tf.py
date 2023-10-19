import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

# Define a simple CNN model
class SuperResolutionModel(models.Model):
    def __init__(self):
        super(SuperResolutionModel, self).__init__()
        self.conv1 = layers.Conv2D(64, (5, 5), padding='same', activation='relu')
        self.conv2 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.conv3 = layers.Conv2D(3, (3, 3), padding='same')  # No activation

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

# Create a model instance
model = SuperResolutionModel()

# Loss and optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_object = tf.keras.losses.MeanSquaredError()

def preprocess(features):
    # Resize to low resolution
    features['image'] = tf.image.resize(features['image'], (32, 32))
    features['image'] = tf.cast(features['image'], tf.float32) / 255.0

    # Create high-resolution targets
    features['high_res'] = tf.image.resize(features['image'], (128, 128))
    return features['image'], features['high_res']

dataset = tfds.load('cifar10', split='train')
dataset = dataset.map(preprocess).batch(64).shuffle(1024)

# Training process
@tf.function
def train_step(low_res, high_res):
    with tf.GradientTape() as tape:
        predictions = model(low_res, training=True)
        loss = loss_object(high_res, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

num_epochs = 5
for epoch in range(num_epochs):
    for batch, (low_res, high_res) in enumerate(dataset):
        loss = train_step(low_res, high_res)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.6f}')

print('Training finished.')
