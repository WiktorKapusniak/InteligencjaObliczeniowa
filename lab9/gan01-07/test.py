from matplotlib import pyplot as plt
from keras.models import load_model
from numpy.random import randn

model = load_model('generator_model_075.h5')
vector = randn(100).reshape(1, 100)
image = model.predict(vector)[0, :, :, 0]

plt.imshow(image, cmap='gray_r')
plt.axis('off')
plt.show()