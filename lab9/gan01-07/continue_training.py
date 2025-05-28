from keras.models import load_model
from gan07 import define_discriminator, define_gan, load_real_samples, train

g_model = load_model('generator_model_003.h5')  # np po 3 epokach

d_model = define_discriminator()
gan_model = define_gan(g_model, d_model)

dataset = load_real_samples()
latent_dim = 100 
train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=256)