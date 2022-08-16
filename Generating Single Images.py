from keras.models import load_model
from numpy.random import randn
import matplotlib.pyplot as plt


"""
Load model
"""
path = 'C:/0-MODELS - bacterial spot/generator_model_290.h5'
model = load_model(path)


"""
Generate points in latent space as 
input for the generator
"""


def generate_latent_points(latent_dim, n_samples):
    """Generate points in the latent space"""
    x_input = randn(latent_dim * n_samples)
    """Reshape into a batch of inputs for the network"""
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


"""
Create and save plots of generated images
"""


def save_plot(examples, n):
    """Plot each image"""
    for i in range(n):
        plt.axis('off')
        plt.imshow(examples[i])
        file_name = f'bs_image_{i}.png'
        plt.savefig(file_name,
                    bbox_inches='tight',
                    pad_inches=0)


"""
Generate n Single Images
"""


def generate_n_single_images(n):
    """
    Generate images
    """
    latent_points = generate_latent_points(100, n)
    X = model.predict(latent_points)
    """
    Scale from [-1, 1] to [0, 1]
    """
    X = (X + 1) / 2.0
    """
    Create the plots
    """
    save_plot(X, n)


generate_n_single_images(15)
