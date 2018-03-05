from setuptools import setup, find_packages

setup(name='particle_env',
      version='0.0.1',
      description='2D particle environment for openai gym', 
      author='Kazuya Otani',
      author_email='kazuotani14@gmail.com',
      packages=find_packages(),
      install_requires=['gym', 'numpy']
)