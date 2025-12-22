"""Setup script for the gym_2048 package."""
from setuptools import find_packages, setup

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(name='gym_2048',
      version = '0.0.1',
      packages=find_packages(),
      install_requires = [
            'absl-py',
            'gymnasium',
            'numpy',
            'pillow',
            'matplotlib',
            'numba'
      ],
      author='Your Name',
      author_email='your.email@example.com',
      description='A simple gym environment to play the 2048 game.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/your-username/gym_2048',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12',
          'Programming Language :: Python :: 3.13',
          'Programming Language :: Python :: 3.14',
      ],
)
