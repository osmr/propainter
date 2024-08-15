from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='propainter',
    description='Streaming ProPainter',
    url='https://github.com/osmr/propainter',
    author='Oleg Sémery',
    author_email='osemery@gmail.com',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    keywords='video-inpaining propainter pytorch',
    packages=find_packages(exclude=['*.others']),
    install_requires=['pytorchcv', 'opencv-python', 'pillow', 'scipy'],
    python_requires='>=3.10',
    include_package_data=True,
)
