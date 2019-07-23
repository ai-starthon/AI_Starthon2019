#nsml: nsml/default_ml:cuda9_torch1.0
from distutils.core import setup
import setuptools

setup(
    name='20_scls_stt',
    version='1.1',
    install_requires=[
        'torch>=1.1.0',
        'numpy',
        'scipy',
        'python_speech_features',
        'librosa==0.6.2'
    ]
)
