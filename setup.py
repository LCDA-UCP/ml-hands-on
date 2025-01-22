from setuptools import setup, find_packages

setup(
    name='ml-hands-on',
    version='0.0.1',
    python_requires='>=3.10',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    zip_safe=False,
    install_requires=['numpy', 'pandas', 'scipy'],
    author='',
    author_email='',
    description='Machine Learning Course @ Catholic University of Portugal, Braga, 2025',
    license='MIT',
    keywords='',
)