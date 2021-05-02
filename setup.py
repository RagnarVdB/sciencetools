from setuptools import setup
import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name='fysfuncties',
    version='0.0.1',
    py_modules=['fysfuncties'],
    description='Nuttige functies voor practica',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/ragnarvdb/fysfuncties',
    author='RagnarVdB',
    author_email='59337368+RagnarVdB@users.noreply.github.com',
    license="MIT",
    install_requires=[
        'numpy',
        'scipy',
        'sympy',
        'pandas'
    ]
)