from setuptools import setup, find_packages

setup(
    name="agent-operations-tracker",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'requests>=2.28.0',
        'aiohttp>=3.8.0',
        'python-dateutil>=2.8.0'
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A library for tracking AI Agent operations and metrics",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/agent-operations-tracker",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
) 