from setuptools import setup, find_packages

setup(
    name="llama_3_8B_encoder_decoder_transformer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "pandas",
        "tqdm",
        "accelerate",
        "peft",
        "bitsandbytes",
        "wandb",
        "huggingface_hub",
        "flash-attn",
        "numba",
    ],
    author="",
    author_email="",
    description="A custom encoder-decoder library based on the Llama 3 8B model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/christianThardy/llama_3_8B_encoder_decoder_transformer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)