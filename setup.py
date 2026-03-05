from setuptools import setup, find_packages

setup(
    name="SFHelium-Recon",
    version="0.0.1",
    description="SFHelium reconstruction tools",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "phiflow[jax]",
        "jax",
        "jaxlib",
        "numpy",
        "scipy",
        "matplotlib",
        "pyvista",
        "pandas",
        "optax"
    ],
)