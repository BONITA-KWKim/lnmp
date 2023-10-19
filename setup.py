import setuptools

setuptools.setup(
    name="Lymph Node Metastasis Prognosis",
    version="0.0.1",
    author="Titedios kyugnwon.KIM",
    author_email="kwkim12@mtsco.co.kr",
    description="This project is for prognosis of lymph node meatastasis",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(include=['lnmp', 'lnmp.src.*']),
    python_requires=">=3.9",
)
