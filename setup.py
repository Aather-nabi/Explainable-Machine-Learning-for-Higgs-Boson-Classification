from setuptools import find_packages, setup

# This function dynamically reads your requirements.txt
def get_requirements(file_path: str) -> list[str]:
    """Reads the requirements.txt file and returns a list of dependencies."""
    requirements = []
    # Adding 'utf-8-sig' automatically strips the hidden Windows BOM characters
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        requirements = file.readlines()
        # Remove the newline characters
        requirements = [req.replace("\n", "") for req in requirements]
        
        # We don't want setup.py to try and install '-e .' as a package from PyPI
        if "-e ." in requirements:
            requirements.remove("-e .")
            
    return requirements

setup(
    name='ML_Project',
    version='0.1.0',
    author='aather nabi',
    author_email='nabiaatir1@gmail.com',
    description='Machine learning project and exploratory data analysis environment',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)