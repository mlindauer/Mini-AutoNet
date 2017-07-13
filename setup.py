# -*- encoding: utf-8 -*-
import setuptools

with open("mini_autonet/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

with open('./requirements.txt') as fh:
    requirements = fh.read()
requirements = requirements.split('\n')
requirements = [requirement.strip() for requirement in requirements]

setuptools.setup(
    name='mini_autonet',
    description='Automated machine learning.',
    version=version,
    packages=setuptools.find_packages(exclude=['test']),
    install_requires=requirements,
    scripts=[],
    author_email='lindauer@cs.uni-freiburg.de',
    license='BSD',
    platforms=['Linux'],
    tests_require=['nose', ],
    test_suite='nose.collector',
    classifiers=[
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ]
)
