<a name="readme-top"></a>

<h3 align="center">HyDec</h3>


<p>
    <div align="center">
        This project is an implementation of the HyDec and HierDec decomposition approaches as described in the papers: 
        <ul align="center">
          <li>HyDec: <a href="https://doi.org/10.1007/978-3-031-20984-0_14">Combining Static and Dynamic Analysis to Decompose Monolithic Application into Microservices</a></li>
          <li>HierDec: <a href="https://doi.org/10.1145/3530019.3530040">A Hierarchical DBSCAN Method for Extracting Microservices from Monolithic Applications</a></li>
        </ul>
    </div>
</p>



<!-- TABLE OF CONTENTS -->

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#preparing-the-input-data">Preparing the input data</a></li>
        <li><a href="#decomposing-the-monolithic-application-using-hydec">Decomposing the monolithic application using HyDec</a></li>
      </ul>
    </li>
    <li>
      <a href="#advanced-usage">Advanced usage</a>
      <ul>
        <li><a href="#using-the-analysis-and-parsing-tools-as-grpc-services">Using the analysis and parsing tools as gRPC services</a></li>
        <li><a href="#using-hierdec-as-a-grpc-server">Using HierDec as a gRPC server</a></li>
        <li><a href="#using-hydec-as-a-python-module">Using HyDec as a Python module</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#authors">Authors</a></li>
    <li><a href="#version-history">Version History</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
# About The Project

This project is an implementation of the HyDec and HierDec decomposition approaches as described in the papers: 
- HyDec: "Combining Static and Dynamic Analysis to Decompose Monolithic Application into Microservices" [[1]](#1)
- HierDec: "A Hierarchical DBSCAN Method for Extracting Microservices from Monolithic Applications" [[2]](#2)

HierDec is a decomposition tool that analyzes the source code of a monolithic Java application through static analysis and a TFIDF based pipeline and suggests the 
recommended microservices for each class in the system using a hierarchical version of the DBSCAN algorithm. 

HyDec is an extension of the HierDec approach that combines the static analysis results with the dynamic analysis results to improve the quality of the decomposition.

The current implementation is a generalization of the approach proposed in HyDec enabling the use of various representations of the source code (with or instead of the static and dynamic analysis results). We provide the default configurations which correspond to the HyDec and HierDec papers. 
 
In order to replicate the results of the approaches, the static analysis results have to be generated using another tool. This implementation is compatible with 
the packages [decomp-java-analysis-service](https://github.com/khaledsellami/decomp-java-analysis-service) and [decomp-parsing-service](https://github.com/khaledsellami/decomp-parsing-service.git) that handle the static analysis part of the process.
Otherwise, it is possible to use your own tool but the input has to conform to the required types and structure.



# Getting Started

## Prerequisites

The main requirements are:
* Python 3.9 or higher

## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/khaledsellami/decomp-hydec.git
   ```
2. Install hydec as a Python module:
   ```sh
   cd decomp-hydec/
   pip install -e .
   ```
   Or you can install only the required libraries:
   ```sh
    pip install -r requirements.txt
    ```



<!-- USAGE EXAMPLES -->
# Usage

## Preparing the input data
In order to use HierDec, the source code of the monolithic application has to be analyzed and parsed to extract the structural and semantic dependencies as described in the paper. You can use our analysis and parsing tools or use your own tools to generate the required data:
### Using the analysis and parsing tools:
The details of how to use these tools and generate the analysis results with them can be found in their corresponding repositories: [decomp-java-analysis-service](https://github.com/khaledsellami/decomp-java-analysis-service) and [decomp-parsing-service](https://github.com/khaledsellami/decomp-parsing-service.git).

### Using a third-party analysis tool:

When using CLI to generate class decompositions with HyDec or HierDec, the folder that contains the static and/or analysis results should contain the following structure:
```text
   path_to_my_data/
   ├── dynamic_data/
   │   ├── class_names.json: a list of strings representing the list of class names in their order in class_calls.npy
   │   └── class_calls.npy: a numpy MxM matrix representing the calls from each class to the others in the execution traces where M is the number of classes
   ├── semantic_data/
   │   ├── class_names.json: a list of strings representing the list of class names in their order in tfidf.npy
   │   └── class_tfidf.npy: a numpy NxV matrix representing the TF-IDF vectors of each class where N is the number of classes and V is the size of the vocabulary
   └── structural_data/
       ├── class_calls.npy: a numpy NxN matrix representing the calls from each class to the others where N is the number of classes
       └── class_names.json: a list of strings representing the list of class names in their order in class_calls.npy
```

## Decomposing the monolithic application using HyDec
The decomposition of the monolithic application can be done using CLI:
```sh
   python main.py decompose your_app_name --dynamic path_to_my_data/dynamic_data \
                                          --semantic path_to_my_data/semantic_data \
                                          --structural path_to_my_data/dynamic_data \
                                          --hyperparams /path/to/hyperparams/file.json \
                                          --approach hyDec
```
The file [hyperparameter_input_example.json](docs%2Fhyperparameter_input_example.json) contains an example of the hyperparameters file that can be used to configure the decomposition process.

The output will be saved in the folder: `./logs/your_app_name/the_experiment_name`


# Advanced usage
## Using the analysis and parsing tools as gRPC services
The analysis and parsing tools can be used as gRPC services. In this case, you do not need to prepare the data for HierDec. You can use the following command to launch the decomposition:
```sh
   python main.py decompose your_app_name --repo /path/or/github/link/to/source/code \
                                          --hyperparams /path/to/hyperparams/file.json \
                                          --approach hierDec
``` 
Keep in mind, this approach only works for HierDec and is not possible with HyDec because generating dynamic analysis results is required and is not automated.

## Using HierDec as a gRPC server
HierDec can be used as a gRPC server and its API can be consumed to generate the decompositions. You can start the server using the following command:
```sh
   python main.py start
```

For more details about the API, you can inspect the protobuf file "hydec.proto" in the [proto/hydec](protos%2Fhydec) folder.

## Using HyDec as a Python module
For more flexibility and advanced usage (such as using your own representations of the source code), HyDec can be used as a Python module and customized to fit your needs through the [HybridDecomp](hydec/hybridDecomp.py) class. 
In order to use your own representations of the source code and integrate them into the HybridApproach pipeline, you will need to implement your own Analysis class based on the [AbstractAnalysis](hydec%2Fanalysis%2FabstractAnalysis.py) class. The class should implement the feature aggregation and the class/method similarity functions. 
For examples to how to use the module, you can inspect the implementations like [TfidfAnalysis.py](hydec%2Fanalysis%2FtfidfAnalysis.py).



<!-- ROADMAP -->
# Roadmap
* Improve the documentation of this module
* Add more analysis representations
* Add an option to use the parsing service as a Python module
* Improve the CLI

<!-- AUTHORS -->
# Authors

Khaled Sellami - [khaledsellami](https://github.com/khaledsellami) - khaled.sellami.1@ulaval.ca

<!-- VERSION -->
# Version History

* 1.3.4
    * Initial Public Release


<!-- CITATION -->
# Citation
If this work was useful for your research, please consider citing it:
```bibtex
@inproceedings{hydec,
author="Sellami, Khaled
and Saied, Mohamed Aymen
and Ouni, Ali
and Abdalkareem, Rabe",
editor="Troya, Javier
and Medjahed, Brahim
and Piattini, Mario
and Yao, Lina
and Fern{\'a}ndez, Pablo
and Ruiz-Cort{\'e}s, Antonio",
title="Combining Static and Dynamic Analysis to Decompose Monolithic Application into Microservices",
booktitle="Service-Oriented Computing",
year="2022",
publisher="Springer Nature Switzerland",
address="Cham",
pages="203--218",
isbn="978-3-031-20984-0",
url = {https://doi.org/10.1007/978-3-031-20984-0_14},
doi = {10.1007/978-3-031-20984-0_14},
}

@inproceedings{hierdec,
author = {Sellami, Khaled and Saied, Mohamed Aymen and Ouni, Ali},
title = {A Hierarchical DBSCAN Method for Extracting Microservices from Monolithic Applications},
year = {2022},
isbn = {9781450396134},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3530019.3530040},
doi = {10.1145/3530019.3530040},
booktitle = {Proceedings of the 26th International Conference on Evaluation and Assessment in Software Engineering},
pages = {201–210},
numpages = {10},
keywords = {Clustering, Legacy decomposition, Microservices, Static Analysis},
location = {Gothenburg, Sweden},
series = {EASE '22}
}
```


<!-- LICENSE -->
# License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.



<!-- REFERENCES -->
# References

<a id="1">[1]</a> 
Sellami, Khaled, Mohamed Aymen Saied, Ali Ouni, and Rabe Abdalkareem. ‘Combining Static and Dynamic Analysis to Decompose Monolithic Application into Microservices’. In Service-Oriented Computing, edited by Javier Troya, Brahim Medjahed, Mario Piattini, Lina Yao, Pablo Fernández, and Antonio Ruiz-Cortés, 203–18. Cham: Springer Nature Switzerland, 2022. https://doi.org/10.1007/978-3-031-20984-0_14.

<a id="2">[2]</a> 
Sellami, Khaled, Mohamed Aymen Saied, and Ali Ouni. ‘A Hierarchical DBSCAN Method for Extracting Microservices from Monolithic Applications’. In Proceedings of the 26th International Conference on Evaluation and Assessment in Software Engineering, 201–10. EASE ’22. New York, NY, USA: Association for Computing Machinery, 2022. https://doi.org/10.1145/3530019.3530040.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

