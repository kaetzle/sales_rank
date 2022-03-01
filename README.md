#Sales_rank
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This project is a part of code to answer a Doctolib Use Case : how to rank prospects from previous data ?


It's organised into the following parts:
* sales_rank/\_\_main__.py which is the core of the project
* a requirements.txt which lists all required libraries to launch the project
* this README.md


<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

This project use Python3.7.

To run this project, you need to install all libraries listed in the file requirements.txt :

  ```
  python3.7 -m pip install requirements.txt
  ```
<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To get started, you'll need to clone the repository, then install the requirements. After you'll need to first run the initialisation to create your keywords file of the train dataset. Afterwards you can test it with a question from the Squad dataset to find the best matching context.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/kaetzle/sales_rank
   ```
2. Install required libraries :
   ```
   python3.7 -m pip install requirements.txt
   ```
<p align="right">(<a href="#top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

You can train a model from  a preprocessed csv containing your features : 
<LogisticRegression or RandomForest> "
                    "<Name of the labels columns> <Name of the feature to remove for the ablation study>
   ```
   python3.7 __main__.py --train path/to/features/file.csv model_architecture_name columns_of_labels
   ```
   ```
   python3.7 __main__.py --train path/to/features/file.csv LogisticRegression opportunity_stage_after_30_days
   ```
For now, only <LogisticRegression or RandomForest> are available for model_architecture


You can also run an ablation study on the same architecture model and with one dropped Feature :
   ```
   python3.7 __main__.py --ablation path/to/features/file.csv model_architecture_name columns_of_labels feature_to_drop
   ```
   ```
   python3.7 __main__.py --ablation path/to/features/file.csv LogisticRegression opportunity_stage_after_30_days has_website
   ```
<!-- CONTACT -->
## Contact

Project Link: [https://github.com/kaetzle/sales_rank](https://github.com/kaetzle/sales_rank)

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
