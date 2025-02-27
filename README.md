# Machine Learning Hands-On Repository

This repository contains a collection of exercises and practice materials for machine learning classes.

## Curricular Unit
Machine Learning, Degree in Applied Data Science, Catholic University of Portugal, Braga, 2024-2025.

## Syllabus

| **Module** | **Topic**                                | **Lecture**                                                                     | **Exercises**                                                                      |
|------------|------------------------------------------|---------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| 1          | Course Introduction + Introduction to ML | [lecture](https://github.com/LCDA-UCP/ml-hands-on/tree/main/lectures/session01) | -                                                                                  |
| 2          | Data Preprocessing                       | [lecture](https://github.com/LCDA-UCP/ml-hands-on/tree/main/lectures/session02) | [exercises](https://github.com/LCDA-UCP/ml-hands-on/tree/main/exercises/session02) |
| 3          | Introduction to Git                      | [lecture](https://github.com/LCDA-UCP/ml-hands-on/tree/main/lectures/session03) | -                                                                                  |
| 4          | Data Scaling and Feature Selection       | [lecture](https://github.com/LCDA-UCP/ml-hands-on/tree/main/lectures/session04) | [exercises](https://github.com/LCDA-UCP/ml-hands-on/tree/main/exercises/session04) |
| 5          | Clustering                               | [lecture](https://github.com/LCDA-UCP/ml-hands-on/tree/main/lectures/session05) | [exercises](https://github.com/LCDA-UCP/ml-hands-on/tree/main/exercises/session05) |
| 6          | Practical Session                        | -                                                                               | [issues #1, #2 and #3](https://github.com/LCDA-UCP/ml-hands-on/issues)             |
| 7          | Dimensionality Reduction                 | [lecture](https://github.com/LCDA-UCP/ml-hands-on/tree/main/lectures/session06) | [exercises](https://github.com/LCDA-UCP/ml-hands-on/tree/main/exercises/session06) |
| 8          | ...                                      | -                                                                               | -                                                                                  |
| 9          | ...                                      | -                                                                               | -                                                                                  |
| 10         | ...                                      | -                                                                               | -                                                                                  |

## ML Package Challenges

| **Issue**                                              | **Topic**             | **Development**   | **Testing**       | **Review**        |
|--------------------------------------------------------|-----------------------|-------------------|-------------------|-------------------|
| [#1](https://github.com/LCDA-UCP/ml-hands-on/issues/1) | Data Management + I/O | @cleopatra-santos | @Dacni            | @D1ana-bd         |
| [#2](https://github.com/LCDA-UCP/ml-hands-on/issues/2) | Feature Selection     | @D1ana-bd         | @cleopatra-santos | @Dacni            |
| [#3](https://github.com/LCDA-UCP/ml-hands-on/issues/3) | Clustering            | @Dacni            | @D1ana-bd         | @cleopatra-santos |


## Setup

First, clone the repository from GitHub to your local machine. You can do this by running the following command in your terminal:

```bash
git clone https://github.com/LCDA-UCP/ml-hands-on.git
```

Next, navigate to the repository directory:

```bash
cd ml-hands-on
```

Finally, install the required dependencies:

```bash
pip install -r requirements.txt
```

**Note:** If you are using a virtual environment (which is strongly recommended), make sure it is activated before running the above command.

You can now commit and push your changes to the repository.

## Create a branch with your name

To avoid conflicts with other students, create a new branch with your name before starting to work on the repository.
The name of the branch should your first name initial followed by your last name, all in lowercase (e.g., 'jcorreia').

You can create a new branch by running the following command:

```bash
git checkout -b <branch-name>
```

This will create a new branch with your name and switch to it.

## Pushing changes to your branch

To push your changes to your branch, run the following command in your terminal:

```bash
git push origin <branch-name>
```

This will push your changes to your branch on GitHub. 

## Merging changes from the main branch to your branch

To ensure your branch is up-to-date with the latest changes from the main branch:

1. Fetch the latest changes from the remote repository:

```bash
git fetch origin
```

2. Merge the changes from the main branch into your branch:

```bash
git merge origin/main
```

This will update your branch with the latest changes from the main branch.

3. Push the changes to your branch on GitHub:

```bash
git push origin <branch-name>
```

This will push the changes to your branch on GitHub.
